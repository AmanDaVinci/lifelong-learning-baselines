import time
import json
import torch
import random
import numpy as np
from pathlib import Path
from transformers import AdamW, AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream
from LL4LM.models.lifelong_learner import LifelongLearner
from LL4LM.trainers.trainer import Trainer
from LL4LM.utils.gradients import pairwise_gradient_similarity, sequential_gradient_interference

import wandb
import logging
log = logging.getLogger(__name__)


class LifelongTrainer(Trainer):

    def __init__(self, config: dict):
        super().__init__(config)
    
    def load_data(self):
        config = self.config.data
        self.dataset_names = self.config.datastream
        datastream = DataStream(self.dataset_names, split="train_split")
        teststream = DataStream(self.dataset_names, split="test_split")
        gradstream = DataStream(self.dataset_names, split="test_split")
        if config.shuffle:
            datastream.shuffle_datasets(self.config.seed)
        datastream.resize_datasets(config.dataset_size)
        teststream.limit_datasets(config.testset_size)
        gradstream.limit_datasets(config.gradset_size)
        examples = datastream.sample_examples(config.n_samples_each_dataset)
        wandb.log({"Sampled_Examples": wandb.Table(dataframe=examples)}, step=0)
        wandb.log({"Data_Stream": wandb.Table(dataframe=datastream.summary())}, step=0)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        self.dataloader = datastream.get_dataloader(
            self.tokenizer, 
            batch_size=config.batch_size,
            concatenate=True,
            shuffle_examples=False
        )
        self.testloaders = teststream.get_dataloader(
            self.tokenizer, 
            batch_size=config.test_batch_size,
            concatenate=False,
            shuffle_examples=False
        )
        self.gradloaders = gradstream.get_dataloader(
            self.tokenizer, 
            batch_size=config.grad_batch_size,
            concatenate=False,
            shuffle_examples=False
        )
        log.info(f"Loaded Data Stream")

    def run(self):
        self.model.train()
        self.model.zero_grad()
        batch_size = self.config.data.batch_size
        test_interval = self.config.test_interval
        test_grad_interval = self.config.test_grad_interval
        train_grad_interval = self.config.train_grad_interval
        examples_seen = 0
        grads, nonzero_indices = None, None
        index, head_weights, head_biases = [], [], []
        def _test_log():
            start = time.perf_counter()
            losses, accuracies = self.test()
            log.info(f"Testing done in {time.perf_counter()-start:.04f} secs")
            wandb.log(losses, step=examples_seen)
            wandb.log(accuracies, step=examples_seen)
            index.append(examples_seen)
            head_weights.append(self.model.head.weight.detach().cpu().numpy())
            head_biases.append(self.model.head.bias.detach().cpu().numpy())
            log.info(
                f"Test Accuracies at {examples_seen}:"\
                f"{json.dumps(accuracies, indent=4)}"
            )
        def _test_grad_log():
            start = time.perf_counter()
            grad_sim, grad_shared = pairwise_gradient_similarity(self.model, self.dataset_names, self.gradloaders)
            log.info(f"Grad sim measured in {time.perf_counter()-start:.04f} secs")
            wandb.log(grad_sim, step=examples_seen)
            wandb.log(grad_shared, step=examples_seen)
        _test_log()
        _test_grad_log()
        for i, batch in enumerate(self.dataloader):
            examples_seen += batch_size
            self.model.zero_grad()
            loss, acc = self.model.step(batch)
            loss.backward()
            self.opt.step()
            wandb.log({"train/loss": loss.item()}, step=examples_seen)
            wandb.log({"train/accuracy": acc}, step=examples_seen)
            if (i+1) % train_grad_interval == 0:
                outputs = sequential_gradient_interference(self.model, grads, nonzero_indices)
                grads, nonzero_indices, interference, overlap = outputs
                wandb.log({"gradient/interference": interference}, step=examples_seen)
                wandb.log({"gradient/overlap": overlap}, step=examples_seen)
            if (i+1) % test_interval == 0:
                _test_log()
            if (i+1) % test_grad_interval == 0:
                _test_grad_log()
        self.model.zero_grad()
        _test_log()
        _test_grad_log()
        np.save(self.output_dir/"head_weights.npy", np.concatenate(head_weights))
        np.save(self.output_dir/"head_biases.npy", np.concatenate(head_biases))
        np.save(self.output_dir/"index.npy", np.array(index))

    def test(self):
        self.model.eval()
        testset_losses, testset_accuracies  = {}, {}
        for name, testloader in zip(self.dataset_names, self.testloaders):
            losses, accuracies = [], [] 
            for batch in testloader:
                with torch.no_grad():
                    loss, acc = self.model.step(batch)
                losses.append(loss.item())
                accuracies.append(acc)
            testset_losses[f"test/{name}/loss"] = np.mean(losses)
            testset_accuracies[f"test/{name}/accuracy"] = np.mean(accuracies)
        self.model.train()
        return testset_losses, testset_accuracies
