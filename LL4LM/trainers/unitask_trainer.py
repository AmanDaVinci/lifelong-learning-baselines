import gc
import random
import torch
import json
import numpy as np
from pathlib import Path
from transformers import AdamW, AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream
from LL4LM.trainers.trainer import Trainer
from LL4LM.utils.gradients import sequential_gradient_interference

import wandb
import logging
log = logging.getLogger(__name__)


class UnitaskTrainer(Trainer):

    def __init__(self, config: dict):
        super().__init__(config)

    def load_data(self):
        config = self.config.data
        self.dataset_names = self.config.datastream
        datastream = DataStream(self.dataset_names, split="train_split")
        teststream = DataStream(self.dataset_names, split="test_split")
        if config.shuffle:
            datastream.shuffle_datasets(self.config.seed)
        datastream.resize_datasets(config.dataset_size)
        teststream.limit_datasets(config.testset_size)
        examples = datastream.sample_examples(config.n_samples_each_dataset)
        wandb.log({"Sampled_Examples": wandb.Table(dataframe=examples)}, step=0)
        wandb.log({"Data_Stream": wandb.Table(dataframe=datastream.summary())}, step=0)
        save_path = self.output_dir/"testsets" 
        teststream.save(save_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        self.dataloaders = datastream.get_dataloader(
            self.tokenizer, 
            batch_size=config.batch_size,
            concatenate=False, # dataloader for each dataset
            shuffle_examples=True # shuffle each dataset
        )
        self.testloaders = teststream.get_dataloader(
            self.tokenizer, 
            batch_size=config.test_batch_size,
            concatenate=False,
            shuffle_examples=False
        )
        log.info(f"Loaded each dataset separately from Data Stream")

    def run(self):
        batch_size = self.config.data.batch_size
        test_interval = self.config.test_interval
        train_grad_interval = self.config.train_grad_interval
        examples_seen = 0
        for name, dataloader, testloader in zip(self.dataset_names, self.dataloaders, self.testloaders):
            self.load_model()
            self.model.train()
            log.info(f"Start training new model on {name}")
            dataset_examples_seen = 0
            grads, nonzero_indices = None, None
            for i, batch in enumerate(dataloader):
                examples_seen += batch_size
                dataset_examples_seen += batch_size
                self.model.zero_grad()
                loss, acc = self.model.step(batch)
                loss.backward()
                self.opt.step()
                wandb.log(
                    {
                        f"train/{name}/loss": loss.item(),
                        f"train/{name}/accuracy": acc,
                        f"{name}_examples_seen": dataset_examples_seen,
                    }, 
                    step=examples_seen
                )
                if (i+1) % train_grad_interval == 0:
                    outputs = sequential_gradient_interference(self.model, grads, nonzero_indices)
                    grads, nonzero_indices, interference, overlap = outputs
                    wandb.log(
                        {
                            f"gradient/{name}/interference": interference,
                            f"gradient/{name}/overlap": overlap,
                            f"{name}_examples_seen": dataset_examples_seen,
                        }, 
                        step=examples_seen
                    )
            self.model.zero_grad()
            loss, acc = self.test(testloader)
            wandb.log({f"test/{name}/loss": loss.item()}, step=examples_seen)
            wandb.log({f"test/{name}/accuracy": acc}, step=examples_seen)
            log.info(f"Test Accuracy on {name}: {acc}")
            self.cleanup()
        log.info(f"Done training on all datasets.")

    def test(self, testloader):
        self.model.eval()
        losses, accuracies = [], [] 
        for batch in testloader:
            with torch.no_grad():
                loss, acc = self.model.step(batch)
            losses.append(loss.item())
            accuracies.append(acc)
        testset_loss = np.mean(losses)
        testset_accuracy = np.mean(accuracies)
        self.model.train()
        return testset_loss, testset_accuracy
    
    def cleanup(self):
        self.model.to("cpu")
        del(self.model)
        gc.collect()
        torch.cuda.empty_cache()