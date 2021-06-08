import random
import torch
import numpy as np
from pathlib import Path
from transformers import AdamW, AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream
from LL4LM.trainers.lifelong_trainer import LifelongTrainer

import wandb
import logging
log = logging.getLogger(__name__)


class MultitaskTrainer(LifelongTrainer):

    def __init__(self, config: dict):
        super().__init__(config)

    def load_data(self):
        config = self.config.data
        self.dataset_names = self.config.datastream
        datastream = DataStream(self.dataset_names, split="train_split")
        teststream = DataStream(self.dataset_names, split="test_split")
        gradstream = DataStream(self.dataset_names, split="test_split")
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
            shuffle_examples=True # shuffle the datastream to enforce i.i.d.
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
        log.info(f"Loaded shuffled (i.i.d.) Data Stream")