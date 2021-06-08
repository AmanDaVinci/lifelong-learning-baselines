import random
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from LL4LM.datastreams import DataStream, load_dataset_ids
from LL4LM.models.lifelong_learner import LifelongLearner

import wandb
import logging
log = logging.getLogger(__name__)

class DatastreamScanner():

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.output_dir)/wandb.run.id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.set_seed(config.seed)
        self.load_data()
        self.load_model()

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info(f"Setting seed: {seed}")

    def load_data(self):
        config = self.config.data
        self.dataset_ids, _ = load_dataset_ids(
            multitask=config.multitask, 
            multilingual=config.multilingual, 
            multidomain=config.multidomain,
            custom=config.custom,
            shuffle=False # keep the ids ordered
        )
        datastream = DataStream(self.dataset_ids)
        if config.data_shuffle:
            datastream.shuffle_datasets(self.config.seed)
        datastream.limit_datasets(config.max_size_each_dataset)
        wandb.log({"Data_Stream": wandb.Table(dataframe=datastream.df())}, step=0)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        self.dataloaders = datastream.get_dataloader(
            self.tokenizer, 
            batch_size=config.batch_size,
            concatenate=False, # dataloader for each dataset
            shuffle_examples=True # shuffle each dataset
        )
        log.info(f"Loaded each dataset separately from Data Stream")

    def load_model(self): 
        config = self.config.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(config.base_model).to(self.device)
        log.info(f"Loaded {config.base_model} on {self.device}")

    def run(self):
        batch_size = self.config.data.batch_size
        for dataset_id, dataloader in zip(self.dataset_ids, self.dataloaders):
            self.model.eval()
            log.info(f"Start scanning {dataset_id}")
            labels, sequence_encodings, token_encodings, token_embeddings = [], [], [], []
            dataset_examples_seen = 0
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                label = batch.pop("label")
                with torch.no_grad():
                    token_embedding = self.model.embeddings.word_embeddings(batch["input_ids"])
                    outputs = self.model.forward(**batch, return_dict=True)
                pooler_output = torch.squeeze(outputs['pooler_output'])
                last_hidden_state = outputs['last_hidden_state']
                dataset_examples_seen += batch_size
                labels.append(label.detach().cpu().numpy())
                sequence_encodings.append(pooler_output.detach().cpu().numpy())
                token_encodings.append(last_hidden_state.detach().cpu().numpy())
                token_embeddings.append(token_embedding.detach().cpu().numpy())
            np.save(self.output_dir/f"{dataset_id}_labels.npy", np.concatenate(labels))
            np.save(self.output_dir/f"{dataset_id}_seq_enc.npy", np.concatenate(sequence_encodings))
            np.save(self.output_dir/f"{dataset_id}_tok_enc.npy", np.concatenate(token_encodings))
            np.save(self.output_dir/f"{dataset_id}_tok_emb.npy", np.concatenate(token_embeddings))
        log.info(f"Done scanning the datastream")