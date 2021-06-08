import random
import torch
import numpy as np
from pathlib import Path
from transformers import AdamW, AutoTokenizer, AutoModel
from LL4LM.models.lifelong_learner import LifelongLearner

import wandb
import logging
log = logging.getLogger(__name__)


class Trainer:

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

    def load_model(self): 
        config = self.config.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model = AutoModel.from_pretrained(config.base_model)
        self.model = LifelongLearner(base_model, device)
        log.info(f"Loaded {config.base_model} on {device}")
        self.opt = AdamW(
            self.model.parameters(), 
            weight_decay=config.wt_decay,
            lr=config.lr
        )
        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
