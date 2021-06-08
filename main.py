import wandb
import hydra
import importlib
from omegaconf import DictConfig, OmegaConf

from LL4LM.trainers.lifelong_trainer import LifelongTrainer
from LL4LM.trainers.multitask_trainer import MultitaskTrainer
from LL4LM.trainers.unitask_trainer import UnitaskTrainer
from LL4LM.trainers.replay_trainer import ReplayTrainer


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    dict_config = OmegaConf.to_container(config, resolve=True)
    with wandb.init(project="lifelong-learning", config=dict_config):
        module = importlib.import_module(config.trainer.module)
        trainer_cls = getattr(module, config.trainer.class_name)
        trainer = trainer_cls(config)
        trainer.run()


if __name__ == '__main__':
    main()
