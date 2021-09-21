import math
import pprint
from dataclasses import dataclass

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

import torch.nn.functional as F

from proofreader.data.splitter import SplitterDataset
from proofreader.model.classifier import Classifier
from proofreader.model.pointnet import PointNetCls
from proofreader.utils.all import get_cpu_count


@dataclass
class DatasetConfig:
    dataset: str = 'cremi'
    num_classes: int = 2
    num_slices = [1, 4]
    radius: int = 96
    context_slices: int = 6
    num_points: int = 1024
    batch_size: int = 8
    val_split: float = 0.15


@dataclass
class ModelConfig:
    model: str = 'pointnet'
    loss = str = 'nll'
    dim: int = 256
    learning_rate: float = 1e-3


@dataclass
class TrainerConfig:
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()

    def toString(self):
        d = pprint.pformat(self.dataset)
        m = pprint.pformat(self.model)
        return f'NAME\n{self.name}\nDATASET{d}\nMODEL\n{m}\n'


def build_dataloader_from_config(config: DatasetConfig, vols):
    dataset = SplitterDataset(
        vols, config.num_slices, config.radius, config.context_slices, num_points=config.num_points, normalize=True, open_vol=True, shuffle=True, torch=True)

    split = math.floor(len(dataset)*config.val_split)
    train_split = list(range(0, split))
    val_split = list(range(split, len(dataset)))
    ds_train = Subset(dataset, train_split)
    ds_val = Subset(dataset, val_split)

    print(f'# train: {len(ds_train)}, # val: {len(ds_val)}')

    cpus = get_cpu_count()

    dl_train = DataLoader(
        dataset=ds_train, batch_size=config.batch_size, num_workers=cpus, drop_last=True)
    dl_val = DataLoader(
        dataset=ds_val, batch_size=config.batch_size, num_workers=cpus, drop_last=True)

    return dl_train, dl_val


def build_model_from_config(model_config: ModelConfig, dataset_config: DatasetConfig):
    if model_config.loss == 'nll':
        loss = F.nll_loss
    elif model_config.loss == 'ce':
        loss = F.cross_entropy

    if model_config.model == 'pointnet':
        backbone = PointNetCls(num_points=dataset_config.num_points)

    model = Classifier(backbone=backbone, loss=loss,
                       learning_rate=model_config.learning_rate)
    return model


def build_trainer_from_config(config: TrainerConfig, args):
    logger = TensorBoardLogger(save_dir='logs/', name=args.config)

    trainer = Trainer.from_argparse_args(
        args, deterministic=config.deterministic, logger=logger)
    return trainer


def get_config(name):
    for c in CONFIGS:
        if c.name == name:
            return c
    raise ValueError(f'config {name} not found.')


CONFIGS = [
    ExperimentConfig('default'),
]