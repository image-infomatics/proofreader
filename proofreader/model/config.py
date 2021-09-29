import math
import pprint
from dataclasses import dataclass

from torch.utils.data import Subset

import torch
import torch.nn.functional as F

from proofreader.data.splitter import SplitterDataset
from proofreader.data.augment import Augmentor
from proofreader.model.pointnet import PointNet
from proofreader.model.curvenet import CurveNet


@dataclass
class AugmentorConfig:
    shuffle: bool = False
    center: bool = False
    rotate: bool = False
    scale: bool = False
    jitter: bool = False
    normalize: tuple = (125, 1250, 1250)


@dataclass
class DatasetConfig:
    dataset: str = 'cremi'
    num_classes: int = 2
    num_slices = [1, 4]
    radius: int = 96
    context_slices: int = 6
    num_points: int = 1024
    val_split: float = 0.15
    epoch_multplier: int = 1
    verbose: bool = False


@dataclass
class ModelConfig:
    model: str = 'pointnet'
    loss: str = 'nll'
    optimizer: str = 'AdamW'
    dim: int = 256
    learning_rate: float = 1e-3


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    augmentor: AugmentorConfig = AugmentorConfig()

    def toString(self):
        d = pprint.pformat(self.dataset)
        m = pprint.pformat(self.model)
        return f'NAME\n{self.name}\nDATASET{d}\nMODEL\n{m}\n'


def build_dataset_from_config(dataset_config: DatasetConfig, aug_config: AugmentorConfig, vols):

    # build augmentor
    augmentor = Augmentor(center=aug_config.center, shuffle=aug_config.shuffle,
                          rotate=aug_config.rotate, scale=aug_config.scale,
                          jitter=aug_config.jitter, normalize=aug_config.normalize)

    # build dataset
    dataset = SplitterDataset(
        vols, dataset_config.num_slices, dataset_config.radius, dataset_config.context_slices,
        num_points=dataset_config.num_points, Augmentor=augmentor, open_vol=True, epoch_multplier=dataset_config.epoch_multplier,  verbose=dataset_config.verbose, shuffle=True, torch=True)

    # split into train and val
    split = math.floor(len(dataset)*dataset_config.val_split)
    train_split = list(range(split, len(dataset)-1))
    val_split = list(range(0, split))
    ds_train = Subset(dataset, train_split)
    ds_val = Subset(dataset, val_split)
    print(f'# train: {len(ds_train)}, # val: {len(ds_val)}')

    return ds_train, ds_val


def build_full_model_from_config(model_config: ModelConfig, dataset_config: DatasetConfig):
    # loss
    if model_config.loss == 'nll':
        loss = F.nll_loss
    elif model_config.loss == 'ce':
        loss = F.cross_entropy

    # optimizer
    if model_config.model == 'pointnet':
        model = PointNet(num_points=dataset_config.num_points)
    elif model_config.model == 'curvenet':
        model = CurveNet()

    # optimizer
    if model_config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=model_config.learning_rate, betas=(0.9, 0.999), weight_decay=0.05)

    return model, loss, optimizer


def get_config(name):
    for c in CONFIGS:
        if c.name == name:
            return c
    raise ValueError(f'config {name} not found.')


CONFIGS = [
    ExperimentConfig('pointnet'),
    ExperimentConfig('curvenet', model=ModelConfig(
        model='curvenet', loss='ce')),
    ExperimentConfig('pn_context_2', dataset=DatasetConfig(context_slices=2)),
    ExperimentConfig('pn_context_1', dataset=DatasetConfig(context_slices=1)),
    ExperimentConfig('pn_context_8', dataset=DatasetConfig(context_slices=8)),
    ExperimentConfig('cn_context_2', model=ModelConfig(
        model='curvenet', loss='ce'), dataset=DatasetConfig(context_slices=2)),
    ExperimentConfig('cn_context_1', model=ModelConfig(
        model='curvenet', loss='ce'), dataset=DatasetConfig(context_slices=1)),
    ExperimentConfig('cn_context_4', model=ModelConfig(
        model='curvenet', loss='ce'), dataset=DatasetConfig(context_slices=4)),
    ExperimentConfig('cn_context_4_aug', model=ModelConfig(
        model='curvenet', loss='ce'), dataset=DatasetConfig(context_slices=4),
        augmentor=AugmentorConfig(shuffle=True, center=True, rotate=True, scale=True)),
    ExperimentConfig('cn_context_4_aug_small', model=ModelConfig(
        model='curvenet', loss='ce'), dataset=DatasetConfig(context_slices=4),
        augmentor=AugmentorConfig(shuffle=True, center=True)),
    ExperimentConfig('cn_context_4_aug_mid', model=ModelConfig(
        model='curvenet', loss='ce'), dataset=DatasetConfig(context_slices=4),
        augmentor=AugmentorConfig(shuffle=True, center=True, scale=True)),
]
