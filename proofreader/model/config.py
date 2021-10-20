import math
import pprint
from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Subset
import torch
import torch.nn.functional as F


from proofreader.data.splitter import NeuriteDataset, SliceDataset
from proofreader.data.augment import Augmentor
from proofreader.model.pointnet import PointNet
from proofreader.model.curvenet import CurveNet
from proofreader.model.transnet import PointTransformerCls
from proofreader.utils.torch import SimpleDataset


@dataclass
class AugmentorConfig:
    shuffle: bool = True
    center: bool = True
    random_scale: bool = False
    normalize: tuple = (125, 1250, 1250)
    num_points: int = None


@dataclass
class DatasetConfig:
    dataset: str = 'slice'
    num_slices = 2
    radius: int = 96
    context_slices: int = 3
    num_points: int = 2048
    val_split: float = 0.15
    verbose: bool = False
    path: str = None
    truncate_canidates: int = 4
    scale: bool = False
    balance_samples: bool = False


@dataclass
class ModelConfig:
    model: str = 'curvenet'
    loss: str = 'bce'
    optimizer: str = 'AdamW'
    learning_rate: float = 1e-3
    num_classes: int = 2
    loss_weights: Tuple = (1.0, 1.0)


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


def load_dataset_from_disk(dataset_config, aug_config):

    # build augmentor
    train_augmentor = Augmentor(center=aug_config.center, shuffle=aug_config.shuffle, normalize=aug_config.normalize, num_points=dataset_config.num_points,
                                random_scale=aug_config.random_scale)

    test_augmentor = Augmentor(center=aug_config.center, shuffle=aug_config.shuffle, normalize=aug_config.normalize, num_points=dataset_config.num_points,
                               random_scale=False)

    path = dataset_config.path
    train_dataset = build_dataset_from_path(
        f'{path}_train.pt', truncate_canidates=dataset_config.truncate_canidates, merge_canidates=True, augmentor=train_augmentor)
    val_dataset = build_dataset_from_path(
        f'{path}_val.pt', truncate_canidates=dataset_config.truncate_canidates, merge_canidates=True, augmentor=test_augmentor)
    test_dataset = build_dataset_from_path(
        f'{path}_test.pt', truncate_canidates=dataset_config.truncate_canidates, merge_canidates=True, augmentor=test_augmentor)

    print(
        f'# train: {len(train_dataset)}, # val: {len(val_dataset)}, # test: {len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset


def build_dataset_from_path(path, truncate_canidates, merge_canidates, augmentor=None):

    x, y = torch.load(path)

    stats = {}
    # number of total neurites we attempt to merge
    stats['total_neurites'] = len(y)
    # number of total times we should merge before truncation of canidates
    stats['merge_opportunities'] = 0
    # the reducation in total possible success rate due to the inital truncation
    stats['truncate_succ_loss'] = 0
    stats['truncated_total_examples'] = 0
    stats['truncated_positive_examples'] = 0
    stats['truncated_negative_examples'] = 0
    stats['multi_merge'] = 0
    stats['truncated_multi_merge'] = 0
    for i in range(len(x)):
        mo = torch.sum(y[i][:, 0]).item()
        stats['merge_opportunities'] += mo
        stats['multi_merge'] += int(mo > 1)
        if truncate_canidates != 0:
            if len(y[i][truncate_canidates:]) > 0:
                stats['truncate_succ_loss'] += torch.sum(
                    y[i][truncate_canidates:][:, 0]).item()
            x[i] = x[i][:truncate_canidates]
            y[i] = y[i][:truncate_canidates]

        pe = (y[i][:, 0] == 1).count_nonzero().item()
        ne = (y[i][:, 0] == 0).count_nonzero().item()
        stats['truncated_total_examples'] += y[i].shape[0]
        stats['truncated_positive_examples'] += pe
        stats['truncated_negative_examples'] += ne
        stats['truncated_multi_merge'] += int(pe > 1)

    stats['truncate_succ_loss'] /= stats['merge_opportunities']
    stats['truncated_positive_examples'] /= stats['truncated_total_examples']
    stats['truncated_negative_examples'] /= stats['truncated_total_examples']

    if merge_canidates:
        x, y = torch.cat(x), torch.cat(y)

    ds = SimpleDataset(x, y, shuffle=False, augmentor=augmentor, info=stats)

    print(path, stats)

    return ds


def build_dataset_from_config(dataset_config: DatasetConfig, aug_config: AugmentorConfig, vols):

    # build augmentor
    augmentor = Augmentor(center=aug_config.center, shuffle=aug_config.shuffle,
                          rotate=aug_config.rotate, scale=aug_config.scale,
                          jitter=aug_config.jitter, normalize=aug_config.normalize)

    # build dataset
    if dataset_config.dataset == 'neurite':
        dataset = NeuriteDataset(
            vols, dataset_config.num_slices, dataset_config.radius, dataset_config.context_slices,
            num_points=dataset_config.num_points, Augmentor=augmentor, open_vol=True,  verbose=dataset_config.verbose, shuffle=True, torch=True)
    elif dataset_config.dataset == 'slice':
        dataset = SliceDataset(vols, dataset_config.num_slices, dataset_config.radius, dataset_config.context_slices,
                               num_points=dataset_config.num_points, Augmentor=augmentor, verbose=False)
        return dataset

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
    elif model_config.loss == 'bce':
        loss = F.cross_entropy

    # optimizer
    if model_config.model == 'pointnet':
        model = PointNet(num_points=dataset_config.num_points,
                         classes=model_config.num_classes)
    elif model_config.model == 'curvenet':
        model = CurveNet()
    elif model_config.model == 'transnet':
        model = PointTransformerCls(
            dim=3, num_classes=model_config.num_classes)

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
    ExperimentConfig('default'),

    ExperimentConfig('CURVENET_ns1_cs_3_m_true_t_4', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=True_ns=1_cs=3_r=96_np=4096_t=0', truncate_canidates=4)),

    ExperimentConfig('TRANSNET_ns1_cs_3_m_true_t_4', model=ModelConfig(model='transnet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=True_ns=1_cs=3_r=96_np=4096_t=0', truncate_canidates=4), augmentor=AugmentorConfig(num_points=2048)),

    ExperimentConfig('CURVENET_ns1_cs_3_m_false_t_4', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=False_ns=1_cs=3_r=96_np=4096_t=0', truncate_canidates=4)),

    ExperimentConfig('CURVENET_ns1_cs_3_m_false_t_4_scale', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=False_ns=1_cs=3_r=96_np=2048_t=0_sc=1000', truncate_canidates=4, scale=True), augmentor=AugmentorConfig(normalize=(6, 1000, 1000))),

    ExperimentConfig('CURVENET_ns1_cs_3_m_true_t_4_scale', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=True_ns=1_cs=3_r=96_np=2048_t=0_sc=1000', truncate_canidates=4, scale=True), augmentor=AugmentorConfig(normalize=(6, 1000, 1000))),

    ExperimentConfig('CURVENET_ns1_cs_3_m_true_t_4_1024', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=True_ns=1_cs=3_r=96_np=4096_t=0', truncate_canidates=4), augmentor=AugmentorConfig(num_points=1024)),

    ExperimentConfig('CURVENET_ns1_cs_3_m_true_t_4_balanced', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=True_ns=1_cs=3_r=96_np=4096_t=0', truncate_canidates=4, balance_samples=True)),

    ExperimentConfig('CURVENET_ns1_cs_2_m_true_t_4', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=True_ns=1_cs=2_r=96_np=2048_t=0_sc=None', truncate_canidates=4)),

    ExperimentConfig('CURVENET_ns1_cs_3_m_true_t_6', model=ModelConfig(model='curvenet'), dataset=DatasetConfig(
        path='/mnt/home/jberman/ceph/pf/dataset/DATASET_m=True_ns=1_cs=3_r=96_np=4096_t=0', truncate_canidates=6)),

]
