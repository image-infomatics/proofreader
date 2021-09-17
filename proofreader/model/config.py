import torch
import pprint
from proofreader.data.splitter import SplitterDataset
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset = 'cremi'
    num_slices = [1, 4]
    radius = 96
    context_slices = 8
    num_points = 1024


@dataclass
class ModelConfig:
    dim: int = 256


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()

    def toString(self):
        d = pprint.pformat(self.dataset)
        m = pprint.pformat(self.model)
        return f'NAME\n{self.name}\nDATASET{d}\nMODEL\n{m}\n'


def build_dataset_from_config(config: DatasetConfig, vols):
    return SplitterDataset(
        vols, config.num_slices, config.radius, config.context_slices, num_points=config.num_points)


def get_config(name):
    for c in CONFIGS:
        if c.name == name:
            return c
    raise ValueError(f'config {name} not found.')


CONFIGS = [
    ExperimentConfig('default'),
]
