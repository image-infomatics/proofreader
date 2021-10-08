import torch
import gc
import os
from .all import readable_bytes
from torch.utils.data import DistributedSampler, Dataset
from torch.utils.data.sampler import Sampler
from typing import Iterator, List, Optional, Union
import numpy as np
from operator import itemgetter


def get_all_live_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensors.append(obj)
        except:
            pass

    return tensors


def print_all_live_tensors():
    print('===========================TENSORS===========================')
    tensors = get_all_live_tensors()
    for t in tensors:
        print(t.dtype, t.device, type(t), t.size())


def count_all_live_tensors():
    count = 0
    mem = 0
    tensors = get_all_live_tensors()
    for t in tensors:
        count += 1
        mem += t.element_size() * t.nelement()

    return count, readable_bytes(mem)


def get_cpu_count():
    cpu_count = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            cpu_count = len(os.sched_getaffinity(0))
            return cpu_count
        except:
            pass

    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count

    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
    except:
        pass

    print('could not get cpu count, returning 1')

    return 1


def save_model(model, path, epoch=None, optimizer=None, loss=None):
    """ Save trained network as file

    Args:
        model (nn.Module): current model
        path (str): file path of saved model
        epoch (int): current epoch
        optimizer (Optimizer): current optimizer
        loss (float): current loss
    """

    path = os.path.join(path, 'checkpoints')

    # make folder if doesnt exist
    if not os.path.exists(path):
        os.makedirs(path)

    fname = os.path.join(path, f'{epoch}.ckpt')
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': loss, }
    torch.save(state, fname)


def load_model(model, path, optimizer=None, map_location=None, strict=True):
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    return model


def weighted_binary_cross_entropy(output, target, weights=None):

    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
            weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + \
            (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py#L135
class BalanceClassSampler(Sampler):
    """Allows you to create stratified sample on unbalanced classes.
    Args:
        labels: list of class label for each elem in the dataset
        mode: Strategy to balance classes.
            Must be one of [downsampling, upsampling]
    Python API examples:
    .. code-block:: python
        import os
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.data import ToTensor, BalanceClassSampler
        from catalyst.contrib.datasets import MNIST
        train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
        train_labels = train_data.targets.cpu().numpy().tolist()
        train_sampler = BalanceClassSampler(train_labels, mode=5000)
        valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
        loaders = {
            "train": DataLoader(train_data, sampler=train_sampler, batch_size=32),
            "valid": DataLoader(valid_data, batch_size=32),
        }
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        runner = dl.SupervisedRunner()
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            logdir="./logs",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
    """

    def __init__(self, labels: List[int], mode: Union[str, int] = "downsampling"):
        """Sampler initialisation."""
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {label: (labels == label).sum()
                             for label in set(labels)}

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist() for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = mode if isinstance(
                mode, int) else max(samples_per_class.values())
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length

# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py#L684


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)
