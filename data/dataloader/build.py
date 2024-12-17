import torch
import os
from torch.utils.data import dataloader, distributed
import numpy as np
import random
from utils import PIN_MEMORY
from utils import RANK
from pathlib import Path
from data.dataset.data_utils import IMG_FORMATS, VID_FORMATS


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
