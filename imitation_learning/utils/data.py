import logging
import math
from itertools import chain, accumulate

import torch
import numpy as np
from hyperject import singleton


class TensorLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i = 0
        self.idxs = None

    def __iter__(self):
        self.i = 0
        self.idxs = self.indices()
        return self

    def indices(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        return idxs

    def __next__(self):
        if self.i >= len(self.dataset):
            raise StopIteration()
        batch = self.dataset[self.idxs[self.i : self.i + self.batch_size]]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class ChainLoader:
    def __init__(self, *loaders):
        self.loaders = loaders
        self.length = sum([len(l) for l in loaders])

    def __len__(self):
        return self.length

    def __iter__(self):
        return chain(*self.loaders)


def _shuffled_getter(dataset):
    i = 0
    idxs = np.arange(len(dataset))
    np.random.shuffle(idxs)

    n = yield
    while True:
        assert i + n <= len(dataset), "End of dataset reached"
        batch = dataset[idxs[i : i + n]]
        i += n
        n = yield batch


class ConcatTensorLoader:
    def __init__(self, datasets, batch_size, collate_fn):
        self.batch_size = batch_size
        self.datasets = datasets
        self.lengths = [len(l) for l in datasets]
        self.sum_lengths = sum(self.lengths)
        self.collate_fn = collate_fn

        self.selection_iterator = None
        self.generators = None

    def __next__(self):
        selection = next(self.selection_iterator)
        counts = np.bincount(selection, minlength=len(self.datasets))
        batches = [d.send(n) for d, n in zip(self.generators, counts) if n > 0]
        return self.collate_fn(batches)

    def __iter__(self):
        self.reset()
        return self

    def reset(self):
        selection = np.array([i for i, n in enumerate(self.lengths) for _ in range(n)])
        np.random.shuffle(selection)
        r = range(self.batch_size, self.sum_lengths, self.batch_size)
        self.selection_iterator = iter(np.split(selection, r))

        self.generators = [_shuffled_getter(d) for d in self.datasets]
        for g in self.generators:
            g.send(None)  # Move forward until first yield

    def __len__(self):
        return (self.sum_lengths - 1) // self.batch_size + 1


class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def to_stack_size(self, stack_size):

        self.dataset = self.dataset.to_stack_size(stack_size)
        return self


def random_split(dataset, lengths, seed=None):

    s = sum([l for l in lengths if l > 0])
    lengths = [l if l > 0 else len(dataset) - s for l in lengths]
    assert sum(lengths) <= len(dataset), "Too many samples requested"
    if seed is not None:
        prev_seed = np.random.get_state()
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
        np.random.set_state(prev_seed)
    else:
        indices = np.random.permutation(len(dataset))
    indices = torch.tensor(indices, dtype=torch.long)

    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


def cycle(iterable):
    """Cycle iterable non-caching."""
    while True:
        for x in iterable:
            yield x


def stack_subsample(stack_size, sample_rate, trajectory):
    stacked = trajectory.stack(stack_size)
    sample = np.random.binomial(1, sample_rate, size=len(stacked[0])).astype(np.bool_)
    return [a[sample] for a in stacked]


TEST_RATIO = 0.2


@singleton
def dataset_stack_size_factory(container):
    return max(
        2,
        container.mode_unnormalized.stack_need,
        container.environment.expert_mode.stack_need,
    )


@singleton
def dataset_factory(container):
    dataset = container.environment.data(container.dataset_stack_size)

    logging.info(f"Dataset size {len(dataset)}")
    return dataset


@singleton
def dataloaders_factory(container):
    num_samples = container.config.num_samples
    print("number of training samples",num_samples)

    dataset = container.dataset
    if num_samples is None:
        num_test = int(max(640.0, len(dataset) * TEST_RATIO))
        num_samples = len(dataset) - num_test
    else:
        num_test = int(max(640.0, num_samples * TEST_RATIO))

    assert num_samples + num_test <= len(dataset), (
        f"Dataset of size {len(dataset)} does not "
        f"contain {num_samples + num_test} samples."
    )

    print("prepare data loader")

    # changes to Subset and store indices to index into the dataset
    train_dataset, test_dataset = random_split(
        dataset,
        [num_samples, num_test],
        seed=(0 if container.config.fixed_data else None),
    )
    train_dataloader = TensorLoader(
        train_dataset, batch_size=container.config.batch_size, shuffle=True
    )
    test_dataloader = TensorLoader(
        test_dataset, batch_size=container.config.batch_size, shuffle=True
    )

    # train_dataloader.dataset.dataset.actions to access elements
    return {"train": train_dataloader, "test": test_dataloader}


class EpochDataset:
    def __init__(self, report_freq, iterator):
        self.i = 0
        self.report_freq = report_freq
        self.iterator = cycle(iterator)

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return self.report_freq

    def __next__(self):
        if self.i == self.report_freq:
            raise StopIteration()

        self.i += 1
        return next(self.iterator)


class TransformedDataloader:
    def __init__(self, source, f):
        self.f = f
        self.source = source

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        iter(self.source)
        return self

    def __next__(self):
        return self.f(next(self.source))


class GrowingArray:
    def __init__(self, data: np.ndarray, buffer_size=1000):
        self._size = len(data)
        self._buffer = data
        self._expand(buffer_size)
        self._buffer_grow_size = buffer_size
        self.dtype = self._buffer.dtype

    @property
    def shape(self):
        return (self._size,) + self._buffer.shape[1:]

    def __len__(self):
        return self._size

    def __array__(self):
        return self._buffer[: self._size]

    def __getitem__(self, item):
        if isinstance(item, slice):
            item = slice(*item.indices(self._size))
            return self._buffer[item]
        return np.array(self)[item]

    def add(self, new_data):
        shortage = self._size + len(new_data) - len(self._buffer)
        if shortage > 0:
            self._expand(self._buffer_grow_size + shortage)

        self._buffer[self._size : self._size + len(new_data)] = new_data
        self._size += len(new_data)

    def _expand(self, n):
        zeros = np.zeros((n,) + self._buffer.shape[1:], dtype=self._buffer.dtype)
        self._buffer = np.concatenate((self._buffer, zeros), 0)


class GrowingTensor:
    def __init__(self, data: torch.Tensor, buffer_size=1000):
        self._size = len(data)
        self._buffer = torch.tensor(data)
        self._buffer_grow_size = buffer_size
        self.dtype = self._buffer.dtype
        self.device = self._buffer.device
        self._expand(buffer_size)

    @property
    def shape(self):
        return (self._size,) + self._buffer.shape[1:]

    def __len__(self):
        return self._size

    def __getitem__(self, *args, **kwargs):
        return self._buffer.__getitem__(*args, **kwargs)

    def add(self, new_data):
        shortage = self._size + len(new_data) - len(self._buffer)
        if shortage > 0:
            self._expand(self._buffer_grow_size + shortage)

        self._buffer[self._size : self._size + len(new_data)] = new_data
        self._size += len(new_data)

    def _expand(self, n):
        zeros = torch.zeros(
            (n,) + self._buffer.shape[1:], dtype=self._buffer.dtype, device=self.device
        )
        self._buffer = torch.cat((self._buffer, zeros), 0)
