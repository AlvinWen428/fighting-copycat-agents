"""Utilities."""
import collections
from copy import copy

import numpy as np
from contextlib import contextmanager
import torch

from typing import TypeVar


T = TypeVar("T")


def call_single(model, x):
    """Wrapper to call model with single datapoint."""
    return model(x[None])[0]


@contextmanager
def no_grad():
    with torch.no_grad():
        yield


def environment_path_repr(e):
    import re

    return re.sub(r"[^A-Za-z0-9]+", "-", repr(e)).strip("-")


class EarlyStopper:
    def __init__(self, steps: int, num_its, report_freq, min_steps=0, max_steps=None, lr_decay_rate=None,
                 decay_times_threshold=3):
        self.stop_steps = steps
        self.num_its = num_its
        self.report_freq = report_freq
        self.steps_after_min = 0
        self.steps = 0
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.min = None
        self.stopped = False
        self.lr_decay_rate = lr_decay_rate
        self.decay_times_threshold = decay_times_threshold
        self.decay_times = 0

    def adjust_lr(self, policy):
        policy.learning_rate_decay(decay_rate=self.lr_decay_rate)
        self.decay_times += 1
        self.steps_after_min = 0
        print('adjust learning rate')

    def get_decay_times(self):
        if self.lr_decay_rate is not None:
            return self.decay_times
        else:
            return np.inf

    def get_all_decay_times(self):
        return self.decay_times_threshold

    def __call__(self, loss: float, policy, it) -> bool:
        if self.stop_steps is None:
            return False

        if self.stopped:
            raise RuntimeError("EarlyStopper can't be reused after stop.")

        self.steps += 1
        if self.steps < self.min_steps:
            return False

        if self.max_steps is not None and self.steps >= self.max_steps:
            self.stopped = True
            return True

        if self.min is None:
            self.min = loss
            return False

        if loss >= self.min:
            self.steps_after_min += 1
            if (
                self.steps_after_min >= self.stop_steps
                and self.steps_after_min >= self.min_steps
            ):
                if self.lr_decay_rate is not None and self.decay_times < self.decay_times_threshold:
                    self.adjust_lr(policy)
                    return False
                else:
                    self.stopped = True
                    return True
        else:
            self.min = loss
            self.steps_after_min = 0

        if ((it // self.report_freq) % (self.num_its // self.report_freq // self.decay_times_threshold) == 0) and \
                (self.lr_decay_rate is not None):
            self.adjust_lr(policy)

        return False


def dict_merge(dct, merge_dct, strict=False, path=()):
    dct = copy(dct)
    for k, v in merge_dct.items():
        new_path = path + (k,)

        if strict and k not in dct:
            raise RuntimeError(f"Path {new_path} not present is dct")
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(merge_dct[k], collections.Mapping)
        ):
            dct[k] = dict_merge(dct[k], merge_dct[k], strict=strict, path=new_path)
        else:
            dct[k] = merge_dct[k]
    return dct


