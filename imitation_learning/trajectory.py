from typing import List
import numpy as np
import torch

from imitation_learning.utils.data import GrowingArray


class Trajectory:
    keys = ["states", "actions", "rewards", "pixels"]

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        pixels: np.ndarray,
        info=None,
    ):
        n = len(states)
        assert n == len(actions) == len(rewards) == len(pixels)
        self.states = GrowingArray(states)
        # Ensure 1D actions
        self.actions = GrowingArray(actions.reshape((n, -1)))
        self.rewards = GrowingArray(np.nan_to_num(rewards))
        self.pixels = GrowingArray(pixels)
        self.info = info or {}

    @property
    def arrays(self):
        return [self.states, self.actions, self.rewards, self.pixels]

    @classmethod
    def from_list(cls, lst):
        return cls(*zip(*lst))

    def reward_sum(self):
        return self.rewards.sum()

    def action_repeat(self):
        if self.actions.dtype == np.float32:
            return float("NaN")
        return (self.actions[1:] == self.actions[:-1]).astype(np.float32).mean()

    def __len__(self):
        return len(self.states)

    @classmethod
    def add_step(cls, old, *arrays, info=None):
        arrays = [np.asarray(x) for x in arrays]
        arrays[1] = np.atleast_1d(arrays[1])  # Ensure 1D actions
        if old is None:
            return cls(
                *[x[None] for x in arrays],
                info={k: [v] for k, v in (info or {}).items()}
            )

        for old_arr, new_arr in zip(old.arrays, arrays):
            old_arr.add(np.asarray(new_arr)[None])

        newinfo = (
            {k: np.concatenate([old.info[k], [info[k]]]) for k in old.info.keys()}
            if info and old.info
            else None
        )
        old.info = newinfo
        return old

    def finished(self):
        self.states = np.asarray(self.states)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)
        self.pixels = np.asarray(self.pixels)

    @staticmethod
    def reward_sum_mean(trajectories):
        return np.mean([t.reward_sum() for t in trajectories]).item()

    @staticmethod
    def reward_sum_std(trajectories):
        return np.std([t.reward_sum() for t in trajectories]).item()

    @staticmethod
    def action_repeat_mean(trajectories):
        return np.mean([t.action_repeat() for t in trajectories]).item()

    @staticmethod
    def info_sum_mean(key, trajectories):
        return np.mean([np.nansum(t.info[key]) for t in trajectories]).item()

    @staticmethod
    def info_mean_mean(key, trajectories):
        return np.mean([np.nanmean(t.info[key]) for t in trajectories]).item()

    def stack(self, stack_size, pad=False):
        """Stack subsequent rows. Order: earlier->later.

        If pad, then prepend with copies of first, so length constant."""
        outs = []
        for arr in self.arrays:
            if pad:
                # Workaround https://github.com/numpy/numpy/issues/11395
                if arr.dtype == np.object_ and arr[0] is None:
                    arr = np.full(len(arr) + stack_size - 1, None)
                else:
                    p = [(stack_size - 1, 0)] + [(0, 0)] * (len(arr.shape) - 1)
                    arr = np.pad(arr, p, "edge")

            out = np.stack(
                [np.roll(arr, i, 0) for i in range(stack_size - 1, -1, -1)], 1
            )
            outs.append(out[(stack_size - 1) :])
        return outs


class Batch:
    absent_sentinel = -111  # Can not use NaN for int actions

    def __init__(self, states, actions, rewards, pixels, expert_actions, indices):

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.pixels = pixels
        self.expert_actions = expert_actions
        self.indices = indices

    def has_labels(self):
        labels = self.expert_actions[:, -1]
        return not (labels == self.absent_sentinel).any()

    def labels(self):
        #change the current action to previous 10 actions
        #labels = self.expert_actions[:, -1]
        labels = self.expert_actions
        assert self.has_labels(), "No expert action provided"
        return labels


def batch_cat(batches: List[Batch]):
    no_pixels = any([len(b.pixels.shape) == 2 for b in batches])
    tensors = []
    n = sum([b.states.shape[0] for b in batches])
    for k in ["states", "actions", "rewards", "pixels", "expert_actions", "indices"]:
        if k == "indices":
            x = torch.cat([b.indices + i * 100000000 for i, b in enumerate(batches)])
        elif k == "pixels" and no_pixels:
            x = torch.zeros(
                n, batches[0].states.shape[1], device=batches[0].states.device
            )
        else:
            x = torch.cat([getattr(b, k) for b in batches])
        tensors.append(x)
    batch = Batch(*tensors)
    return batch


class TransitionDataset:
    """Stacked transitions efficiently stored."""

    def __init__(
        self, states, actions, rewards, pixels, expert_actions, done, stack_size
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.pixels = pixels
        self.expert_actions = (
            expert_actions
            if expert_actions is not None
            else self.actions.new_full(self.actions.shape, Batch.absent_sentinel)
        )
        self.done = done
        self.stack_size = stack_size
        self.starts = self.compute_starts(done, stack_size)
        self.stack_arange = torch.arange(self.stack_size, dtype=torch.long)

    @property
    def tensors(self):
        return [
            self.states,
            self.actions,
            self.rewards,
            self.pixels,
            self.expert_actions,
            self.done,
        ]

    def __getitem__(self, indices: np.ndarray):
        """Input [idx], output Batch, tensors of shape [idx, stack]."""

        if isinstance(indices, range):
            indices = np.arange(
                indices.start or 0, indices.stop or len(self), indices.step or 1
            )

        indices = np.atleast_1d(indices)
        assert indices.size > 0, "Indices can not be empty"

        indices = self.starts[indices]
        indices = indices[:, None] + self.stack_arange[None, :]


        tensors = [t[indices] for t in self.tensors[:-1]]
        return Batch(*tensors, indices=indices)

    def to(self, device):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.pixels = self.pixels.to(device)
        self.expert_actions = self.expert_actions.to(device)
        return self

    @staticmethod
    def compute_starts(done: torch.Tensor, stack_size) -> torch.Tensor:
        """Compute indices where stack starts."""
        assert done[-1]
        done_indices = torch.nonzero(done).view(-1)
        # Indices we can not start
        if stack_size == 1:
            mask_indices = done_indices
        else:
            mask_indices = (
                done_indices[:, None]
                - torch.arange(stack_size - 1, dtype=torch.long)[None, :]
            )
            mask_indices = mask_indices.view(-1).clamp(0, len(done) - 1)

        # We invert by explicitly creating the mask
        mask = torch.zeros_like(done)
        mask[mask_indices] = 1
        start_indices = torch.nonzero(1 - mask).view(-1)
        return start_indices

    def __len__(self):
        return len(self.starts)

    @classmethod
    def from_trajectories(
        cls,
        trajectories: List[Trajectory],
        stack_size: int,
        expert_trajectories=False,
        expert_actions=None,
    ):

        if len(trajectories) == 0:
            return None
        arrays = zip(*[t.arrays for t in trajectories])
        lenghts = np.array([len(t) for t in trajectories])
        length = np.sum(lenghts).item()
        done = np.zeros(length, dtype=np.uint8)
        done[np.cumsum(lenghts) - 1] = 1
        tensors = [
            torch.from_numpy(np.concatenate(arr))
            if arr[0].dtype != np.object_
            else torch.zeros(length)
            for arr in arrays
        ]

        if expert_trajectories:
            expert_actions = tensors[1]
        return cls(
            *tensors,
            expert_actions=expert_actions,
            done=torch.from_numpy(done),
            stack_size=stack_size
        )

    @classmethod
    def cat(cls, a, b):
        if a is None:
            return b
        assert a.stack_size == b.stack_size, "Stack sizes must match"
        tensors = [
            torch.cat([t_a, t_b.to(t_a.device)])
            for t_a, t_b in zip(a.tensors, b.tensors)
        ]
        return cls(*tensors, stack_size=a.stack_size)

    def to_stack_size(self, stack_size):
        return TransitionDataset(*self.tensors, stack_size)

    def returns(self, discount):
        ret = 0
        returns = torch.zeros_like(self.rewards)
        for i in range(len(self.rewards) - 1, -1, -1):
            if self.done[i]:
                ret = 0
            ret = returns[i] = discount * ret + self.rewards[i]
        return returns
