from typing import Union
import torch
import numpy as np
from gym.spaces import Box

from hyperject import singleton

from imitation_learning.utils.misc import environment_path_repr
from imitation_learning.utils.experiment import data_path
from imitation_learning.utils.data import TensorLoader


class StackMode:
    """Base mode."""

    def __init__(self, env, stack_size, skip=1, max=1, state_mask_slice=None):

        self.skip = skip
        self.max = max
        self.stack_size = stack_size
        self.stack_need = (stack_size - 1) * skip + max
        self.state_mask_slice = state_mask_slice

        self.action_space = env.action_space
        self.state_shape = env.observation_space.shape
        self.action_dims = (
            np.prod(env.action_space.shape).item()
            if isinstance(env.action_space, Box)
            else 1
        )

    def batch(self, batch):
        assert batch.states.shape[1] >= self.stack_need
        if self.max == 1:
            x = batch.states[:, -self.stack_need:: self.skip].float()
        else:
            xs = [
                batch.states[:, -self.stack_need + m:: self.skip]
                for m in range(self.max)
            ]
            x = torch.max(torch.stack(xs, 0).float(), dim=0)[0]

        x = self.modify_states(x)
        x = self.adjust_shape(x, 'batch')
        return x

    def step(self, state, pixels, trajectory):
        if self.stack_size == 1:
            state = self.modify_states(state)
            return self.adjust_shape(state.astype(np.float32), 'step')
        if trajectory:
            states = np.concatenate(
                [trajectory.states[-self.stack_need:], state[None]]
            )
        else:
            states = state[None]

        pad = self.stack_need - len(states)
        if pad > 0:
            s = states.shape
            states = np.pad(states.reshape(len(states), -1), [(pad, 0), (0, 0)], "edge")
            states = states.reshape(self.stack_need, *s[1:])

        if self.max == 1:
            selected_states = states[-self.stack_need:: self.skip]
        else:
            max_stack = [
                states[-self.stack_need + m:: self.skip] for m in range(self.max)
            ]
            selected_states = np.max(np.stack(max_stack, axis=0), axis=0)
        selected_states = self.modify_states(selected_states)

        return self.adjust_shape(selected_states.astype(np.float32), 'step')

    def modify_states(self, states):
        if self.state_mask_slice is not None:
            states[..., self.state_mask_slice] = 0
        return states

    def shape(self):
        return (self.stack_size * self.state_shape[0],) + self.state_shape[1:]

    def adjust_shape(self, states, part):
        if part == 'batch':
            return states.view(-1, *self.shape())
        elif part == 'step':
            return states.reshape(self.shape())


class NoneMode(StackMode):
    def __init__(self, env):
        super().__init__(env, 1)


ModeType = Union[StackMode, "ModeWrapper"]


class ModeWrapper:
    def __init__(self, child_mode: ModeType):
        self.child_mode = child_mode
        self.stack_need = child_mode.stack_need
        self.action_space = child_mode.action_space
        self.action_dims = child_mode.action_dims

    def batch(self, batch):
        raise NotImplementedError()

    def step(self, state, pixels, trajectory):
        raise NotImplementedError()

    def shape(self):
        """Shape of transformed input."""
        return self.child_mode.shape()

    def adjust_shape(self, states):
        return self.child_mode.adjust_shape(states)

    def learn_normalization(self, batch):
        assert hasattr(
            self.child_mode, "learn_normalization"
        ), "No TransformMode in decorator chain"
        return self.child_mode.learn_normalization(batch)

    def load_normalization(self, normalizer):
        assert hasattr(
            self.child_mode, "load_normalization"
        ), "No TransformMode in decorator chain"
        return self.child_mode.load_normalization(normalizer)

    def create_cache(self, *datasets):
        if isinstance(self.child_mode, ModeWrapper):
            self.child_mode.create_cache(*datasets)


class TransformMode(ModeWrapper):
    def __init__(self, child_mode: ModeType):
        super().__init__(child_mode)
        self.mean, self.std = None, None

    def learn_normalization(self, batch):
        x = self.child_mode.batch(batch)
        self.mean = x.mean().item()
        self.std = x.std().item()
        return {"mean": self.mean, "std": self.std}

    def load_normalization(self, normalizer):
        self.mean = normalizer["mean"]
        self.std = normalizer["std"]

    def batch(self, batch):
        if self.mean is None:
            raise RuntimeError("Normalization not set")

        x = self.child_mode.batch(batch)
        return (x - self.mean) / self.std

    def step(self, state, pixels, trajectory):
        if self.mean is None:
            raise RuntimeError("Normalization not set")

        x = self.child_mode.step(state, pixels, trajectory)
        return (x - self.mean) / self.std


class ObstructMode(ModeWrapper):
    def __init__(self, child_mode: ModeType, coords):
        super().__init__(child_mode)
        self.height_slice = slice(coords[0][0], coords[1][0])
        self.width_slice = slice(coords[0][1], coords[1][1])

    def batch(self, batch):
        x = self.child_mode.batch(batch)
        x[..., self.height_slice, self.width_slice] = 0
        return x

    def step(self, state, pixels, trajectory):
        x = self.child_mode.step(state, pixels, trajectory)
        x[..., self.height_slice, self.width_slice] = 0
        return x


def construct_mode(
        mode: str,
        environment,
        stack_size,
        normalize=False,
):
    env = environment.env
    state_mask_slice = environment.state_mask_slice

    transform = TransformMode

    def states():
        return transform(StackMode(env, stack_size, state_mask_slice=state_mask_slice))

    def none():
        return transform(NoneMode(env))

    mode_obj = {
        "none": none,
        "stack": states,
    }.get(mode, lambda: None)()

    if mode_obj is None:
        raise RuntimeError(f"Mode {mode} not found")

    if normalize:
        load_normalization(environment, mode, mode_obj)
    return mode_obj


def load_normalization(environment, mode, mode_obj):
    import json

    key = f"{environment_path_repr(environment)}-{mode}"

    with open(data_path("normalization.json"), "r") as f:
        data = json.load(f)
        print(key)

        assert key in data, "No normalization found"
        mode_obj.load_normalization(data[key])


def normalize(environment, dataset, mode, mode_obj):
    import json

    key = f"{environment_path_repr(environment)}-{mode}"

    with open(data_path("normalization.json"), "r") as f:
        data = json.load(f)

    if key in data:
        mode_obj.load_normalization(data[key])
        return

    print("Constructing normalization")
    loader = TensorLoader(dataset, batch_size=10 * 64, shuffle=True)
    batch = next(iter(loader))
    normalizer = mode_obj.learn_normalization(batch)

    data[key] = normalizer
    with open(data_path("normalization.json"), "w") as f:
        json.dump(data, f, indent=2)


@singleton
def mode_factory(container):
    return construct_mode(
        mode=container.config.mode,
        environment=container.environment,
        stack_size=container.config.stack_size,
        normalize=True,
    )


@singleton
def mode_unnormalized_factory(container):
    return construct_mode(
        mode=container.config.mode,
        environment=container.environment,
        stack_size=container.config.stack_size,
    )
