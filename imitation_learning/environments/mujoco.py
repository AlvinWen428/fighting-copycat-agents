from contextlib import contextmanager
from functools import partial

import gym
import numpy as np
import tensorflow as tf
import torch

from imitation_learning.utils.experiment import data_path
from imitation_learning.environments.utils import RewardScaleWrapper, expert_data
from imitation_learning.modes import NoneMode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_non_pos_slice = {"Ant": slice(13, None), "Hopper": slice(5, None), "Reacher": slice(6, 8),
                  "HalfCheetah": slice(8, None), "Humanoid": slice(22, None), "Walker2d": slice(8, None)
                  }


class ClampWrapper(gym.ActionWrapper):
    def action(self, action):
        # return action
        return np.clip(action, -1, 1)


def ant_reset(env, old):
    old()
    env = env.unwrapped
    qpos = np.array(env.init_qpos)
    qpos[7:] = env.np_random.uniform(size=env.model.nq - 7, low=-1, high=1)
    qvel = env.init_qvel + env.np_random.randn(env.model.nv) * 1
    env.set_state(qpos, qvel)
    return env._get_obs()


class MujocoEnvironment:
    names = ["Ant", "Hopper", "Reacher", "HalfCheetah", "Humanoid", "Walker2d"]
    discrete = False
    expert_steps = 1
    obstruction = None

    def __init__(
        self,
        envname,
        frame_skip=1,
        reset_fn=None,
        episode_steps=500,
        mod=None,
    ):
        self.envname = envname

        self.env = gym.make(envname + "-v2")

        if reset_fn:
            self.env.reset = partial(reset_fn, self.env, self.env.reset)

        if frame_skip is not None:
            org_frame_skip = self.env.unwrapped.frame_skip
            self.env.unwrapped.frame_skip = frame_skip
            ratio = org_frame_skip / frame_skip
            self.env._max_episode_steps = int(episode_steps * ratio)
            self.env = RewardScaleWrapper(self.env, 1 / ratio)

            self.name = envname + "-fs" + str(frame_skip)
        else:
            self.env._max_episode_steps = episode_steps
            self.name = envname
            self.frame_skip = self.env.unwrapped.frame_skip

        if mod is not None:
            self.name = self.name + "-" + mod

        if envname == "Hopper" or envname == "Walker2d":
            self.env = ClampWrapper(self.env)

        self.data_path = data_path("trajectories/" + self.name + ".npy")
        self.data = partial(expert_data, path=self.data_path)

        self.expert_mode = NoneMode(self.env)
        self.frame_skip = frame_skip
        self.expert = None

        self.causal_mask = [1] * self.env.observation_space.shape[0] + [
            0
        ] * self.env.action_space.shape[0]

    @staticmethod
    def criterion(a, b, reduce=True):
        d = (a - b).pow(2).sum(-1)
        if reduce:
            d = d.mean()
        return d

    @property
    def state_mask_slice(self):
        return _non_pos_slice[self.envname]

    def expert_fn(self, state):
        return self.expert(state)

    def agent_fn(self, policy, state):
        return np.array(policy(torch.tensor(state, dtype=torch.float, device=device)).cpu())

    @contextmanager
    def setup(self):
        """Setup environment and expert. Use as context."""
        import tf_util
        import load_tf_policy

        self.expert = load_tf_policy.load_policy(
            data_path("experts/" + self.envname + "-v1.pkl")
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config):
            tf_util.initialize()
            yield self

    def __repr__(self):
        return "<MujocoEnvironment " + self.name + ">"
