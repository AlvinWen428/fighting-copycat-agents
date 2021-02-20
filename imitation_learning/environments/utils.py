import gym
import numpy as np
from imitation_learning.trajectory import TransitionDataset


class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, self.scale * rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def expert_data(stack_size, path) -> TransitionDataset:
    data = np.load(path,allow_pickle=True)
    return TransitionDataset.from_trajectories(
        data, stack_size, expert_trajectories=True
    )
