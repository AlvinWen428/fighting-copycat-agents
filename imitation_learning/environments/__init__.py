import numpy as np
from hyperject import singleton

from .mujoco import MujocoEnvironment, ant_reset


def construct_environment(env: str):
    if env in MujocoEnvironment.names:
        environment = MujocoEnvironment(env)
    else:
        raise RuntimeError(f"Env {env} not found")

    return environment


def build_environment_factory():
    @singleton
    def factory(container):
        return construct_environment(env=container.config.env)

    return factory


@singleton
def input_dims_factory(container):
    return np.prod(container.mode.shape()).item()


@singleton
def output_dims_factory(container):
    env = container.environment.env
    return sum(env.action_space.shape)
