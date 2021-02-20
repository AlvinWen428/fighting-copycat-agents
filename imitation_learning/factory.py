import sys
import random

from hyperject import Container, singleton
import torch

from imitation_learning.modes import mode_factory, mode_unnormalized_factory
from imitation_learning.environments import (
    build_environment_factory,
    input_dims_factory,
    output_dims_factory,
)
from imitation_learning.utils.misc import dict_merge
from imitation_learning.utils.experiment import (
    identifier_factory,
    out_dir_fn_factory,
    logger_factory,
)
from imitation_learning.utils.data import (
    dataset_factory,
    dataloaders_factory,
    dataset_stack_size_factory,
)


@singleton
def seed_factory(container):
    if container.config.seed:
        return container.config.seed
    return random.randrange(sys.maxsize)


@singleton
def device_factory(container):
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


factory_graph = {
    "mode": mode_factory,
    "mode_unnormalized": mode_unnormalized_factory,
    "environment": build_environment_factory(),
    "input_dims": input_dims_factory,
    "output_dims": output_dims_factory,
    "seed": seed_factory,
    "identifier": identifier_factory,
    "out_dir_fn": out_dir_fn_factory,
    "logger": logger_factory,
    "dataset": dataset_factory,
    "dataset_stack_size": dataset_stack_size_factory,
    "device": device_factory,
    "dataloaders": dataloaders_factory,
    "train_dataloader": singleton(lambda c: c.dataloaders["train"]),
    "test_dataloader": singleton(lambda c: c.dataloaders["test"]),
}

default_config = {
    "identifier": None,
    "source_commit": None,
    "mode": "stack",
    "env": None,
    "command": None,
    "stack_size": 2,
    "policy": {
        "policy_mode": "fca",  # fca / bc-so / bc-oh
        "embedding_noise_std": None,
        "gan_loss_weight": 1.0,
    },
    "batch_size": 64,
    "optim": {"learning_rate": 2e-4, "discriminator_lr": 2e-4, "lr_decay_rate": 0.1, "lr_decay_times": 3},
    "eval_expert": False,
    "eval_runs": 100,
    "load_path": None,
    "num_samples": None,
    "name": None,
    "fixed_data": True,
    "seed": None,
    "out_dir": "./results",
    "report_freq": None,
    "num_its": 300000,
    "eval_at_end": False,
    "early_stop": 50,
}


def build_container(config):
    combined_config = dict_merge(default_config, config, strict=True)
    return Container.make(factory_graph, combined_config)
