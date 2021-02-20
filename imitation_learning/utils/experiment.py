import inspect
import json
import logging
import os.path
import secrets
import sys
from collections import OrderedDict
from datetime import datetime
from functools import reduce
from os import path as osp, environ, makedirs
from pprint import pformat
from subprocess import check_output
from timeit import default_timer as timer
from typing import List

from hyperject import singleton

import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from tensorboardX import SummaryWriter
from tqdm import tqdm

from imitation_learning.utils.misc import dict_merge


def data_path(*elements):

    return osp.join("./data", *elements)


@singleton
def identifier_factory(container):
    i = str(int(datetime.utcnow().timestamp())) + secrets.token_hex(8)

    if container.config.name:
        identifier = f"{container.config.name}-{i}"
    else:
        identifier = i

    return identifier


@singleton
def out_dir_fn_factory(container):
    out_dir = container.config.out_dir

    def out_dir_fn(category=None, filename=None):
        path = os.path.join(out_dir, container.config.command, container.identifier)
        if category is not None:
            path = os.path.join(path, category)
        makedirs(path, exist_ok=True)
        if filename is not None:
            return os.path.join(path, filename)
        return path

    return out_dir_fn


@singleton
def logger_factory(container):
    logdir = container.out_dir_fn("runs")
    tb_writer = SummaryWriter(logdir)
    return TbLogger(tb_writer)


def setup_experiment(container, config):
    torch.manual_seed(container.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(container.seed)

    setup_logging(container.out_dir_fn(filename="out.log"))

    commit = environ.get("SOURCE_COMMIT", "")
    if not commit:
        commit = (
            check_output(["git", "describe", "--always", "--dirty"])
            .decode("utf-8")
            .strip()
        )

    data = {
        **config,
        "identifier": container.identifier,
        "source_commit": commit,
        "seed": container.seed,
    }
    logging.info(pformat(data))
    with open(container.out_dir_fn(filename="config.json"), "w") as f:
        json.dump(data, f, indent=2)


def setup_logging(path):
    formatter = logging.Formatter("%(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger("ignite.engine.engine.Engine").propagate = False
    logging.getLogger(
        "ignite.handlers.early_stopping.DelayedEarlyStopping"
    ).propagate = True


class Logger:
    def log_scalars(self, data, it=None):
        pass

    def log_image(self, key, img, it=None):
        pass

    def close(self):
        pass


class TbLogger(Logger):
    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def log_scalars(self, data, it=None):
        data = sort_dict(data)
        for k, v in data.items():
            self.writer.add_scalar(k, v, it)

    def log_image(self, key, img, it=None):
        assert isinstance(img, torch.Tensor), "Input must be Tensor"
        assert len(img.shape) == 3, "Image must be 3D"
        assert img.shape[0] == 3, "Image must be 3xWxH"
        assert img.max() <= 1, "Image must have max <= 1"
        assert img.min() >= 0, "Image must have min >= 0"
        self.writer.add_image(key, img, it)

    def close(self):
        self.writer.close()


class TqdmLogger(Logger):
    def __init__(self, t: tqdm):
        self.t = t

    def log_scalars(self, data, it=None):
        data = sort_dict(data)
        self.t.set_postfix(**data)


class MultiLogger(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

        methods = dict(inspect.getmembers(Logger(), predicate=inspect.ismethod))
        for f_name in methods.keys():

            def f(*args, **kwargs):
                for l in self.loggers:
                    getattr(l, f_name)(*args, **kwargs)

            setattr(self, f_name, f)


def wrap_tqdm_logger(logger: Logger, t: tqdm) -> Logger:
    return MultiLogger([logger, TqdmLogger(t)])


def plt_save_show(path: str):
    if path:
        plt.savefig(path)
    if "DOCKER" not in environ:
        plt.show()
    else:
        plt.close()


def tensor_read_image(path):
    img = Image.open(path).convert("RGB")
    t = torch.tensor(np.array(img)).permute([2, 0, 1]).float() / 255
    return t


class Timer:
    def __init__(self):
        self.duration = 0
        self.start_time = None

    def start(self):
        self.start_time = timer()

    def stop(self):
        if self.start_time is None:
            return
        self.duration += timer() - self.start_time
        self.start_time = None


class StateTimer:
    def __init__(self, names):
        self.timers = {n: Timer() for n in names}
        self.running = []

    def start(self, name):
        self.timers[name].start()
        self.running.append(name)

    def stop(self, name=None):
        if name is None:
            for n in self.running:
                self.timers[n].stop()

    def switch(self, name):
        self.stop()
        self.start(name)

    def report(self):
        return {n: t.duration for n, t in self.timers.items()}


def combine_logs(logs):
    keys = sorted(logs[-1].keys())
    return OrderedDict([(k, np.array([r[k] for r in logs])) for k in keys])


def mean_logs(logs):
    return OrderedDict((k, v.mean()) for k, v in combine_logs(logs).items())


def print_log_summary(it, report_freq, results):
    res = combine_logs(results[-report_freq:])
    entries = ["{}={:.4f}".format(k, values.mean()) for k, values in res]
    out = "\r{}: {}".format(it, " ".join(entries))
    print(out, end="")


def plot_logs(results, smoothen=False, figsize=(10, 10), path=None):
    plt.figure(figsize=figsize)
    results = combine_logs(results)
    _, axes = plt.subplots(len(results), 1, figsize=figsize)

    for ax, (key, values) in zip(axes, results.items()):
        if smoothen:
            values = savgol_filter(values, 101, 3)
        ax.set_title(key)
        ax.semilogy(np.arange(len(values)), values)
    plt_save_show(path)


def sort_dict(data):
    if isinstance(data, OrderedDict):
        return data
    return OrderedDict(sorted(data.items(), key=lambda x: x[0]))


def combine_configs(paths, updates):
    """
    Combine configs into one big JSON-style object.
    Paths earlier in list are overwritten by later ones.
    Configs in paths are overwritten by items in updates.

    :param paths: List of paths to YAML files
    :param updates: JSON-style object

    :return: JSON-style object
    """
    configs = []
    for path in paths:
        with open(path) as f:
            configs.append(yaml.load(f))
    return reduce(dict_merge, configs + [updates])
