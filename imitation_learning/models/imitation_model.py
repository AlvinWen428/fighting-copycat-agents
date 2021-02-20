import os
import torch

from .utils import learning_rate_decay


class ImitationPolicy:
    def __init__(self):
        self.models = {}
        self.optimizers = {}

    def init_weights(self, init_fn):
        for name, model in self.models.items():
            model.apply(init_fn)

    def load_weight(self, load_path):
        for name, model in self.models.items():
            model.load_state_dict(torch.load(os.path.join(load_path, "{}.pkl".format(name))))

    def train(self):
        for name, model in self.models.items():
            model.train()

    def eval(self):
        for name, model in self.models.items():
            model.eval()

    def zero_grad(self):
        for name, model in self.models.items():
            model.zero_grad()

    def optim_step(self, optim_name_list):
        for name in optim_name_list:
            if name in self.optimizers:
                self.optimizers[name].step()

    def learning_rate_decay(self, decay_rate):
        learning_rate_decay(self.optimizers, decay_rate)

    def save_model(self, model_save_dir):
        for name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(model_save_dir, "{}.pkl".format(name)))

    def train_step(self, x, y, criterion):
        """
        x: observations, y: actions, criterion: loss function
        """
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError
