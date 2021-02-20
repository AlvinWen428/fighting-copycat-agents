import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def learning_rate_decay(optimizers, decay_rate):
    if isinstance(optimizers, list):
        for opt in optimizers:
            for p in opt.param_groups:
                p['lr'] *= decay_rate
    elif isinstance(optimizers, dict):
        for name, opt in optimizers.items():
            for p in opt.param_groups:
                p['lr'] *= decay_rate
