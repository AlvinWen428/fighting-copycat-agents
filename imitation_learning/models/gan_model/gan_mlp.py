import torch
import torch.nn as nn

from .utils import GanModule


class Generator(GanModule, nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
        super(GanModule, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(self._input_dims, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, self._output_dims)
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(GanModule, nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
        super(GanModule, self).__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self._input_dims, 300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, self._output_dims)
        )

    def forward(self, input):
        output = self.main(input)
        return output


class F(GanModule, nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
        super(GanModule, self).__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self._input_dims, 300),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(300, self._output_dims)
        )

    def forward(self, input):
        output = self.main(input)
        return output
