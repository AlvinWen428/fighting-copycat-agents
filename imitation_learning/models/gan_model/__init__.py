import os
import torch

from ..imitation_model import ImitationPolicy
from .gan_mlp import Generator as MLPGenerator, Discriminator as MLPDiscriminator, F as MLPF
from ..utils import weights_init, learning_rate_decay


class FightCopycatPolicy(ImitationPolicy):
    def __init__(self, state_dim, stack_size, output_dims, policy_mode, load_path=None,
                 learning_rate=0.0002,discriminator_lr=0.0002, embedding_noise_std=0, gan_loss_weight=1.0, device=None):
        super(FightCopycatPolicy, self).__init__()
        self.state_dim = state_dim
        self.output_dims = output_dims
        self.load_path = load_path
        self.policy_mode = policy_mode
        self.device = device

        self.embedding_noise_std = embedding_noise_std
        self.gan_loss_weight = gan_loss_weight

        embedding_dim = 100
        self.cmi_input_dim = embedding_dim + self.output_dims

        if self.policy_mode == "fca":
            discrimitor_input_dim = self.cmi_input_dim
        else:
            discrimitor_input_dim = embedding_dim

        self.netG = MLPGenerator(state_dim * stack_size, embedding_dim)
        self.netD = MLPDiscriminator(discrimitor_input_dim, self.output_dims,)
        self.netF = MLPF(embedding_dim, self.output_dims)

        self.init_weights(weights_init)
        if self.load_path is not None:
            self.load_weights(self.load_path)

        self.netG, self.netD, self.netF = self.netG.to(device), self.netD.to(device), self.netF.to(device)

        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        self.optimizerF = torch.optim.Adam(self.netF.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        self.models = {'netG': self.netG, 'netD': self.netD, 'netF': self.netF}
        self.optimizers = {'G': self.optimizerG, 'D': self.optimizerD, 'F': self.optimizerF}

    def init_weights(self, init_fn):
        self.netG.apply(init_fn)
        self.netD.apply(init_fn)
        self.netF.apply(init_fn)

    def load_weights(self, load_path):
        self.netG.load_state_dict(torch.load(os.path.join(load_path, 'netG.pkl')))
        self.netD.load_state_dict(torch.load(os.path.join(load_path, 'netD.pkl')))
        self.netF.load_state_dict(torch.load(os.path.join(load_path, 'netF.pkl')))

    def train(self):
        self.netG.train()
        self.netD.train()
        self.netF.train()

    def eval(self):
        self.netG.eval()
        self.netD.eval()
        self.netF.eval()

    def zero_grad(self):
        self.netG.zero_grad()
        self.netD.zero_grad()
        self.netF.zero_grad()

    def optim_step(self, optim_name_list):
        for name in optim_name_list:
            if name in self.optimizers:
                self.optimizers[name].step()

    def learning_rate_decay(self, decay_rate):
        learning_rate_decay(self.optimizers, decay_rate)

    def save_model(self, model_save_dir):
        torch.save(self.netG.state_dict(), os.path.join(model_save_dir, "netG.pkl"))
        torch.save(self.netD.state_dict(), os.path.join(model_save_dir, "netD.pkl"))
        torch.save(self.netF.state_dict(), os.path.join(model_save_dir, "netF.pkl"))

    def forward_G(self, state):
        return self.netG(state)

    def forward_D(self, D_input):
        return self.netD(D_input)

    def forward_G_D(self, x, y):
        current_action = y[:, -1]
        prev_action = y[:, -2]
        current_state = x[:, -self.state_dim:]
        prev_states = x[:, :-self.state_dim]

        if self.policy_mode == 'bc-so':
            embedding = unnoised_embedding = self.netG(torch.cat((torch.zeros_like(prev_states).to(self.device), current_state), 1))
        elif self.policy_mode == 'bc-oh':
            embedding = unnoised_embedding = self.netG(x)
        else:
            embedding = unnoised_embedding = self.netG(x)

        if self.policy_mode == 'fca' and self.embedding_noise_std is not None:
            embedding = embedding + self.embedding_noise_std * torch.randn(embedding.shape).to(self.device)

        if self.policy_mode == 'fca':
            D_input = torch.cat((embedding, current_action), 1)
            unnoised_D_input = torch.cat((unnoised_embedding, current_action), 1)
        else:
            D_input = embedding
            unnoised_D_input = unnoised_embedding

        predicted_prev_action = self.netD(D_input)
        return predicted_prev_action, unnoised_embedding, unnoised_D_input

    def forward_curr_action(self, x, embedding):
        predicted_curr_action = self.netF(embedding)
        return predicted_curr_action

    def train_step(self, x, y, criterion):
        current_action = y[:, -1]
        prev_action = y[:, -2]

        # update Discriminator
        self.zero_grad()

        predicted_prev_action, embedding, D_input = self.forward_G_D(x, y)

        loss_D = criterion(predicted_prev_action, prev_action, reduce=False)
        loss_D = loss_D.mean()

        loss_D.backward(retain_graph=True)
        self.optim_step('D')

        # update generator, F and H
        self.zero_grad()

        predicted_prev_action = self.forward_D(D_input)

        predicted_curr_action = self.forward_curr_action(x, embedding)

        loss_Generator = -criterion(predicted_prev_action, prev_action, reduce=False)
        loss_Generator = loss_Generator.mean()
        if self.policy_mode != 'fca':
            loss_Generator = 0 * loss_Generator

        loss_curr_pred = criterion(predicted_curr_action, current_action, reduce=False)
        loss_curr_pred = loss_curr_pred.mean()

        loss_G = self.gan_loss_weight * loss_Generator + loss_curr_pred

        loss_G.backward(retain_graph=True)

        self.optim_step(['G', 'F', 'H'])

        info = {
            "generator_loss": loss_Generator.item(),
            "discriminator_loss": loss_D.item(),
            "train_loss": loss_curr_pred.item(),
        }
        return info

    def __call__(self, x):
        current_state = x[:, -self.state_dim:]
        prev_states = x[:, :-self.state_dim]

        if self.policy_mode == 'bc-so':
            embedding = self.netG(torch.cat((torch.zeros_like(prev_states).to(self.device), current_state), 1))
        else:
            embedding = self.netG(x)
        predicted_curr_action = self.netF(embedding)
        return predicted_curr_action
