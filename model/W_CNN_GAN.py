import torch.nn as nn
from torch.nn import Conv1d
import torch
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(32),
            Conv1d(32, 64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            Conv1d(64, 128, 64),
            nn.BatchNorm1d(128)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([128, 505]),
            nn.ConvTranspose1d(128, 64, 64),
            nn.ReLU(),
            nn.LayerNorm([64, 568]),
            nn.ConvTranspose1d(64, 32, 32),
            nn.LayerNorm([32, 599]),
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x, min, max):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.reshape(-1, 599)
        x = x * (max - min) + min
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(1),
            Conv1d(1, 16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            Conv1d(16, 32, 16),
            nn.BatchNorm1d(32),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([32, 577]),
            nn.ConvTranspose1d(32, 16, 16),
            nn.ReLU(),
            nn.LayerNorm([16, 592]),
            nn.ConvTranspose1d(16, 1, 8),
            nn.LayerNorm([1, 599]),
        )

        self.fc = nn.Linear(599, 1)
    def forward(self, x):
        x = x.reshape(-1, 599, 1)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(-1, 599)
        x = self.fc(x)
        return x

class utils:
    def __init__(self):
        self.Lambda = 10

    def calc_gradient_penalty(self, netD, real_data, fake_data, batch_size):
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.Lambda
        return gradient_penalty
