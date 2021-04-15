import torch.nn as nn
from torch.nn import Conv1d

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.encoder = nn.Sequential(
            Conv1d(1, 16, 8),
            Conv1d(16, 32, 16),
            Conv1d(32, 64, 32),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 32),
            nn.ConvTranspose1d(32, 16, 16),
            nn.ConvTranspose1d(16, 1, 8),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1,599)
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.cnn = nn.Sequential(
            Conv1d(1, 32, 16),
            Conv1d(32, 16, 8),
        )

        self.fc1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(577, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = x.reshape(-1,577)
        x = self.fc2(x)
        return x
