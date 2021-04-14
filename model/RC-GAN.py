from torch.nn import GRU
import torch.nn as nn

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(599, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x


class generator (nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.rnn = GRU(input_size = 1, hidden_size = 5, bidirectional = True, bias = True, batch_first = True)
        self.gen = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(10, 6),
            )

    def forward(self, x):
        x, h = self.rnn(x)
        x = x.reshape([-1, 1000])
        x = x.reshape([-1, 100, 10])
        x = self.gen(x)
        x = x.reshape([-1, 600])
        x = x[:, :599]
        return x
