from torch.nn import GRU
import torch.nn as nn

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.rnn = GRU(input_size=10, hidden_size=32, bias=True, batch_first=True)
        self.embedding = nn.Linear(1, 10)
        self.fc = nn.Linear(32,1)
        self.dis = nn.Sequential(
            nn.Linear(599, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.reshape(-1,599,1)
        x = self.embedding(x)
        x,h = self.rnn(x)
        x = self.fc(x)
        x = x.reshape(-1,599)
        x = self.dis(x)
        return x


class generator (nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.rnn = GRU(input_size = 256, hidden_size = 256, bidirectional = True, bias = True, batch_first = True)
        self.gen = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 6),
            nn.Sigmoid()
            )

    def forward(self, x):
        x, h = self.rnn(x)
        x = x.reshape([-1, 51200])
        x = x.reshape([-1, 100, 512])
        x = self.gen(x)
        x = x.reshape([-1, 600])
        x = x[:, :599]
        return x
