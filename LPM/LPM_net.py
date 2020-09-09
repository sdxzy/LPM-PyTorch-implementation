import torch
import os
import torch.nn as nn

class LPM(nn.Module):
    def __init__(self, in_channel):
        super(LPM, self).__init__()
        self.in_channel = in_channel
        self.net = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.Sigmoid(),
            nn.Linear(in_channel, in_channel*2),
            nn.Sigmoid(),
            nn.Linear(in_channel*2, in_channel//2),
            nn.Sigmoid(),
            nn.Linear(in_channel//2, in_channel//4),
            nn.Sigmoid(),
            nn.Linear(in_channel//4, 1)
        )

    def forward(self, x):
        return self.net(x)






# if __name__ == '__main__':
#     net = LPM(224)
#     print(net)
