import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Model(nn.Module):
    def __init__(self, input_dim=None, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        """
        self.input_layer = nn.Linear(self.input_dim, 128)

        self.hidden_layer1 = nn.Linear(128, 128)
        self.hidden_layer2 = nn.Linear(128, 128)
        self.hidden_layer3 = nn.Linear(128, 128)

        self.policy_layer = nn.Linear(128, self.output_dim)
        self.value_layer = nn.Linear(128, 1)
        """
        self._policy = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

        self._value = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        print(self)


    def forward(self,  input):
        """
        h = F.relu(self.input_layer(input))
        h = F.relu(self.hidden_layer1(h))
        h = F.relu(self.hidden_layer2(h))
        h = F.relu(self.hidden_layer3(h))

        pi = Categorical(logits=self.policy_layer(h))
        value = self.value_layer(h)
        """
        pi = Categorical(logits=self._policy(input))
        value = self._value(input)
        return pi, value

    def save(self, modelpath):
        torch.save(self.to('cpu').state_dict(), modelpath)
        self.to(device)

    def load(self, modelpath):
        self.load_state_dict(torch.load(modelpath, map_location=device))
