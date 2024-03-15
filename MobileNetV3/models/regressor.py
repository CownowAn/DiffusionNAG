import torch
import torch.nn as nn
import torch.optim as optim

from . import utils

@utils.register_model(name='MLPRegressor')
class MLPRegressor(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size):
    def __init__(self, config):
        super().__init__()
        input_size = int(config.data.max_node * config.data.n_vocab)
        hidden_size = config.model.hidden_size
        output_size = config.model.output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, X, time_cond, maskX):
        x = X.view(X.size(0), -1)
        x = self.activation(self.fc1(x))
        x= self.activation(self.fc2(x))
        x= self.activation(self.fc3(x))
        x= self.fc4(x)
        return x