import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

# Q_network
class Q_net(nn.Module):
    """
    Returns q_vals, h_ts, final_h on __call__
    """

    def __init__(self, args, state_space=None, action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert (
            state_space is not None
        ), "None state_space input: state_space should be selected."
        assert (
            action_space is not None
        ), "None action_space input: action_space should be selected."

        self.args = args
        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        # self.lstm    = nn.LSTM(self.hidden_space,self.hidden_space, batch_first=True)
        self.GRU = nn.GRU(self.hidden_space, self.hidden_space, batch_first=True)
        self.Linear2 = nn.Linear(
            self.hidden_space, self.action_space
        )  # serves as QHead
        self.to(args.device)

    def forward(self, x, h):
        x = F.relu(self.Linear1(x))
        # x, (new_h, new_c) = self.lstm(x,(h,c)) # c was passed as param into function
        h_ts, final_h = self.GRU(x, h)
        q_vals = self.Linear2(h_ts)
        # return x, new_h,
        return q_vals, h_ts, final_h

    def sample_action(self, obs, h, epsilon):
        output = self.forward(obs, h)

        # if random.random() < epsilon:
        #     return random.randint(0,1), output[1], output[2]
        # else:
        #     return output[0].argmax().item(), output[1] , output[2]
        if random.random() < epsilon:
            return random.randint(0, 1), output[1]
        else:
            return output[0].argmax().item(), output[1]

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        # if training is True:
        #     return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        # else:
        #     return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space])
