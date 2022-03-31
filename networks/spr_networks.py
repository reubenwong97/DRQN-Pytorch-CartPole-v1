import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TransitionModel(nn.Module):
    def __init__(self, action_space, args):
        super().__init__()
        self.args = args
        layers = [
            nn.Linear(args.hidden_space + action_space, args.tran1_dim),
            nn.ReLU(),
            nn.Linear(args.tran1_dim, args.tran2_dim),
            nn.ReLU(),
            nn.Linear(args.tran2_dim, args.hidden_space),
        ]
        self.network = nn.Sequential(*layers)
        self.train()
        self.to(args.device)

    def forward(self, x):
        # stacked = torch.cat(x, action_onehot, dim=1)
        # stacking done externally
        next_state = self.network(x)

        return next_state


class ProjectionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hidden_space, 256)
        self.fc2 = nn.Linear(256, args.projection_out_dim)
        self.to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class PredictionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(args.projection_out_dim, args.prediction_dim)
        self.fc2 = nn.Linear(args.prediction_dim, args.projection_out_dim)
        self.to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
