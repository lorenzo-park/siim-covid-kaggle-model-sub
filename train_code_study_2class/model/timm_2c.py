import timm
import torch

import torch.nn as nn
import torch.nn.functional as F

from model.component import init_weights
from model.timm import TimmModel
from utils.etc import get_in_features


class TimmModel2C(nn.Module):
    def __init__(self, config, hidden_dim=192):
        super(TimmModel2C, self).__init__()
        self.model = TimmModel(config.model_config)
        if config.model_path:
            self.model.load_state_dict(torch.load(config.model_path))

        self.model.head = nn.Linear(hidden_dim, 1)

    def forward(self,x):
        x = self.model(x)
        return x
