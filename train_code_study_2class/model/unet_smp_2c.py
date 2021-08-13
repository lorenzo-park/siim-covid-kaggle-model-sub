import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from model.component import init_weights
from model.unet_smp import SMPModel
from utils.etc import get_in_features


class SMP2CModel(nn.Module):
    def __init__(self, config, hidden_dim=512):
        super(SMP2CModel, self).__init__()

        self.smp_model = SMPModel(config.unet_smp)
        if config.smp_path:
            self.smp_model.load_state_dict(torch.load(config.smp_path))

        self.smp_model.head = nn.Linear(hidden_dim, 1)

    def forward(self,x):
        x = self.smp_model(x)
        return x

    def forward_mask(self, x):
        seg_features = self.smp_model.forward_mask(x)
        return seg_features
