import timm
import torch

import torch.nn as nn
import torch.nn.functional as F

from model.component import init_weights
from utils.etc import get_in_features


class TimmModel(nn.Module):
    def __init__(self, config):
        super(TimmModel, self).__init__()
        self.backbone_name = config.backbone_name
        self.hidden_dim = config.hidden_dim
        in_features = get_in_features(self.backbone_name, None)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.base = timm.create_model(
            self.backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool='',
        )
        if config.pretrained_path:
            self.base.load_state_dict(torch.load(config.pretrained_path))

        if config.neck_type == "D":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
            )
            head_feature_dim = in_features
        elif config.neck_type == "E":
            self.neck = nn.Identity()
            head_feature_dim = in_features
        elif config.neck_type == "F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, self.hidden_dim, bias=True),
                nn.BatchNorm1d(self.hidden_dim),
                torch.nn.PReLU(),
            )
            head_feature_dim = self.hidden_dim

        self.neck.apply(init_weights)
        self.head = nn.Linear(head_feature_dim, config.num_classes)

    def forward(self, x):
        x = self.base(x)

        # print(x.shape)
        if 'vit' not in self.backbone_name and 'swin' not in self.backbone_name:
            x = self.global_pool(x)
            x = x[:,:,0,0]
        # print(x.shape)
        x = self.neck(x)

        logits = self.head(x)

        return logits