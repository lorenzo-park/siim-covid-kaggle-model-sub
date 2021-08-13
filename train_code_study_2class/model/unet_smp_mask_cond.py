import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from model.component import init_weights
from utils.etc import get_in_features


class SMPModelMaskCond(nn.Module):
    def __init__(self, config):
        super(SMPModelMaskCond, self).__init__()
        in_features = get_in_features(config.backbone_name, config.model_type) + 16

        self.hidden_dim = config.hidden_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if config.decoder_channels:
            assert len(config.decoder_channels.split(",")) == config.decoder_blocks
            decoder_channels = list(map(int, config.decoder_channels.split(",")))
        else:
            if config.decoder_blocks == 3:
                decoder_channels = (512, 32, 16)
            elif config.decoder_blocks == 4:
                decoder_channels = (512, 64, 32, 16)
            elif config.decoder_blocks == 5:
                decoder_channels = (512, 128, 64, 32, 16)

        self.seg = smp.UnetPlusPlus(
            encoder_name=config.backbone_name,
            encoder_weights=config.encoder_weights,
            classes=config.classes,
            activation=None,
        )

        delattr(self.seg, "decoder")
        delattr(self.seg, "segmentation_head")

        self.seg.decoder = smp.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels=self.seg.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=config.decoder_blocks,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.seg.segmentation_head = smp.unetplusplus.model.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=config.classes,
            activation=None,
            kernel_size=3,
        )

        self.mask_type = config.mask_type

        if config.neck_type == "D":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
            )
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
        self.head = nn.Linear(head_feature_dim, 4)

        self.mask_features = nn.Sequential(
            nn.Conv2d(2,4,kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4,8,kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,16,kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x, mask_output):
        mask_output = self.mask_features(mask_output)
        mask_output = self.global_pool(mask_output)
        mask_output = mask_output[:,:,0,0]
        x = self.seg.encoder(x)[-1]
        x = self.global_pool(x)
        x = x[:,:,0,0]
        x = self.neck(torch.cat([x, mask_output], dim=1))
        logits = self.head(x)
        return logits

    def forward_mask(self, x):
        global_features = self.seg.encoder(x)
        seg_features = self.seg.decoder(*global_features)
        seg_features = self.seg.segmentation_head(seg_features)
        return seg_features
