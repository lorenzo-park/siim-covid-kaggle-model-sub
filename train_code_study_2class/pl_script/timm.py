from torch.utils.data import DataLoader

import os
import torch
import torchmetrics

import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl

from dataset import DistillCOVIDDataset, COVIDDataset
from model.timm import TimmModel
from utils.etc import split_df, get_lr_scheduler
from utils.augmentation import get_study_transform
from utils.loss import distillation_loss


class LitTimm(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()
        self.config = config
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr
        self.num_classes = 4

        self.model = TimmModel(self.config.model_config)

        self.train_map = torchmetrics.BinnedAveragePrecision(num_classes=4)
        self.val_map = torchmetrics.BinnedAveragePrecision(num_classes=4)
        self.test_map = torchmetrics.BinnedAveragePrecision(num_classes=4)

    def setup(self, stage):
        if self.config.distill:
            train_df = pd.read_csv(os.path.join(self.config.data_root, "train.csv"))
        else:
            train_df = pd.read_csv(os.path.join(self.config.data_root, "train_psl_none.csv"))

        if self.config.second_train:
            _, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)
        else:
            train_df, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)

        if self.config.album:
            train_transform, val_transform = get_study_transform(self.config.img_size)
            if self.config.distill:
                self.train_set = DistillCOVIDDataset(root=self.config.data_root, img_size=self.config.img_size, df=train_df, crop_with_lung=True, transform=train_transform)
                self.test_set = DistillCOVIDDataset(root=self.config.data_root, img_size=self.config.img_size, df=val_df, crop_with_lung=True, transform=val_transform)
            else:
                self.train_set = COVIDDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=train_df, transform=train_transform)
                self.test_set = COVIDDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=val_df, transform=val_transform)
        else:
            transform = T.Compose([
                T.Resize(self.config.img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            if self.config.distill:
                self.train_set = DistillCOVIDDataset(root=self.config.data_root, img_size=self.config.img_size, df=train_df, crop_with_lung=True, transform=transform)
                self.test_set = DistillCOVIDDataset(root=self.config.data_root, img_size=self.config.img_size, df=val_df, crop_with_lung=True, transform=transform)
            else:
                self.train_set = COVIDDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=train_df, transform=transform)
                self.test_set = COVIDDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=train_df, transform=transform)

    def training_step(self, batch, _):
        stage = "train"
        if self.config.distill:
            inputs, targets, psl_targets = batch
        else:
            inputs, targets = batch
            inputs = inputs.float()

        outputs = self.model(inputs)

        if self.config.distill:
            loss = distillation_loss(outputs, targets, psl_targets, \
                self.config.temperature, self.config.alpha)
        else:
            loss = F.cross_entropy(outputs, targets)

        ap = self.train_map(torch.softmax(outputs, dim=1), targets)
        mean_ap = sum(ap) / len(ap)

        self.log(f"{stage}_mAP", mean_ap, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_epoch_end(self, _):
        ap = self.train_map.compute()
        mean_ap = sum(ap) / len(ap)
        self.log("train_mAP_epoch", mean_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, _):
        stage = "val"

        if self.config.distill:
            inputs, targets, psl_targets = batch
        else:
            inputs, targets = batch
            inputs = inputs.float()

        outputs = self.model(inputs)
        if self.config.distill:
            loss = distillation_loss(outputs, targets, psl_targets, \
                self.config.temperature, self.config.alpha)
        else:
            loss = F.cross_entropy(outputs, targets)
        ap = self.val_map(torch.softmax(outputs, dim=1), targets)
        mean_ap = sum(ap) / len(ap)

        self.log(f"{stage}_mAP", mean_ap, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, _):
        ap = self.val_map.compute()
        mean_ap = sum(ap) / len(ap)
        self.log("val_mAP_epoch", mean_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, _):
        stage = "test"

        if self.config.distill:
            inputs, targets, psl_targets = batch
        else:
            inputs, targets = batch
            inputs = inputs.float()

        outputs = self.model(inputs)
        if self.config.distill:
            loss = distillation_loss(outputs, targets, psl_targets, \
                self.config.temperature, self.config.alpha)
        else:
            loss = F.cross_entropy(outputs, targets)

        ap = self.test_map(torch.softmax(outputs, dim=1), targets)
        mean_ap = sum(ap) / len(ap)

        self.log(f"{stage}_mAP", mean_ap, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_epoch_end(self, _):
        ap = self.test_map.compute()
        mean_ap = sum(ap) / len(ap)
        self.log("test_mAP_epoch", mean_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        model_params = [
            {"params": self.model.base.parameters(), "lr": self.learning_rate},
            {"params": list(self.model.neck.parameters())+list(self.model.head.parameters()), "lr": self.learning_rate},
        ]
        if self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model_params,
                lr=self.learning_rate,
                momentum=self.config.momentum,
                weight_decay=1e-5,
                # nesterov=True,
            )
        elif self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model_params,
                lr=self.learning_rate,
            )
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model_params,
                lr=self.learning_rate,
            )

        if self.config.lr_schedule.name:
            lr_scheduler = get_lr_scheduler(self.config.lr_schedule, optimizer)

            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.batch_size,
                          shuffle=True, pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                          pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                          pin_memory=True, num_workers=self.config.num_workers, drop_last=True)
