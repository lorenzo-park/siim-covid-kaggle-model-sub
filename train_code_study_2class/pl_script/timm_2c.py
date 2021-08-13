from torch.utils.data import DataLoader

import os
import torch
import torchmetrics

import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl

from dataset import COVID2CDataset
from model.timm_2c import TimmModel2C
from utils.etc import split_df, get_lr_scheduler
from utils.augmentation import get_study_transform
from utils.loss import distillation_loss


class LitTimm2C(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()
        self.config = config
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr
        self.num_classes = 4

        self.model = TimmModel2C(self.config)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def setup(self, stage):
        train_df = pd.read_csv(os.path.join(self.config.data_root, "train_psl_none.csv"))
        if self.config.second_train:
            _, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)
        else:
            train_df, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)

        if self.config.album:
            train_transform, val_transform = get_study_transform(self.config.img_size)
            self.train_set = COVID2CDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=train_df, transform=train_transform)
            self.test_set = COVID2CDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=val_df, transform=val_transform)
        else:
            transform = T.Compose([
                T.Resize(self.config.img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.train_set = COVID2CDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=train_df, transform=transform)
            self.test_set = COVID2CDataset(root=self.config.data_root, mask=None, img_size=self.config.img_size, df=val_df, transform=transform)

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
            loss = F.binary_cross_entropy_with_logits(outputs, targets.unsqueeze(1).float())

        acc = self.train_acc(outputs > 0.5, targets.unsqueeze(1).long())
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_epoch_end(self, _):
        acc = self.train_acc.compute()
        self.log("train_acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
            loss = F.binary_cross_entropy_with_logits(outputs, targets.unsqueeze(1).float())

        acc = self.val_acc(outputs > 0.5, targets.unsqueeze(1).long())
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, _):
        acc = self.val_acc.compute()
        self.log("val_acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
            loss = F.binary_cross_entropy_with_logits(outputs, targets.unsqueeze(1).float())

        acc = self.test_acc(outputs > 0.5, targets.unsqueeze(1).long())
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_epoch_end(self, _):
        acc = self.test_acc.compute()
        self.log("test_acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        model_params = [
            {"params": self.model.model.base.parameters(), "lr": self.learning_rate * 0.1},
            {"params": list(self.model.model.neck.parameters())+list(self.model.model.head.parameters()), "lr": self.learning_rate},
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
