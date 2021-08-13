
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import LovaszLoss, DiceLoss, FocalLoss

import os
import torch
import torchmetrics

import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl

from model.unet_smp import SMPModel
from dataset import NIHDataset
from utils.etc import collate_fn, split_df, get_image_size_from_decoder_blocks, get_lr_scheduler
from utils.augmentation import get_study_transform


class LitUnetSmpNIH(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()
        self.config = config
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr
        self.num_classes = 4

        self.model = SMPModel(self.config.unet_smp)

        self.train_map = torchmetrics.BinnedAveragePrecision(num_classes=15)
        self.val_map = torchmetrics.BinnedAveragePrecision(num_classes=15)
        self.test_map = torchmetrics.BinnedAveragePrecision(num_classes=15)

        self.mask_img_size = get_image_size_from_decoder_blocks(config.unet_smp.decoder_blocks, config.img_size)

        self.lovasz_loss = LovaszLoss(mode="multilabel")
        self.dice_loss = DiceLoss(mode="multilabel")
        self.focal_loss = FocalLoss(mode="multilabel")

    def setup(self, stage):
        train_df = pd.read_csv(os.path.join(self.config.data_root, "train.csv")).rename(columns={"Patient ID": "PatientID"})
        train_df = train_df[~train_df["id"].duplicated(keep=False)].reset_index(drop=True)
        if self.config.second_train:
            _, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)
        else:
            train_df, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)

        if self.config.album:
            train_transform, val_transform = get_study_transform(self.config.img_size)
            self.train_set = NIHDataset(root=self.config.data_root, df=train_df, img_size=384, transform=train_transform)
            self.test_set = NIHDataset(root=self.config.data_root, df=val_df, img_size=384, transform=val_transform)
        else:
            transform = T.Compose([
                T.Resize(self.config.img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.train_set = NIHDataset(root=self.config.data_root, df=train_df, img_size=384, transform=transform)
            self.test_set = NIHDataset(root=self.config.data_root, df=val_df, img_size=384, transform=transform)

    def training_step(self, batch, _):
        inputs, targets = batch

        loss = None
        losses_to_use = self.config.losses.split(",")

        stage = "train"
        if "cls" in losses_to_use:
            outputs = self.model(inputs)
            loss_cls = F.binary_cross_entropy_with_logits(outputs, targets)
            self.log(f"{stage}_loss_cls", loss_cls, on_step=False, on_epoch=True, sync_dist=True)
            if loss is None:
                loss = loss_cls
            else:
                loss += loss_cls

            ap = self.train_map(torch.softmax(outputs, dim=1), targets)
            mean_ap = sum(ap) / len(ap)

            self.log(f"{stage}_mAP", mean_ap, on_step=False, on_epoch=True, sync_dist=True)

        loss = loss / len(losses_to_use)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outs):
        ap = self.train_map.compute()
        mean_ap = sum(ap) / len(ap)
        self.log("train_mAP_epoch", mean_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        loss = None
        losses_to_use = self.config.losses.split(",")

        stage = "val"
        if "cls" in losses_to_use:
            outputs = self.model(inputs)
            loss_cls = F.binary_cross_entropy_with_logits(outputs, targets)
            self.log(f"{stage}_loss_cls", loss_cls, on_step=False, on_epoch=True, sync_dist=True)
            if loss is None:
                loss = loss_cls
            else:
                loss += loss_cls

            ap = self.val_map(torch.softmax(outputs, dim=1), targets)
            mean_ap = sum(ap) / len(ap)

            self.log(f"{stage}_mAP", mean_ap, on_step=False, on_epoch=True, sync_dist=True)

        loss = loss / len(losses_to_use)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outs):
        ap = self.val_map.compute()
        mean_ap = sum(ap) / len(ap)
        self.log("val_mAP_epoch", mean_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch

        loss = None
        losses_to_use = self.config.losses.split(",")

        stage = "test"
        if "cls" in losses_to_use:
            outputs = self.model(inputs)
            loss_cls = F.binary_cross_entropy_with_logits(outputs, targets)
            self.log(f"{stage}_loss_cls", loss_cls, on_step=False, on_epoch=True, sync_dist=True)
            if loss is None:
                loss = loss_cls
            else:
                loss += loss_cls

            ap = self.test_map(torch.softmax(outputs, dim=1), targets)
            mean_ap = sum(ap) / len(ap)

            self.log(f"{stage}_mAP", mean_ap, on_step=False, on_epoch=True, sync_dist=True)

        loss = loss / len(losses_to_use)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_epoch_end(self, outs):
        ap = self.test_map.compute()
        mean_ap = sum(ap) / len(ap)
        self.log("test_mAP_epoch", mean_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        model_params = [
            {"params": self.model.seg.encoder.parameters(), "lr": self.learning_rate * self.config.w_enc},
            {"params": list(self.model.seg.decoder.parameters())+list(self.model.seg.segmentation_head.parameters()), "lr": self.learning_rate * self.config.w_seg},
            {"params": list(self.model.neck.parameters())+list(self.model.head.parameters()), "lr": self.learning_rate * self.config.w_cls},
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


    def calc_seg_loss(self, outputs_masks, masks):
        masks_sum = masks.sum(dim=-1).sum(dim=-1).sum(dim=-1)

        outputs_masks[masks_sum < 10] *= 0
        masks[masks_sum < 10] *= 0

        loss_seg = 0.0
        total = 0

        if self.config.loss_pooling:
            outputs_masks = F.max_pool2d(outputs_masks, self.config.loss_pooling)
            masks = F.max_pool2d(masks, self.config.loss_pooling)

        if self.config.lambda_bce:
            loss_seg += self.config.lambda_bce * F.binary_cross_entropy_with_logits(outputs_masks, masks)
            total += 1
        if self.config.lambda_lovasz:
            loss_seg += self.config.lambda_lovasz * self.lovasz_loss(outputs_masks, masks)
            total += 1
        if self.config.lambda_dice:
            loss_seg += self.config.lambda_dice * self.dice_loss(outputs_masks, masks)
            total += 1
        if self.config.focal_loss:
            loss_seg += self.config.lambda_focal * self.focal_loss(outputs_masks, masks)
            total += 1

        assert total != 0

        return loss_seg / total