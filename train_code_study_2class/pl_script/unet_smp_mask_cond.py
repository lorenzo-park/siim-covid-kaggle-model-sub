
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import LovaszLoss, DiceLoss, FocalLoss

import os
import torch
import torchmetrics

import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl

from model.unet_smp_mask_cond import SMPModelMaskCond
from dataset import COVIDDataset, VinBigDataset
from utils.etc import collate_fn, split_df, get_image_size_from_decoder_blocks, get_lr_scheduler
from utils.augmentation import get_study_transform

class LitUnetSmpMaskCond(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()
        self.config = config
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr
        self.num_classes = 4

        self.model = SMPModelMaskCond(self.config.unet_smp)

        self.train_map = torchmetrics.BinnedAveragePrecision(num_classes=4)
        self.val_map = torchmetrics.BinnedAveragePrecision(num_classes=4)
        self.test_map = torchmetrics.BinnedAveragePrecision(num_classes=4)

        self.mask_img_size = get_image_size_from_decoder_blocks(config.unet_smp.decoder_blocks, config.img_size)

        self.lovasz_loss = LovaszLoss(mode="multilabel")
        self.dice_loss = DiceLoss(mode="multilabel")
        self.focal_loss = FocalLoss(mode="multilabel")

    def setup(self, stage):
        if "vin" in self.config.data_root:
            train_df = pd.read_csv(os.path.join(self.config.data_root, "train.csv"))
            train_df, val_df = split_df(train_df, 8888, 0, cv="kf")
            train_transform, val_transform = get_study_transform(self.config.img_size)
            self.train_set = VinBigDataset(root=self.config.data_root, mask=self.config.mask_type, img_size=self.config.img_size, df=train_df, transform=train_transform, mask_img_size=self.mask_img_size)
            self.test_set = VinBigDataset(root=self.config.data_root, mask=self.config.mask_type, img_size=self.config.img_size, df=val_df, transform=val_transform, mask_img_size=self.mask_img_size)
        else:
            if self.config.no_ricord_val:
                assert self.config.dataset_postfix == "-merge" or self.config.dataset_postfix == "-pseudo" or  self.config.dataset_postfix == "-pseudo2"
                train_df = pd.read_csv(os.path.join(self.config.data_root, "train_psl_none.csv"))
                ricord_df = train_df[train_df["merge"] == True]
                train_df = train_df[train_df["merge"] == False]

                if self.config.second_train:
                    _, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)
                else:
                    train_df, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)

                train_df = pd.concat((train_df, ricord_df))
            else:
                train_df = pd.read_csv(os.path.join(self.config.data_root, "train_psl_none.csv"))

                if self.config.second_train:
                    _, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)
                else:
                    train_df, val_df = split_df(train_df, self.config.seed, self.config.fold, self.config.cv)


            if self.config.album:
                train_transform, val_transform = get_study_transform(self.config.img_size)
                self.train_set = COVIDDataset(root=self.config.data_root, mask=self.config.mask_type, img_size=self.config.img_size, df=train_df, transform=train_transform, mask_img_size=self.mask_img_size)
                self.test_set = COVIDDataset(root=self.config.data_root, mask=self.config.mask_type, img_size=self.config.img_size, df=val_df, transform=val_transform, mask_img_size=self.mask_img_size)
            else:
                transform = T.Compose([
                    T.Resize(self.config.img_size),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                self.train_set = COVIDDataset(root=self.config.data_root, mask=self.config.mask_type, img_size=self.config.img_size, df=train_df, transform=transform, mask_img_size=self.mask_img_size)
                self.test_set = COVIDDataset(root=self.config.data_root, mask=self.config.mask_type, img_size=self.config.img_size, df=val_df, transform=transform, mask_img_size=self.mask_img_size)

    def training_step(self, batch, _):
        inputs, targets, masks = self.get_batch(batch, mask=self.config.mask_type)

        loss = None
        losses_to_use = self.config.losses.split(",")

        inputs = torch.stack(inputs)

        stage = "train"

        if "seg" in losses_to_use:
            masks = torch.stack(masks)

            outputs_masks = self.model.forward_mask(inputs)
            loss_seg = self.calc_seg_loss(outputs_masks, masks)

            self.log(f"{stage}_loss_seg", loss_seg, on_step=False, on_epoch=True, sync_dist=True)
            if loss is None:
                loss = loss_seg
            else:
                loss += loss_seg

        if "cls" in losses_to_use:
            outputs = self.model(inputs, outputs_masks)
            loss_cls = F.cross_entropy(outputs, targets)
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
        inputs, targets, masks = self.get_batch(batch, mask=self.config.mask_type)

        loss = None
        losses_to_use = self.config.losses.split(",")

        inputs = torch.stack(inputs)

        stage = "val"

        if "seg" in losses_to_use:
            masks = torch.stack(masks)

            outputs_masks = self.model.forward_mask(inputs)
            loss_seg = self.calc_seg_loss(outputs_masks, masks)

            self.log(f"{stage}_loss_seg", loss_seg, on_step=False, on_epoch=True, sync_dist=True)
            if loss is None:
                loss = loss_seg
            else:
                loss += loss_seg

        if "cls" in losses_to_use:
            outputs = self.model(inputs, outputs_masks)
            loss_cls = F.cross_entropy(outputs, targets)
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
        inputs, targets, masks = self.get_batch(batch, mask=self.config.mask_type)

        loss = None
        losses_to_use = self.config.losses.split(",")

        inputs = torch.stack(inputs)

        stage = "test"

        if "seg" in losses_to_use:
            masks = torch.stack(masks)

            outputs_masks = self.model.forward_mask(inputs)
            loss_seg = self.calc_seg_loss(outputs_masks, masks)

            self.log(f"{stage}_loss_seg", loss_seg, on_step=False, on_epoch=True, sync_dist=True)
            if loss is None:
                loss = loss_seg
            else:
                loss += loss_seg

        if "cls" in losses_to_use:
            outputs = self.model(inputs, outputs_masks)
            loss_cls = F.cross_entropy(outputs, targets)
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
                        collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                        collate_fn=collate_fn, pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                        collate_fn=collate_fn, pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def get_batch(self, batch, mode="train", mask=None):
        if mask:
            imgs, _, _, boxes, targets_image, targets_study, masks = batch

            if mode == "train":
                img = [img.float() for img in imgs]
                masks = [mask.float() for mask in masks]
                targets_study = torch.stack(targets_study)
                return img, targets_study, masks
        else:
            imgs, _, _, boxes, targets_image, targets_study = batch

            if mode == "train":
                img = [img.float() for img in imgs]
                targets_study = torch.stack(targets_study)
                return img, targets_study, None

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
