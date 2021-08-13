from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import os
import torch
import torchmetrics

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from dataset import COVIDDataset
from metric.image_map import ImageLevelMAP
from model.faster_rcnn_detector import get_model
from utils.etc import collate_fn, split_df


class LitFasterRCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_classes = 2

        self.model = get_model(num_classes=self.num_classes, trainable_backbone_layers=5, pretrained=True)

    def setup(self, stage):
        transform = T.Compose([
            T.Resize(self.config.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_df = pd.read_csv(os.path.join(self.config.data_root, "merged_drop_multi_image_study.csv"))
        train_df, val_df = split_df(train_df, self.config.seed, self.config.cv_split)

        self.train_set = COVIDDataset(root=self.config.data_root, img_size=self.config.img_size, df=train_df, transform=transform)
        self.test_set = COVIDDataset(root=self.config.data_root, img_size=self.config.img_size, df=val_df, transform=transform)

        self.val_map = ImageLevelMAP()
        self.test_map = ImageLevelMAP()

    def training_step(self, batch, _):
        inputs, targets = self.get_batch(batch)

        outputs = self.model(inputs, targets)

        loss = sum(l for l in outputs.values())

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        inputs, targets = self.get_batch(batch)

        outputs = self.model(inputs)

        scores, labels, boxes, \
            targets_labels, targets_boxes = self.get_preds(outputs, targets)

        self.val_map(scores, labels, boxes, targets_labels, targets_boxes)

        return None

    def validation_epoch_end(self, outs):
        self.log("val_mAP", self.val_map.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = self.get_batch(batch)

        outputs = self.model(inputs)

        scores, labels, boxes, \
            targets_labels, targets_boxes = self.get_preds(outputs, targets)

        self.test_map(scores, labels, boxes, targets_labels, targets_boxes)

        return None

    def test_epoch_end(self, outs):
        self.log("test_mAP", self.test_map.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.batch_size,
                        collate_fn=collate_fn, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                        collate_fn=collate_fn, pin_memory=True, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                        collate_fn=collate_fn, pin_memory=True, num_workers=self.config.num_workers)

    def get_batch(self, batch, mode="train"):
        imgs, _, _, boxes, targets_image, targets_study = batch
        if mode == "train":
            img = [img.float() for img in imgs]
            target_obj = []
            for box, image_level_target in zip(boxes, targets_image):
                d = {}
                d["boxes"] = torch.stack([b.float() for b in box])
                d["labels"] = image_level_target
                target_obj.append(d)
            targets_study = torch.stack(targets_study)
            return img, target_obj

    def get_preds(self, outputs, targets):
        scores = []
        labels = []
        boxes = []

        targets_labels = []
        targets_boxes = []

        for output, target in zip(outputs, targets):
            # index_output = torch.Tensor([img_id_map[image_id]] * len(output["labels"])).long().to(self.device)

            # indexes_outputs.append(index_output)
            scores.append(output["scores"])
            labels.append(output["labels"])
            boxes.append(output["boxes"])

            # index_target = torch.Tensor([img_id_map[image_id]] * len(target["labels"])).long().to(self.device)
            # indexes_targets.append(index_target)
            targets_labels.append(target["labels"])
            targets_boxes.append(target["boxes"])

        return scores, labels, boxes, targets_labels, targets_boxes