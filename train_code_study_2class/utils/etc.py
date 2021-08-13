from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import torch.optim as optim
import numpy as np

from gskfold import StratifiedGroupKFold


def collate_fn(batch):
    return tuple(zip(*batch))


def split_df(train_df, seed, split, cv="sgkf"):
        if cv == "sgkf":
            train_df["stratified_key"] = np.argmax(train_df[['Negative for Pneumonia','Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']])
            sgkf = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
            # skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            train_df['fold'] = -1
            for fold, (_, val_idx) in enumerate(sgkf.split(train_df, y=train_df.stratified_key, groups=train_df.PatientID)):
            # for fold, (_, val_idx) in enumerate(skf.split(train_df, y=train_df.stratified_key.tolist())):
                train_df.loc[val_idx, 'fold'] = fold

        elif cv == "skf":
            train_df["stratified_key"] = np.argmax(train_df[['Negative for Pneumonia','Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']])
            skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
            train_df['fold'] = -1
            for fold, (_, val_idx) in enumerate(skf.split(train_df, y=train_df.stratified_key.tolist())):
                train_df.loc[val_idx, 'fold'] = fold

        elif cv == "gkf":
            skf = GroupKFold(n_splits=5)
            for fold, (_, val_idx) in enumerate(skf.split(train_df, groups=train_df.PatientID)):
                train_df.loc[val_idx, 'fold'] = fold

        elif cv == "kf":
            kf = KFold(n_splits=5)
            for fold, (_, val_idx) in enumerate(kf.split(train_df)):
                train_df.loc[val_idx, 'fold'] = fold

        val_df = train_df[train_df["fold"] == split]
        val_df.pop("fold")
        if "stratified_key" in val_df:
            val_df.pop("stratified_key")
        train_df = train_df[train_df["fold"] != split]
        train_df.pop("fold")

        if "stratified_key" in train_df:
            train_df.pop("stratified_key")

        return train_df, val_df


def get_image_size_from_decoder_blocks(decoder_blocks, img_size):
    if img_size == 512:
        image_sizes = [32, 64, 128, 256, 512]
    elif img_size == 640 or img_size == 1280:
        image_sizes = [None, None, 160, None, 640]
    elif img_size == 384:
        image_sizes = [None, None, 96, None, 384]

    return image_sizes[decoder_blocks-1]


def get_lr_scheduler(config, optimizer):
    if config.name == "reduce_lr_on_plateau":
        return {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", min_lr=1e-8, patience=2),
            "monitor": "val_loss",
        }
    if config.name == "cosine_annealing":
        return {
            "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, config.max_epoch, eta_min=1e-8),
            "interval": "epoch",
        }
    if config.name == "cosine_annealing_warm_starts":
        return {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.t_0, T_mult=1, eta_min=1e-8),
            "interval": "epoch",
        }

def get_in_features(backbone, model_type):
    if 'tf_efficientnetv2_m' in backbone:
        if model_type == "pspnet":
            in_features = 80
        else:
            in_features = 512
    elif 'tf_efficientnetv2_l' in backbone:
        if model_type == "pspnet":
            in_features = 80
        else:
            in_features = 640
    elif 'efficientnet' in backbone:
        if "b7" in backbone:
            in_features = 640
        if "b6" in backbone:
            in_features = 576
        else:
            in_features = 512
    elif 'inceptionresnet' in backbone:
        if model_type == "pspnet":
            in_features = 320
        else:
            in_features = 1536
    elif 'resnet' in backbone:
        in_features = 2048
    elif 'densenet169' in backbone:
        in_features = 1664
    elif 'densenet161' in backbone:
        in_features = 2208
    elif 'vit' in backbone:
        in_features = 768
    elif 'swin' in backbone:
        in_features = 1536
    elif 'xception' in backbone:
        in_features = 2048
    elif 'inception' in backbone:
        in_features = 1536
    elif 'resnext' in backbone:
        in_features = 2048
    else:
        in_features = 2048
    return in_features

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__