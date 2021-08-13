from pl_script.unet_smp import LitUnetSmp
from numpy import save
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

import datetime
import hydra
import os
import uuid

import pytorch_lightning as pl

from pl_script.faster_rcnn_detector import LitFasterRCNN
from pl_script.unet_smp import LitUnetSmp
from pl_script.unet_smp_2c import LitUnetSmp2C
from pl_script.timm_2c import LitTimm2C
# from pl_script.unet_smp_downconv import LitUnetSmpDownConv
# from pl_script.unet_smp_mask_cond import LitUnetSmpMaskCond
from pl_script.timm import LitTimm
from pl_script.unet_smp_nih import LitUnetSmpNIH
from pl_script.timm_nih import LitTimmNIH


def get_model(config, load_path=None):
    if config.model_name == "unet_smp":
        if load_path:
            return LitUnetSmp.load_from_checkpoint(load_path, config=config)
        else:
            return LitUnetSmp(config)
    elif config.model_name == "unet_smp_2c":
        return LitUnetSmp2C(config)
    elif config.model_name == "unet_smp_downconv":
        return LitUnetSmpDownConv(config)
    elif config.model_name == "unet_smp_mask_cond":
        return LitUnetSmpMaskCond(config)
    elif config.model_name == "timm":
        return LitTimm(config)
    elif config.model_name == "timm_2c":
        return LitTimm2C(config)
    elif config.model_name == "timm_nih":
        return LitTimmNIH(config)
    elif config.model_name == "unet_smp_nih":
        return LitUnetSmpNIH(config)
    if config.model_name == "FasterRCNN":
        return LitFasterRCNN(config)


def get_model_name_wandb(config):
    if config.model_name == "unet_smp":
        model_name = config.unet_smp.backbone_name.replace("-","_")
    elif config.model_name == "timm":
        model_name = f'timm-{config.model_config.backbone_name.replace("-","_")}'
    else:
        model_name = config.model_name

    if config.album == True:
        aug_indicator = "-aug"
    else:
        aug_indicator = ""

    if config.focal_loss == True:
        focal_indicator = "-focal-loss"
    else:
        focal_indicator = ""

    img_size_indicator = f"-{config.img_size}"

    return f"{model_name}{img_size_indicator}{aug_indicator}{focal_indicator}"


@hydra.main(config_path=".", config_name="config")
def run(config):
    model_name_wandb = get_model_name_wandb(config)
    if config.logger:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            entity="monet-kaggle",
            project=config.project,
            name=model_name_wandb,
            config=config,
        )
    else:
        logger = pl.loggers.TestTubeLogger(
            "output", name=f"siiim-covid")
        logger.log_hyperparams(config)

    if type(config.gpus) == str:
        config.gpus = [int(config.gpus.replace("cuda:",""))]

    save_path = os.path.join(config.root, "pl_output")

    callbacks = []
    if config.lr_schedule.name:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config.es_patience is not None:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=config.es_patience,
            verbose=False,
            mode='min'
        ))
    if config.second_train:
        callbacks.append(ModelCheckpoint(
            dirpath=f"/shared/lorenzo/second_trains_{model_name_wandb}_{config.seed}_{config.img_size}",
            filename=f"{model_name_wandb}-"+"{epoch:02d}-"+f"{config.cv}{config.fold}-"+"{val_loss:.4f}",
            save_top_k=-1,
        ))
    if not config.model_name == "FasterRCNN":
        callbacks.append(ModelCheckpoint(
            dirpath="/home/lorenzo/siim-covid/pl_output",
            filename=f"{model_name_wandb}-"+"{epoch:02d}-"+f"{config.cv}{config.fold}-"+"{val_loss:.4f}",
            monitor="val_loss",
            mode="min"
        ))

    pl.seed_everything(config.seed)

    trainer = pl.Trainer(
        precision=16,
        gradient_clip_val=0.5,
        accumulate_grad_batches=config.grad_accum,
        auto_lr_find=True if config.lr_finder else None,
        callbacks=callbacks,
        deterministic=True,
        check_val_every_n_epoch=1,
        gpus=config.gpus,
        logger=logger,
        max_epochs=config.epoch,
        weights_summary="top",
        # accelerator='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False),
    )

    model = get_model(config, load_path=config.load_path)

    if config.lr_finder:
        lr_finder = trainer.tuner.lr_find(model, min_lr=1e-8, max_lr=1e-1, num_training=100)
        model.hparams.lr = lr_finder.suggestion()
        print(model.hparams.lr)
    else:
        trainer.fit(model)
        trainer.test()


if __name__ == '__main__':
  run()