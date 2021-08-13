## Trained on 1 RTX3090 GPU
## Unet++ Efficientnetv2-m Training with img size 640
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" fold=0 gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" fold=1 gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" fold=2 gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" fold=3 gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" fold=4 gpus=1 es_patience=6
## Unet++ Efficientnetv2-m Training with img size 512
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_m fold=0 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=512 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_m fold=1 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=512 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_m fold=2 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=512 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_m fold=3 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=512 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_m fold=4 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=512 unet_smp.neck_type=F gpus=1 es_patience=6
## Unet++ Efficientnetv2-l Training with img size 384
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_l fold=0 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=384 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_l fold=1 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=384 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_l fold=2 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=384 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_l fold=3 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=384 unet_smp.neck_type=F gpus=1 es_patience=6
python run.py lr=3e-4 logger=True project="siim-covid-reproduce" unet_smp.backbone_name=tu-tf_efficientnetv2_l fold=4 unet_smp.model_type=unetplusplus unet_smp.decoder_blocks=3 mask_type=both unet_smp.classes=2 cv=sgkf img_size=384 unet_smp.neck_type=F gpus=1 es_patience=6
