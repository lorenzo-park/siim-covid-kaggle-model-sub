## Trained on 1 RTX A6000 GPU
## Swin Transformer pretraining
python run.py lr=5e-5 logger=True model=timm_nih project="siim-covid-reproduce" model_config.backbone_name=swin_large_patch4_window12_384_in22k fold=0 cv=sgkf img_size=384 gpus=1 es_patience=6 data_root=/home/lorenzo/kaggle_model/train_code_study_2class/data-nih cv=gkf batch_size=32
## Efficientnet Pretraining
python run.py lr=5e-4 logger=True model=unet_smp_nih project="siim-covid-reproduce" fold=0 cv=sgkf img_size=640 gpus=1 es_patience=6 data_root="/home/lorenzo/kaggle_model/train_code_study_2class/data-nih" cv=gkf batch_size=32 epoch=5
