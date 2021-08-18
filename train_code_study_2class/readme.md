# 8th Solution of [SIIM-FISABIO-RSNA COVID19 Detection Competition](https://www.kaggle.com/c/siim-covid19-detection)

## Environment Setup
Anaconda virtual env is recommended. Python version is 3.7.
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Dataset Preparation
We need the resized datasets. Resize are done by [this kernel](https://www.kaggle.com/jihunlorenzopark/multiprocess-siim-covid-19-convert-to-jpg-256px). Each dataset contains `meta.csv` and `train_psl_none.csv`. `meta.csv` contains the original image size and `train_psl_none.csv` contains labels, PatientID, ... . `train_psl_none.csv` contains other pseudo labeled columns but not used. Basically, the required columns are `"id","boxes","label","StudyInstanceUID","Negative for Pneumonia","Typical Appearance","Indeterminate Appearance","Atypical Appearance","PatientID"`

```bash
pip install kaggle # Kaggle API
vi ~/.kaggle/kaggle.json # Kaggle Profile - Account Tab - API - Create New API Token  ex) {"usernames":"jihunlorenzopark", "key": "xxxxx"}

# !!!Note that all dataset folders should be located in project root!!!
mkdir data-640 # Naming format is data-{image_size}
cd data-640
kaggle datasets download -d jihunlorenzopark/siim-fisabio-rsna-data-640
unzip siim-fisabio-rsna-data-640.zip

# !!!Note that all dataset folders should be located in project root!!!
mkdir data-384
cd data-384
kaggle datasets download -d jihunlorenzopark/covidsiimresized
unzip covidsiimresized.zip
cp ../data-640/train_psl_none.csv .
cp ../data-640/meta.csv .

# !!!Note that all dataset folders should be located in project root!!!
mkdir data-512
cd data-512
kaggle datasets download -d jihunlorenzopark/covidsiim512
unzip covidsiim512.zip
cp ../data-640/train_psl_none.csv .
cp ../data-640/meta.csv .

# !!!Note that all dataset folders should be located in project root!!!
# For NIH CHEST X-ray dataset pretraining, run the below
mkdir data-nih
cd data-nih
kaggle datasets download -d jihunlorenzopark/nih640
unzip nih640.zip
```

## Run
Training script is `run.py` configured by `config.yaml` and `yaml`file of each model in `./model` directory.

Before running the script, update `root` parameter in `config.yaml` file. For example,
```
...
root: /home/lorenzo/kaggle_model/train_code_study_2class # Project root directory
...
```

Also, unless you use wandb for logging training metrics/losses, turn off `logger` option to use default build-in logger of pytorch. i.e. `logger=False`

For reproducing all models, please run the following scripts in order, on the right hardware settings. Each hardware settings of each bash script is commented on the top.
**IMPORTANT: All scripts assume the project root path is `/home/lorenzo/kaggle_model/train_code_study_2class`. Please modify all path related variables before you run it indicated below by coments.**
```bash
bash 1.pretraining.bash # Before running this script, update the `data_root` parameters in this file.

bash 2.unet_training_with_nihpretrained.bash # Before running this script, generate pretrained model weight in `gen_pretrain.ipynb` and update the `unet_smp.pretrain_path` parameter.
bash 2.unet_training_with_nihpretrained.bash # Before running this script, generate pretrained model weight in `gen_pretrain.ipynb` and update the `model_config.pretrained_path` parameter.

bash 3.unet_training.bash
bash 4.finetuning_2class.bash
```

# Wandb Logs
Submitted models training info can be found this [wandb link](https://wandb.ai/monet-kaggle/siim-covid-final%20submission?workspace=user-lorenzo-kaggle)