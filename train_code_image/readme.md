# 8th Solution of [SIIM-FISABIO-RSNA COVID19 Detection Competition](https://www.kaggle.com/c/siim-covid19-detection)

## Environment Setup
Anaconda virtual env is recommended. Python version is 3.7.
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install openmim
mim install mmdet
git clone https://github.com/ultralytics/yolov5
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
mkdir data-512
cd data-512
kaggle datasets download -d jihunlorenzopark/covidsiim512
unzip covidsiim512.zip
cp ../data-640/train_psl_none.csv .
cp ../data-640/meta.csv .
```

## Run
### YOLOv5
YOLOv5 requires a certain bounding box style and directory structure. So, first run 'get_yolo_bbox.ipynb', which convert label bbox to yolo style bbox, and 'make_yolo_dataset.ipynb', which makes directory structure for yolo.

and then, run 'train_yolo.ipynb' to train yolo models.

### Cascade RCNN
Using mmdetection, run 'train_cascadercnn.ipynb'.