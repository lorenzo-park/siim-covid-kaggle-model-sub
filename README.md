The solution archive contains following three parts. Note that our model consists three parts, study-level classification model, image-level opacity detection model, 2class binary classification model.

Study-level classification model and 2class binary classification model training code in `train_code_study_2class` folder and image-level opacity detection model in `train_image` To reproduce training, please follow the instruction for each folder.

# ARCHIVE CONTENTS
comp_preds                  : model predictions
train_code_study_2class     : code to train study 4-class and image binary classification models from scratch.
train_code_image            : code to train image opacity detection models from scratch.
predict_code                : code to generate predictions from model binaries. we attached a ipynb file runned on Kaggle GPU Notebook (Container version: 2021-04-22).

# HARDWARE: (The following specs were used to create the original solution and it is different by models)
## `train_code_study_2class`
Debian GNU/Linux 10
AMD Ryzen Threadripper PRO 3995WX 64-Cores
1 x NVIDIA RTX A6000

Debian GNU/Linux 10
AMD Ryzen Threadripper 3970X 32-Core Processor
1 x NVIDIA RTX 3090

## `train_code_image`
Debian GNU/Linux 10
AMD Ryzen Threadripper 3970X 32-Core Processor
2 x NVIDIA RTX 3090

# SOFTWARE
Python 3.7
CUDA 11.2
Driver version: 460.84

# Commands to Reproduce
Please follow readme files in each train folders, `train_code_study_2class`, `train_code_image`