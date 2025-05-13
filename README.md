<div align="center">
<h1> 🔥DMAF-Net🎉 </h1>
<h3>Cross-Modal Clustering-Guided Negative Sampling for Self-Supervised Joint Learning from Medical Images and Reports</h3>

[Hongxing Li](https://orcid.org/0009-0002-7958-3976)<sup>1</sup> ,[Zunhui Xia](https://orcid.org/0009-0008-6706-5817)<sup>1</sup> ,[Libin Lan](https://orcid.org/0000-0003-4754-813X)<sup>1</sup> :email:</sup>

🏢 <sup>1</sup> College of Computer Science and Engineering, Chongqing University of Technology.  (<sup>:email:</sup>) corresponding author.
</div>

## 👇Overview
  
### • Abstract
Incomplete multi-modal medical image segmentation faces critical challenges from modality imbalance, including imbalanced modality missing rates and heterogeneous modality contributions. Existing methods are constrained by idealized complete-modality assumptions and fail to dynamically balance contributions while ignoring structural relationships between modalities, resulting in suboptimal performance in real-world clinical scenarios. To address these limitations, we propose the Dynamic Modality-Aware Fusion Network (DMAF-Net), which integrates three key innovations: 1) A Dynamic Modality-Aware Fusion (DMAF) module that combines transformer attention with adaptive masking to suppress missing-modality interference while dynamically weighting modality contributions through attention maps; 2) A synergistic relation distillation and prototype distillation framework that enforces global-local feature alignment via covariance consistency and masked graph attention, while ensuring semantic consistency through cross-modal class-specific prototype alignment; 3) A Dynamic Training Monitoring Strategy (DTMS) that stabilizes optimization under imbalanced missing rates by tracking distillation gaps in real-time, and adaptively reweighting losses and scaling gradients to balance convergence speeds across modalities. Extensive experiments on BraTS2020 and MyoPS2020 demonstrate that DMAF-Net outperforms existing methods for incomplete multi-modal medical image segmentation. This work not only advances the field of incomplete multi-modal medical image segmentation but also provides more reliable technical support for real-world clinical diagnosis.
### • DMAF-Net
<div align="center">
<img src="assets/cluster.png" />
</div>

### • Architecture
<div align="center">
<img src="assets/framework.png" />
</div>

###  Installation
To install Python dependencies:
```
pip install -r requirements.txt
```
### Dataset downloading
Datasets we used are as follows:
- **MIMIC-CXR**: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset as the radiographs. Paired medical reports can be downloaded in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip).

- **CheXpert**: We downloaded the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset which consisting of 224,316 chest radiographs of 65,240 patients.

- **RSNA**: We used the stage 2 of RSNA dataset in [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data). 

- **COVIDx**: We used the version 6 of COVIDx dataset in [Kaggle](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2).

- **SIIM**: We downloaded the stage 1 of SIIM dataset in [Kaggle](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data).

- **Object-CXR**: We downloaded the object-CXR dataset in its [official website](https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5).

After downloading datasets, please check if the path in `cgns/constants.py` is correct.

### Data Preprocessing
We preprocessed these datasets and split the dataset into train/val/test set using the code in `cgns/preprocess`.

### Pre-training
```
CUDA_VISIBLE_DEVICES=0,1 python cgns_module.py --gpus 2 --strategy ddp
```

### Finetune on downstream tasks
#### Linear classification
```
CUDA_VISIBLE_DEVICES=1 python cgns_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.01
```
We can use `--dataset` to set specific dataset for finetuning. Here, 3 datsets are available: chexpert, rsna and covidx.

#### Object detection
```
CUDA_VISIBLE_DEVICES=0 python cgns_detector.py --devices 1 --dataset rsna --data_pct 1 --learning_rate 5e-4
```
Here, 2 datsets are available: rsna and object_cxr.

#### Semantic segmentation
```
CUDA_VISIBLE_DEVICES=0 python cgns_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 16 --learning_rate 5e-4
```
Here, 2 datsets are available: rsna and siim.
