# FoMo4Wheat 
The official implementation of the paper **FoMo4Wheat: Toward reliable crop vision foundation models with globally curated data**. [Paper](https://arxiv.org/abs/2509.06907)
Contact：Shouyang Liu (shouyang.liu@njau.edu.cn)，Hao Lu (hlu@hust.edu.cn)，Yanfeng Ding (dingyf@njau.edu.cn)

# Abstract
Vision-driven in-field crop monitoring is essential for advancing digital agriculture whether supporting commercial decisions on-farm or augmenting research experiments in breeding and agronomy. Existing crop vision models struggle to generalize across fine-scale, highly variable canopy structures, and fluctuating outdoor environments. In this work, we present FoMo4Wheat, one of the first crop-orientated vision foundation models and demonstrate that delivers strong performance across a wide range of agricultural vision tasks. Centered on wheat, the most globally significant food crop, we curated ImAg4Wheat—the largest and most diverse wheat image dataset to date. It comprises 2.5 million high-resolution images collected over a decade from breeding and experimental fields, spanning more than 2,000 genotypes and 500 distinct environmental conditions across 30 global sites. A suite of FoMo4Wheat models was pre-trained using self-supervised learning on this dataset. Benchmark results across ten crop-related downstream tasks show that FoMo4Wheat consistently outperforms state-of-the-art models trained on general-domain datasets. Beyond strong cross-task generalization within wheat crops, FoMo4Wheat is highly robust in limited-data regimes but on previously unseen crop data. Notably, it contributes significantly to vision tasks in rice and multiplw crop/weed images, highlighting its cross-crop adaptability. In delivering one of the first open-source foundation models for wheat, our results demonstrate the value of such crop-specific foundation models that will support the development of versatile high-performing vision systems in crop breeding and precision agriculture. 

# Demo
The demonstration website for inferring embeddings is located at [Demo](https://fomo4wheat.phenix-lab.com/).

https://github.com/user-attachments/assets/2f2f21b4-4638-41c6-8bdf-37d8ad458eb6

🎥 **Visualization of Unlabeled wheat features.**

# Method
<img width="1267" height="1459" alt="Fig 1" src="https://github.com/user-attachments/assets/1d095d9b-2de4-4080-b68c-7da83f12edc1" />
<b>Fig 1.</b> Overview of ImAg4Wheat dataset and FoMo4Wheat model.

# Installation
The training and evaluation code is developed with PyTorch 2.5.1 and requires Linux environment with multiple third-party dependencies. To set up all required dependencies for training and evaluation, please follow the instructions below:
```
conda env create -f conda.yaml
conda activate FoMo4Wheat
```

# Data Preparation
ImAg4Wheat comprises 2,500,000 million images over 2,000 wheat genotypes cultivated under 500 distinct environmental conditions across 30 sites in 10 countries spanning a decade, covering the full crop growth cycle. [ImAg4Wheat](https://huggingface.co/datasets/PheniX-Lab/ImAg4Wheat)
(Note: The complete dataset will be made publicly available after the peer-review process of the associated paper is completed.)

# Pretrained models
| model                | # of params  | download  | 
| :---------------------:| -----------: |:--------------:| 
| ViT-B/14             |   86 M   | [FoMo4Wheat_base.pth](https://huggingface.co/PheniX-Lab/FoMo4Wheat/blob/main/weight/FoMo4Wheat_base.pth) |
| ViT-L/14             |   300 M  | [FoMo4Wheat_large.pth](https://huggingface.co/PheniX-Lab/FoMo4Wheat/blob/main/weight/FoMo4Wheat_large.pth) |
| ViT-G/14             |   1,100 M  | [FoMo4Wheat_giant.pth](https://huggingface.co/PheniX-Lab/FoMo4Wheat/blob/main/weight/FoMo4Wheat_giant.pth) | 

# Training
**Training FoMo4Wheat on ImAg4Wheat**

Run FoMo4Wheat training on 6 A800-80GB nodes (48 GPUs) in a SLURM cluster environment with submitit:
```
MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 python FoMo4Wheat/run/train/
    --nodes 6 \
    --config-file FoMo4Wheat/configs/train/vitg_14_224.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=TestDataset:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

# License
FoMo4Wheat code and model weights are released under the MIT License. See LICENSE for additional details.


# Citation
If you use our project in your research or wish to refer to the results of the project, please use the following BibTeX entry.

```bibtex
@article{2025FoMo4Wheat,
  title={FoMo4Wheat: Toward reliable crop vision foundation models with globally curated data}, 
  author={Bing Han, Chen Zhu, Dong Han, Rui Yu, Songliang Cao, Jianhui Wu, Scott Chapman, Zijian Wang, Bangyou Zheng, Wei Guo, Marie Weiss, Benoit de Solan, Andreas Hund, Lukas Roth,  Kirchgessner Norbert, Andrea Visioni, Yufeng Ge, Wenjuan Li, Alexis Comar, Dong Jiang, Dejun Han, Fred Baret, Yanfeng Ding, Hao Lu and Shouyang Liu},
  journal={arXiv:2509.06907},
  year={2025}
}
```

# Collaborators 
<img width="1929" height="1057" alt="logo" src="https://github.com/user-attachments/assets/6bee265b-8a2d-4ae7-aeec-5cb71a29818e" />

