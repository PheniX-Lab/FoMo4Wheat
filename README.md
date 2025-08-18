# FoMo4Wheat 
The official implementation of the paper **Crop-specific Vision Foundation Model enabling Generalized Field Monitoring**
# Abstract
Vision-driven in-field crop monitoring is essential for advancing digital agriculture whether supporting commercial decisions on-farm or augmenting research experiments in breeding and agronomy. Existing crop vision models struggle to generalize across fine-scale, highly variable canopy structures, and fluctuating outdoor environments. In this work, we present FoMo4Wheat, one of the first crop-orientated vision foundation models and demonstrate that delivers strong performance across a wide range of agricultural vision tasks. Centered on wheat, the most globally significant food crop, we curated ImAg4Wheat—the largest and most diverse wheat image dataset to date. It comprises 2.5 million high-resolution images collected over a decade from breeding and experimental fields, spanning more than 2,000 genotypes and 500 distinct environmental conditions across 30 global sites. A suite of FoMo4Wheat models was pre-trained using self-supervised learning on this dataset. Benchmark results across ten crop-related downstream tasks show that FoMo4Wheat consistently outperforms state-of-the-art models trained on general-domain datasets. Beyond strong cross-task generalization within wheat crops, FoMo4Wheat is highly robust in limited-data regimes but on previously unseen crop data. Notably, it contributes significantly to vision tasks in rice and multiplw crop/weed images, highlighting its cross-crop adaptability. In delivering one of the first open-source foundation models for wheat, our results demonstrate the value of such crop-specific foundation models that will support the development of versatile high-performing vision systems in crop breeding and precision agriculture. 
# Demo
![Demo](./video.mp4)
# Method
<img width="1256" height="1460" alt="image" src="https://github.com/user-attachments/assets/89b475ab-d8c3-4997-a4ec-bd4062b2f986" />
**Fig 1. Overview of ImAg4Wheat dataset and FoMo4Wheat model.**

# Installation
The training and evaluation code is developed with PyTorch 2.5.1 and requires Linux environment with multiple third-party dependencies. To set up all required dependencies for training and evaluation, please follow the instructions below:
```
conda env create -f conda.yaml
conda activate FoMo4Wheat
```
# Training
```
MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 python dinov2/run/train/
    --nodes 6 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=TestDataset:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```
# License
