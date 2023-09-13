# Space Engage: Collaborative Space Supervision for Contrastive-based Semi-Supervised Semantic Segmentation (ICCV 2023)
![cover figure.pdf](https://github.com/WangChangqi98/CSS/files/12594221/cover.figure.pdf)

This repository contains the code of **CSS** from the paper: [Space Engage: Collaborative Space Supervision for Contrastive-based Semi-Supervised Semantic Segmentation](https://arxiv.org/pdf/2307.09755.pdf)

In this paper, we propose a novel apporach to use the pseudo-labels from the logit and representation space in a collabrative way. Meanwhile, we use the softmax similarity as the indicator to tilt training in representation space.
## Updates
**Sep. 2023** -- Upload the code.

## Prepare
CSS is evaluated with two datasets: PASCAL VOC 2012 and CityScapes. 
- For PASCAL VOC, please download the original training images from the [official PASCAL site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar): `VOCtrainval_11-May-2012.tar` and the augmented labels [here](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip): `SegmentationClassAug.zip`. 
Extract the folder `JPEGImages` and `SegmentationClassAug` as follows:
```
├── data
│   ├── VOCdevkit
│   │   ├──VOC2012
│   │   |   ├──JPEGImages
│   │   |   ├──SegmentationClassAug
```
- For CityScapes, please download the original images and labels from the [official CityScapes site](https://www.cityscapes-dataset.com/downloads/): `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`.
Extract the folder `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` as follows:
```
├── data
│   ├── cityscapes
│   │   ├──leftImg8bit
│   │   |   ├──train
│   │   |   ├──val
│   │   ├──train
│   │   ├──val
```
Folders `train` and `val` under `leftImg8bit` contains training and validation images while folders `train` and `val` under `leftImg8bit` contains labels.

The data split folder of VOC and CityScapes is as follows:
```
├── VOC(CityScapes)_split
│   ├── labeled number
│   │   ├──seed
│   │   |   ├──labeled_filename.txt
│   │   |   ├──unlabeled_filename.txt
│   │   |   ├──valid_filename.txt
```
You need to change the name of folders (labeled number and seed) according to your actual experiments.

CSS uses ResNet-101 pretrained on ImageNet and ResNet-101 with deep stem block, please download from [here](https://download.pytorch.org/models/resnet101-63fe2227.pth) for ResNet-101 and [here](https://drive.google.com/file/d/131dWv_zbr1ADUr_8H6lNyuGWsItHygSb/view?usp=sharing) for ResNet-101 stem. Remember to change the directory in corresponding python file.

In order to install the correct environment, please run the following script:
```
conda create -n css_env python=3.8.5
conda activate css_env
pip install -r requirements.txt
```
It may takes a long time, take a break and have a cup of coffee!
It is OK if you want to install environment manually, remember to check CAREFULLY!

## Run
You can run our code with a single GPU or multiple GPUs.
- For single GPU users, please run the following script:
```
python prcl_sig.py [--config]
```
You need to change the file name after --config according to your actual experiments.
- For multiple GPUs users, please run the following script: 
```
run ./script/batch_train.sh
```
The seed in our experiments is 3407. You can change the label rate and seed as you like, remember to change the corresponding config files and data_split directory.
## Hyper-parameters
Some critical hyper-parameters used in the code are shown below:
|Name        | Discription  |  Value |
| :-: |:-:| :-:|
| `alpha_t`     | update speed of teacher model  |  `0.99`  |
| `alpha_p`     | update speed of prototypes  |  `0.99`  |
| `un_threshold`     | threshold in unsupervised loss  |  `0.97`  |
| `weak_threshold`     | weak threshold in contrastive loss  |  `0.7`  |
| `strong_threshold`     | strong threshold in contrastive loss  |  `0.8`  |
| `temp`     | temperature in contrastive loss  |  `0.5`  |
| `num_queries`     | number of queries in contrastive loss  |  `256`  |
| `num_negatives`     | number of negatives in contrastive loss  |  `512`  |
| `warm_up`     | warm up epochs if is needed  |  `20`  |


## Acknowledgement
The data processing and augmentation (CutMix, CutOut, and ClassMix) are borrowed from ReCo.
- ReCo: https://github.com/lorenmt/reco

Thanks a lot for their splendid work!

## Citation
If you think this work is useful for you and your research, please considering citing the following:
```
@article{wang2023space,
  title={Space Engage: Collaborative Space Supervision for Contrastive-based Semi-Supervised Semantic Segmentation},
  author={Wang, Changqi and Xie, Haoyu and Yuan, Yuhui and Fu, Chong and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2307.09755},
  year={2023}
}
```

## Contact
If you have any questions or meet any problems, please feel free to contact us.
- Changqi Wang, [wangchangqi98@gmail.com](mailto:wangchangqi98@gmail.com)
