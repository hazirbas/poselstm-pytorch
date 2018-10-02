# PoseLSTM and PoseNet implementation in PyTorch
This is the PyTorch implementation for PoseLSTM and PoseNet, developed based on [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) code.

## Prerequisites
- Linux
- Python 3.5.2
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Clone this repo:
```bash
git clone https://github.com/hazirbas/posenet-pytorch
cd posenet-pytorch
pip install -r requirements.txt
```

### PoseNet train/test
- Download a Cambridge Landscape dataset (e.g. [KingsCollege](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)) under datasets/ folder.
- Compute the mean image
```bash
python util/compute_image_mean.py --dataroot datasets/KingsCollege --height 256 --width 455 --save_resized_imgs
```
- Train a model:
```bash
python train.py --model posenet --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/beta500 --beta 500 --gpu 0
```
- To view training errors and loss plots, set `--display_id 1`, run `python -m visdom.server` and click the URL http://localhost:8097. Checkpoints are saved under `./checkpoints/posenet/KingsCollege/beta500/`.
- Test the model:
```bash
python test.py --model posenet  --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/beta500 --gpu 0
```
The test errors will be saved to a text file under `./results/posenet/KingsCollege/beta500/`.

### PoseLSTM train/test
- Train a model:
```bash
python train.py --model poselstm --dataroot ./datasets/KingsCollege --name poselstm/KingsCollege/beta500 --beta 500 --niter 1200 --gpu 0
```
- Test the model:
```bash
python test.py --model poselstm --dataroot ./datasets/KingsCollege --name poselstm/KingsCollege/beta500 --gpu 0
```

### Initialize the network with the pretrained googlenet trained on the Places dataset
If you would like to initialize the network with the pretrained weights, download the places-googlenet.pickle file under the *pretrained_models/* folder:
``` bash
wget https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/places-googlenet.pickle
```
### Optimization scheme and loss weights
* We use the training scheme defined in [PoseLSTM](https://arxiv.org/abs/1611.07890)
* Note that mean subtraction **is not used** in PoseLSTM models
* Results can be improved with a hyper-parameter search

| Dataset       | beta | PoseNet (CAFFE) | PoseNet | PoseLSTM (TF) | PoseLSTM |
| ------------- |:----:| :----: | :----: | :----: | :----: |
| King's College  | 500  | 1.92m 5.40° | [1.19m 4.51°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/posenet/KingsCollege.zip) | 0.99m 3.65° | [0.90m 3.96°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/poselstm/KingsCollege.zip)|
| Old Hospital   | 1500 | 2.31m 5.38° | [1.91m 4.05°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/posenet/OldHospital.zip) | 1.51m 4.29° | [1.79m 4.28°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/poselstm/OldHospital.zip)|
| Shop Façade    | 100  | 1.46m 8.08° | [1.30m 8.13°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/posenet/ShopFacade.zip) | 1.18m 7.44° | [0.98m 6.20°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/poselstm/ShopFacade.zip)|
| St Mary's Church | 250  | 2.65m 8.48° | [1.89m 7.27°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/posenet/StMarysChurch.zip) | 1.52m 6.68° | [1.68m 6.41°](https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/poselstm/StMarysChurch.zip) |

## Citation
```
@inproceedings{PoseNet15,
  title={PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization},
  author={Alex Kendall, Matthew Grimes and Roberto Cipolla },
  journal={ICCV},
  year={2015}
}
@inproceedings{PoseLSTM17,
  author = {Florian Walch and Caner Hazirbas and Laura Leal-Taixé and Torsten Sattler and Sebastian Hilsenbeck and Daniel Cremers},
  title = {Image-based localization using LSTMs for structured feature correlation},
  month = {October},
  year = {2017},
  booktitle = {ICCV},
  eprint = {1611.07890},
  url = {https://github.com/NavVisResearch/NavVis-Indoor-Dataset},
}
```
## Acknowledgments
Code is inspired by [pytorch-CycleGAN-and-pix2pix]((https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)).
