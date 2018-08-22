# Posenet implementation in PyTorch
## [[PoseLSTM]](https://github.com/HEIMDAL13/posenet-pytorch/tree/lstm)

This is an ongoing PyTorch implementation for PoseNet, developed based on [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) code.

## Prerequisites
- Linux
- Python 3.5.2
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
- Clone this repo:
```bash
git clone https://github.com/hazirbas/posenet-pytorch
cd posenet-pytorch
pip install -r requirements.txt
```

### PoseNet train/test
- Download a Cambridge Landscape dataset (e.g. [KingsCollege](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)) under datasets/ folder.
- Compute image mean
```bash
python util/compute_image_mean.py --dataroot datasets/KingsCollege --height 256 --width 455 --save_resized_imgs
```
- Train a model:
```bash
python train.py --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/beta500 --beta 500 --gpu 0
```
- To view training errors and loss plots, set `--display_id 1`, run `python -m visdom.server` and click the URL http://localhost:8097. Checkpoints are saved under `./checkpoints/posenet/KingsCollege/beta500/`.
- Test the model:
```bash
python test.py --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/beta500 --gpu 0
```
The test errors will be saved to a text file under `./results/posenet/KingsCollege/beta500/`.

### Initialize model with pretrained googlenet on Places dataset
If you would like to initialize the model with pretrained weights, download the places-googlenet.pickle file under pretrained_models/ folder:
``` bash
wget https://vision.in.tum.de/webarchive/hazirbas/posenet-pytorch/places-googlenet.pickle
```
### Optimization scheme and loss weights
We use the training scheme defined in [PoseLSTM](https://arxiv.org/abs/1611.07890). Best models are determined by the median error wrt position.

| Dataset       | beta | PoseNet | Ours | Model |
| ------------- |:----:| :----: | :----: | :----: |
| King's College  | 500  | 1.92m 5.40° | 1.34m 4.33° | [epoch495](https://vision.in.tum.de/webarchive/hazirbas/posenet-pytorch/KingsCollege.zip) |
| Old Hospital   | 1500 | 2.31m 5.38° | 2.58m 5.77° | [epoch455](https://vision.in.tum.de/webarchive/hazirbas/posenet-pytorch/OldHospital.zip) |
| Shop Façade    | 100  | 1.46m 8.08° | 1.44m 8.26° | [epoch470](https://vision.in.tum.de/webarchive/hazirbas/posenet-pytorch/ShopFacade.zip) |
| St Mary's Church | 250  | 2.65m 8.48° | 2.40m 9.56° | [epoch470](https://vision.in.tum.de/webarchive/hazirbas/posenet-pytorch/StMarysChurch.zip) |

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
