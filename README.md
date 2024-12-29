<div align="center">

# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

[Jinglei Zhang]()<sup>1</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>2</sup> &emsp; [Chao Ma](https://scholar.google.com/citations?user=syoPhv8AAAAJ&hl=en)<sup>1</sup> &emsp; [Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>2</sup> &emsp;  

<sup>1</sup>Shanghai Jiao Tong University, China
<sup>2</sup>Imperial College London, UK <br>

<a href='https://hawor-project.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
</div>

This is the official implementation of **[HaWoR](https://hawor-project.github.io/)**, a hand reconstruction model in the world coordinates:

![teaser](assets/teaser.png)

## Installation
 
### Installation
```
git clone --recursive https://github.com/ThunderVVV/HaWoR.git
cd HaWoR
```

The code has been tested with PyTorch 1.13 and CUDA 11.7. It is suggested to use an anaconda environment to install the the required dependencies:
```bash
conda create --name hawor python=3.10
conda activate hawor

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# Install requirements
pip install -r requirements.txt
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0
```

### Install masked DROID-SLAM:

```
cd thirdparty/DROID-SLAM
python setup.py install
```

Download DROID-SLAM official weights [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing), put it under `./weights/external/`.

### Install Metric3D

Download Metric3D official weights [metric_depth_vit_large_800k.pth](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link), put it under `thirdparty/Metric3D/weights`.

### Download the model weights

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./weights/hawor/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and put the hand model to the `_DATA/data/mano/MANO_RIGHT.pkl` and `_DATA/data_left/mano_left/MANO_LEFT.pkl`. 

Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
## Demo

For visualizaiton in world view, run with:
```bash
python demo.py --video_path ./example/video_0.mp4  --vis_mode world
```

For visualizaiton in camera view, run with:
```bash
python demo.py --video_path ./example/video_0.mp4 --vis_mode cam
```

## Training
The training code will be released soon. 

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HaMeR](https://github.com/geopavlakos/hamer/)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [SLAHMR](https://github.com/vye16/slahmr)
- [TRAM](https://github.com/yufu-wang/tram)
- [CMIB](https://github.com/jihoonerd/Conditional-Motion-In-Betweening)


## License 
HaWoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.
## Citing
If you find HaWoR useful for your research, please consider citing our paper:

```bibtex

```
