# Deepfake Detection based on Microexpression Analysis

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#pretrained-models">Pretrained Models</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#data-structure">Data Structure</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains practical results of my Final Qualifying Work on topic «Synthetic Video Recognition Based on Facial Micro-expression Analysis for Online Conferences». In general there are 2 models: **FAUModel** for video analysis based on Facial Action Units and **DepthModel** for image analysis based on Depth Estimation.

## Installation
Clone repo and create environment:
```bash
git clone https://github.com/7embl4/dfdet.git
cd dfdet
conda create --name dfdet python=3.10
```

For GPU support install `torch` and `torchvision` separately:
```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

Install other dependencies:
```bash
python -m pip install -r requirements.txt
```

Also the [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2.git) is necessary:
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_model
```

## Models
Load models from [here](https://drive.google.com/drive/folders/1AHUIwPd3cJ00vFt58Q8fKbcVr0924qG9?usp=drive_link) 
and put them in `models` folder in root directory in the following format:
```bash
models
└── fau
  ├── model_best.pth
  ├── resnet.pt
└── depth
  ├── model_best.pth
└── face_detection_yunet_2023mar.onnx
```

## Data Preprocessing
For both training and inferencing there is specific data structure (more about structure in `preprocess_data.py`).
You can obtain such structure using `preprocess_data.py` script. Only thing, that your data should be in the following format:
```bash
DataDirectory
└── real
  ├── video1.mp4
  ├── video2.mp4
  ├── ...
└── fake
  ├── video1.mp4
  ├── video2.mp4
  ├── ...
```

or in case of different types of real and fake videos

```bash
DataDirectory
└── real
  └── real_type1
    ├── video1.mp4
    ├── video2.mp4
    ├── ...
  └── real_type2
    ├── video1.mp4
    ├── video2.mp4
    ├── ...
  ...
└── fake
  └── fake_type1
    ├── video1.ext
    ├── video2.ext
    ├── ...
  └── fake_type2
    ├── video1.ext
    ├── video2.ext
    ├── ...
  ...
```

When your data in such format just run script:
```bash
python preprocess_data.py --data_path DataDirectory
```

Note, that the preprocessing may take some time for video, since there is face detecting. 
Duration depends on CPU (for example, it's less than one second for a video on Ryzen 5 5600).

## Training
To train a model with basic parameters run the following command:
```bash
python train.py --confing-name pretrain_{fau/depth}_expert
```

Also you can configure your own parameters straight from command line:
```bash
python train.py 
  --confing-name pretrain_{fau/depth}_expert 
  dataloader.batch_size=16 
  trainer.n_epochs=100 
  optimizer.lr=0.001 
  etc...
```

For more parameters check corresponding config in `src/configs` folder.

## Inference

You may run inference on your data:
```bash
python inference.py 
  --data_path path_to_dataset
  --data_type video
  --save_mistakes 
```

`save_mistakes` will save classification mistakes in `saved/mistakes.json` (disabled by default).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)