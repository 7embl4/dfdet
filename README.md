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

```bash
git clone https://github.com/7embl4/dfdet.git
cd dfdet
conda create --name dfdet python=3.10
python -m pip install -r requirements.txt
```

This installation is CPU only. For GPU support install `torch` and `torchvision` separately:
```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

## Pretrained Models
For now pretrained models aren't available, since the project is still on go. But you can train models by yourself following the steps below.

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

After training you may run inference on your data:
```bash
python synthesize.py 
  --data_path {path_to_dataset}
  --data_type {video/image}  
  --save_mistakes 
```

`save_mistakes` will save classification mistakes in `mistakes.json` (disabled by default).

## Data Stucture
Note that there is specific data structure for both training and inferencing. Image dataset must be in the following format, where `ext` is *jpg*, *jpeg* or *png*:
```bash
ImageDatasetDirectory
└── real
    └── real_type1
        ├── image1.ext
        ├── image2.ext
        ├── ...
    ...
└── fake
    └── fake_type1
        ├── image1.ext
        ├── image2.ext
        ├── ...
    ...
```

For video dataset the structure is:
```bash
VideoDatasetDirectory
└── real
    └── real_type1
        └── faces
          ├── face1.npz
          ├── face2.npz
          ├── ...
        └── videos
          ├── video1.ext
          ├── video2.ext
          ├── ...
        ...
    ...
└── fake
    └── fake_type1
        └── faces
          ├── face1.npz
          ├── face2.npz
          ├── ...
        └── videos
          ├── video1.ext
          ├── video2.ext
          ├── ...
        ...
    ...
```

Where `ext` is *mp4*, *avi*, *mov* or *mpeg*. And `face.npz` is a file with face coordinates for video of format:
```bash
[
  [frame_id1, x1, x2, y1, y2],
  [frame_id2, x1, x2, y1, y2],
  ...,
  [frame_idN, x1, x2, y1, y2],
]
```

You can get these `face.npz` files with `extract_bboxes.py` script and [YuNet model](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)