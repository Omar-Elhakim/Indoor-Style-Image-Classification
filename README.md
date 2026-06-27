# Indoor Style Classification

Classify photos of indoor spaces into 17 interior-design styles using deep
learning models built and trained in PyTorch.

## Overview

This project tackles a multi-class image classification problem: given a photo
of an indoor space, predict its interior-design **style** (e.g. *scandinavian*,
*industrial*, *boho*). It was developed for a Kaggle-style competition, so the
inference scripts produce a `submission.csv` in `id,label` format.

Several architectures are explored side by side — some implemented and trained
from scratch, others standard CNNs — to compare how each handles a fairly
fine-grained and class-imbalanced dataset.

## The 17 style classes

```
asian          boho           coastal        contemporary   country
eclectic       farmhouse      industrial     mediterranean  minimalist
modern         rustic         scandinavian   shabby-chic    southwestern
traditional    tropical
```

The training data is class-imbalanced (e.g. *boho* has ~918 images while
*minimalist* has ~555); the main notebook handles this with a
`WeightedRandomSampler` and online data augmentation.

## Models

| Notebook | Model | Notes |
|----------|-------|-------|
| `notebooks/main.ipynb` | LightVGG | Lightweight VGG-style CNN trained from scratch; weighted sampling + augmentation |
| `notebooks/vgg16-from-scratch.ipynb` | VGG16 | Full VGG16 trained from scratch |
| `notebooks/GoogleNet.ipynb` | GoogleNet | Inception-style CNN |
| `notebooks/mobilenetv2-model.ipynb` | MobileNetV2 | Reimplemented MobileNetV2 (inverted residuals) |
| `notebooks/vit-model.ipynb`, `notebooks/vit-model-from-scratch.ipynb` | Vision Transformer | Custom ViT (patch embedding + Transformer encoder) built from scratch |

Each model has a matching standalone inference script under `scripts/`.

## Tech stack

- **Python** + **PyTorch** / **torchvision**
- **Pillow** for image loading
- **NumPy** / **pandas** for data handling and submission output
- **scikit-learn** for stratified splitting
- **matplotlib** for visualization
- Jupyter notebooks for training/experiments

## Getting started

### Prerequisites

- Python 3.x
- A GPU is recommended for training (the code falls back to CPU automatically).

### Install

```bash
pip install -r requirements.txt
```

### Data and weights

The dataset (`StyleClassificationIndoors/`) and the trained model weights
(`*.pth`) are **not** included in the repository because of their size. To run
training, place the dataset at the repo root with `train/` and `test/`
subfolders (one folder per class under `train/`). To run inference you also need
the corresponding `.pth` weight file referenced at the top of each script.

## Usage

### Train

Open any notebook under `notebooks/` (start with `main.ipynb`) and run the cells.
Launch Jupyter from the repo root so the relative dataset paths resolve:

```bash
jupyter notebook
```

### Inference

Each script under `scripts/` loads a trained model, predicts on a test folder,
and writes a `submission.csv`. Paths to the weights, test directory, and output
file are set as constants at the top of each script — edit them, then run:

```bash
python scripts/MobileNetV2.py   # MobileNetV2
python scripts/VGG_script.py    # VGG16
python scripts/ViT.py           # Vision Transformer
```

## Project structure

```
.
├── notebooks/        # Training experiments (one per architecture)
├── scripts/          # Standalone inference scripts -> submission.csv
├── requirements.txt
├── LICENSE
└── README.md
```

## License

Released under the [MIT License](LICENSE).
