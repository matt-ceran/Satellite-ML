# Satellite ML: EuroSAT RGB Land-Use Classifier

This is a small educational PyTorch project that trains a CNN from scratch on the
EuroSAT RGB dataset. The model predicts one of 10 land-use classes from a 64x64
satellite image patch.

The first milestone intentionally avoids pretrained models. The goal is to see
the whole learning loop: dataset -> model -> training -> evaluation -> inference.

## Project Layout

```text
satellite-landuse/
  data/        local datasets, ignored by git
  models/      saved checkpoints, ignored by git
  reports/     generated CSV reports, ignored by git
  src/         training and inference code
```

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download EuroSAT RGB

Use the RGB-only dataset, not the 13-band multispectral version.

Official dataset page:
https://zenodo.org/records/7711810

Command-line download:

```bash
mkdir -p data/raw
curl -L -o data/raw/EuroSAT_RGB.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
unzip -q data/raw/EuroSAT_RGB.zip -d data/raw
```

The zip may extract into a folder named `2750`. Rename it so the training command
matches the examples:

```bash
mv data/raw/2750 data/raw/EuroSAT_RGB
```

Expected layout:

```text
data/raw/EuroSAT_RGB/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

## Smoke Test Without Real Data

If the download is blocked or you just want to test the code path, run:

```bash
python -m src.train --smoke-test --epochs 1 --batch-size 16
python -m src.evaluate --smoke-test --checkpoint models/eurosat_cnn.pt
```

This uses fake images and labels. It only checks that the model, dataloaders,
training loop, checkpoint save, CSV writing, and evaluation reporting work.

## Train

```bash
python -m src.train --data-dir data/raw/EuroSAT_RGB --epochs 20 --batch-size 64 --lr 0.001
```

What happens:

1. `ImageFolder` reads one folder per class.
2. The code creates a deterministic train/validation split.
3. Training images get random flips and rotations.
4. A small CNN learns from scratch.
5. The best validation checkpoint is saved to `models/eurosat_cnn.pt`.
6. Epoch metrics are saved to `reports/training_history.csv`.

## Evaluate

```bash
python -m src.evaluate --data-dir data/raw/EuroSAT_RGB --checkpoint models/eurosat_cnn.pt
```

Evaluation prints validation accuracy, per-class accuracy, and a confusion
matrix. The confusion matrix uses rows as true classes and columns as predicted
classes.

## Predict One Image

```bash
python -m src.predict --image path/to/image.jpg --checkpoint models/eurosat_cnn.pt
```

For example:

```bash
python -m src.predict --image data/raw/EuroSAT_RGB/Forest/Forest_1.jpg --checkpoint models/eurosat_cnn.pt
```

## Learning Notes

The CNN starts with random weights. During training, each batch produces class
scores. Cross-entropy loss measures how wrong those scores are, backpropagation
computes gradients, and Adam updates the weights. Validation data is kept
separate so you can see whether the model is learning patterns that generalize
beyond the training images.

Generated data, checkpoints, reports, and local Codex memory are ignored by git.
