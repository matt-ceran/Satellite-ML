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

## Train The Improved From-Scratch CNN

```bash
python -m src.train --model improved --data-dir data/raw/EuroSAT_RGB --epochs 20 --batch-size 64 --lr 0.001 --weight-decay 0.0001
```

This trains a stronger CNN from scratch and saves:

```text
models/eurosat_improved_cnn.pt
reports/improved_training_history.csv
```

The improved model uses BatchNorm, more convolution filters, an extra
convolution block, dropout, and global average pooling. It should be compared
against the small CNN baseline instead of overwriting it.

## Evaluate

```bash
python -m src.evaluate --data-dir data/raw/EuroSAT_RGB --checkpoint models/eurosat_cnn.pt
```

Evaluation prints validation accuracy, per-class accuracy, and a confusion
matrix. The confusion matrix uses rows as true classes and columns as predicted
classes.

It also saves:

```text
reports/confusion_matrix.png
reports/per_class_accuracy.csv
```

## Plot Training History

```bash
python -m src.plot_history --history-csv reports/training_history.csv --output reports/learning_curves.png
```

This turns the epoch-by-epoch training history into accuracy and loss curves.

## Predict One Image

```bash
python -m src.predict --image path/to/image.jpg --checkpoint models/eurosat_cnn.pt
```

For example:

```bash
python -m src.predict --image data/raw/EuroSAT_RGB/Forest/Forest_1.jpg --checkpoint models/eurosat_cnn.pt
```

## Predict A Batch Of Images

```bash
python -m src.predict_batch --data-dir data/raw/EuroSAT_RGB --checkpoint models/eurosat_cnn.pt --samples-per-class 20
```

This writes:

```text
reports/prediction_samples.csv
reports/prediction_samples_summary.txt
```

The CSV is useful for finding individual images the model misses.

## Create A Future Train/Validation/Test Split Manifest

```bash
python -m src.create_splits --data-dir data/raw/EuroSAT_RGB --val-split 0.15 --test-split 0.10 --seed 42
```

This writes a deterministic class-balanced manifest:

```text
reports/split_manifest.csv
reports/split_summary.csv
```

The current baseline checkpoint was trained with the original train/validation
split. This manifest is for the next training phase where a final test set should
be held out before training starts.

## Compare Baseline And Improved Reports

```bash
python -m src.compare_experiments
```

This compares baseline and improved histories, per-class accuracy, and sampled
prediction summaries. It writes:

```text
reports/phase_2b_comparison.csv
reports/phase_2b_comparison.md
```

## Learning Notes

The CNN starts with random weights. During training, each batch produces class
scores. Cross-entropy loss measures how wrong those scores are, backpropagation
computes gradients, and Adam updates the weights. Validation data is kept
separate so you can see whether the model is learning patterns that generalize
beyond the training images.

Generated data, checkpoints, reports, and local Codex memory are ignored by git.
