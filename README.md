# Spatio-Temporal Diffusion Model (STDM)

This repository contains (NON-OFFICIAL) an implementation of the Spatio-Temporal Diffusion Model (STDM), from the paper 
[A Spatio-Temporal Diffusion Model for Missing and Real-Time Financial Data Inference [Fang, Liu, Huang (2024)]](https://dl.acm.org/doi/10.1145/3627673.3679806).

## Disclaimer
This is a NON-OFFICIAL implementation of the STDM model. The original implementation has not been released by the authors as of the time of writing this README.
The implementation is based on the instructions provided in the paper, with strong assumptions where the paper is not clear.

Points to consider:
- The K=129 feature selection has been taken from OSAP's, Firm Level Characteristics/Full Sets/signed_predictors_dl_wide.zip
- The choice 129 out of the 211 available features was made based on column % of missing values. May not fit the original paper's choice.
- Preprocessing/splitting the data does not result in the same train/test split as the paper.
- Batch size (number of periods), learning rate, and other hyperparameters are not specified in the paper.
- The Firm-GCN has not been implemented yet, as the source data to construct the graph is unclear.
- The implementation is most probably poorly done, the performance are terrible.
- Terrible memory usage.

## Hardware requirements
To run the code (training + evaluation) on the full signed_predictors_dl_wide.csv dataset.
- 64GB of RAM or more (our experienced usage was around 50GB)
- 24GB of VRAM (our experienced usage was around 17GB on 1 RTX 3090)

If running on CPU (not recommend nor tested):
- Consider 128GB of RAM or more


## Features
- Data preprocessing
- STDM implementation
- STDM training
- DDIM denoising
- Evaluation metrics
- Mean imputation for comparison

## Running the code

### Setup environment
```bash
conda env create -f environment.yml
conda activate STDM-env
```

### Quick start

#### Download the data

[Download the data](https://drive.google.com/file/d/1T-nogu88A4hcFXijjftSO41K5P4Hj27y/view?usp=drive_link) and extract it to the root of the repository.

```bash
cp signed_predictors_dl_wide.zip .
unzip signed_predictors_dl_wide.zip
head -n 10000 signed_predictors_dl_wide.csv > signed_predictors_dl_wide_less.csv
```

#### Run the code

Training a model from scratch:

```bash
python main.py \
--data-file-path "signed_predictors_dl_wide.csv" \
--not-enough-months-L 60 \
--outlier-n-std 5 \
--train-start-month 197001 \
--train-end-month 199512 \
--val-start-month 199512 \
--val-end-month 200512 \
--test-start-month 200512 \
--test-end-month 201912 \
--train \
--epochs 400 \
--train-batch-size 3 \
--val-batch-size 3 \
--test-batch-size 2 \
--save-model-dir "saved_models" \
--save-datasets
```

Evaluating a model by loading a pre-trained model:
```bash
python main.py \
--load-model-path "saved_models/stdm_epoch_152_val_loss_0.1172.pt" \
--no-preprocessing \
--test-batch-size 3
```

### Development

#### Testing

To run the unit tests:

```bash
pytest
```