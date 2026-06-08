# SMUG: Towards Robust MRI Reconstruction by Smoothed Unrolling (TMI 2023)

Repository with code to reproduce the results for **SMUG** in our paper:

**SMUG: Towards Robust MRI Reconstruction by Smoothed Unrolling**
Paper link: https://arxiv.org/abs/2303.12735

This repository contains code for reproducing the results of SMUG and the upcoming journal version.

---

## Overview

This repository provides code to reproduce the results from the **SMUG** method for robust MRI reconstruction. SMUG systematically integrates **Regularization by Smoothing (RS)** with **MoDL** using a deep unrolled architecture.

The method addresses the instabilities of MoDL by optimizing where to apply randomized smoothing in the unrolled architecture and introduces an unrolling loss to improve training efficiency.

---

## Features

* **Robust MRI Reconstruction**
  Enhances MoDL robustness through systematic integration of randomized smoothing.

* **Deep Unrolled Architecture**
  Uses unrolling techniques for efficient MRI reconstruction.

* **Instability Mitigation**
  Reduces major instability issues in MoDL under perturbations and adversarial attacks.

* **Weighted SMUG**
  Extends SMUG by using a trainable weighting encoder to perform weighted randomized smoothing.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sjames40/SMUG_journal.git
cd SMUG_journal-main
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset

This project uses the fastMRI dataset.

Dataset website:
https://fastmri.med.nyu.edu/

Please download and organize the dataset before training. The download may take some time.

In the current experiments, the k-space data and mask data are stored as:

```bash
/SMUG/SMUG_journal-main/data/NEW_KSPACE
/SMUG/SMUG_journal-main/data/MASK_4X
```

---

## Directory Structure

```text
data/
    Contains datasets and sampling masks for MRI reconstruction.

models/
    Contains model architectures for SMUG, MoDL, Weighted SMUG, and related experiments.

options/
    Contains configuration files and command-line options for different experiments.

util/
    Contains utility scripts for data loading, evaluation, metrics, and other helper functions.

checkpoints/
    Stores trained model checkpoints.
```

---

# Reproduction Pipeline

The full training and evaluation pipeline consists of the following steps:

1. Prepare dataset and masks.
2. Pretrain the denoiser.
3. Finetune SMUG.
4. Finetune Weighted SMUG.
5. Evaluate.

---

## Step 1: Prepare Dataset and Masks

Make sure the data folders exist:

```bash
/SMUG_journal-main/data/NEW_KSPACE
/SMUG_journal-main/data/MASK_4X
```

The k-space data should be stored in:

```bash
/SMUG_journal-main/data/NEW_KSPACE
```

The 4x undersampling masks should be stored in:

```bash
/SMUG_journal-main/data/MASK_4X
```

---

## Step 2: Pretrain the Denoiser

Before training SMUG or Weighted SMUG, first pretrain the denoiser.

```bash
cd /SMUG_journal-main

python pretrain_denoiser.py \
  --dataroot /SMUG_journal-main/data/NEW_KSPACE \
  --mask_dataroot /SMUG_journal-main/data/MASK_4X \
  --gpu_ids 0 \
  --trainSize 3000 \
  --valiSize 32 \
  --batchSize 2 \
  --epoch 60 \
  --lr 1e-4 \
  --smoothing_epsilon 0.01 \
  --n_res_blocks 3 \
  --checkpoints_dir /SMUG_journal-main/checkpoints \
  --name pretrain_denoiser
```

### Output Checkpoint

```bash
/SMUG_journal-main/checkpoints/pretrain_denoiser/vali_best.pth
```

This checkpoint will be used to initialize the denoiser in SMUG and Weighted SMUG fine-tuning.

---

## Step 3: Finetune SMUG

Finetune SMUG using the pretrained denoiser from Step 2.

```bash

python train_SMUG.py \
  --dataroot /SMUG_journal-main/data/NEW_KSPACE \
  --mask_dataroot /SMUG_journal-main/data/MASK_4X \
  --netGpath /SMUG_journal-main/checkpoints/pretrain_denoiser/vali_best.pth \
  --gpu_ids 0 \
  --trainSize 3000 \
  --valiSize 32 \
  --batchSize 2 \
  --epoch 60 \
  --lr 1e-4 \
  --blockIter 8 \
  --num_sample 10 \
  --smoothing_epsilon 0.01 \
  --n_res_blocks 3 \
  --LossLambda 1.0 \
  --checkpoints_dir /SMUG_journal-main/checkpoints \
  --name smug_modl
```

### Output Checkpoint

```bash
/SMUG_journal-main/checkpoints/smug_modl/vali_best.pth
```

---

## Step 4: Finetune Weighted SMUG

Weighted SMUG extends ordinary SMUG by introducing a trainable weighting encoder.

```bash

python train_Weighted_SMUG.py \
  --dataroot /SMUG_journal-main/data/NEW_KSPACE \
  --mask_dataroot /SMUG_journal-main/data/MASK_4X \
  --netGpath /SMUG_journal-main/checkpoints/pretrain_denoiser/vali_best.pth \
  --gpu_ids 0 \
  --trainSize 3000 \
  --valiSize 32 \
  --batchSize 2 \
  --epoch 60 \
  --lr 1e-4 \
  --blockIter 8 \
  --num_sample 10 \
  --smoothing_epsilon 0.01 \
  --n_res_blocks 3 \
  --LossLambda 1.0 \
  --checkpoints_dir /SMUG_journal-main/checkpoints \
  --name weighted_smug_modl
```

### Important Note

Weighted SMUG requires a separate fine-tuning run.

The ordinary SMUG checkpoint does **not** contain the trainable weighting encoder `E_phi`. Therefore, the ordinary SMUG checkpoint cannot be used as a complete Weighted SMUG model without training the new weighting encoder.

---

## Step 5: Evaluate Clean, Random-Noise, and PGD Robustness

Use `test.py` to evaluate clean reconstruction performance, random-noise robustness, and PGD robustness.

### Evaluate Ordinary SMUG

```bash

python test.py \
  --dataroot /SMUG_journal-main/data/NEW_KSPACE \
  --mask_dataroot /SMUG_journal-main/data/MASK_4X \
  --netGpath /SMUG_journal-main/checkpoints/smug_modl/vali_best.pth \
  --gpu_ids 0 \
  --smoothing SMUG \
  --num_sample 10 \
  --smoothing_epsilon 0.01 \
  --n_res_blocks 3
```

For ordinary SMUG, use:

```bash
--netGpath /SMUG_journal-main/checkpoints/smug_modl/vali_best.pth
--smoothing SMUG
```

---

### Evaluate Weighted SMUG

For Weighted SMUG, use the combined checkpoint produced by `train_Weighted_SMUG.py`.

```bash
--netGpath /SMUG_journal-main/checkpoints/weighted_smug_modl/vali_best.pth
--smoothing WeightedSMUG
```

---

# Citation

If you use this code, please cite the SMUG paper:

```bibtex
@article{smug2023,
  title={SMUG: Towards Robust MRI Reconstruction by Smoothed Unrolling},
  author={},
  journal={arXiv preprint arXiv:2303.12735},
  year={2023}
}
```

Please update the BibTeX entry with the official TMI citation when available.
