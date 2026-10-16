# 3D Cardiac MRI Segmentation using PyTorch & MONAI

A deep learning project for automatic cardiac MRI segmentation using a 3D U-Net architecture with PyTorch and MONAI.

## Overview

This project implements a complete medical image segmentation pipeline, including:

- NIfTI (`.nii.gz`) medical image loading
- MONAI-based preprocessing
- 3D U-Net model training
- Dice coefficient evaluation
- Sliding window inference
- Segmentation visualization

The goal is to predict voxel-level segmentation masks from cardiac MRI volumes.

---

## Dataset

**Medical Segmentation Decathlon - Task02 Heart**

The dataset contains cardiac MRI volumes and corresponding segmentation labels.

Dataset format:
.nii.gz

The dataset is not included in this repository due to size limitations.

---

## Model

Architecture:

**3D U-Net**

Features:

- 3D convolution layers
- Encoder-decoder architecture
- Skip connections
- Multi-scale feature extraction

Training configuration:

Optimizer: AdamW
Loss: DiceCELoss
Epochs: 100
Mixed Precision: AMP (FP16)
Inference: Sliding Window


---

## Results

**Author-reported best validation Dice: 0.8821.** This figure is stated in the project README; it has not been independently reproduced in this documentation. The README does not establish the exact patient-level split, seed, evaluation script/commit, held-out cohort, or external-test performance. Do not treat the number as a validated comparative benchmark.


Example prediction:

![Segmentation Result](results/figures/la_024_result.png)

---

## Installation

Clone the repository:

git clone https://github.com/aradmanamnaoon/medical-image-segmentation.git

cd medical-image-segmentation

## Install dependencies:

pip install -r requirements.txt


---

## Evidence boundaries and responsible use

This repository documents an exploratory segmentation pipeline for the Medical Segmentation Decathlon Task02 Heart dataset. It is **research software, not a clinically validated or regulated medical device**. No patient-care, diagnosis, or deployment claim is made.

Before using the reported Dice value in a publication, grant, academic application, or benchmark comparison, preserve and independently check the exact evaluation script, dataset version, patient-level split definition, checkpoint identifier, inference settings, random seed, full per-case results, and environment. If any of these artifacts are unavailable, describe the value only as an *unreplicated, author-reported validation result*. The absence of test-set and external-site validation limits claims of generalizability.

**Data and privacy:** The dataset is not distributed in this repository. Obtain it from its original authorized source and comply with its terms; do not upload identifiable medical images or derived patient records to public issues, logs, or repositories.

**Contribution provenance:** GitHub hosting demonstrates the existence of this project, but does not independently verify individual authorship contributions, clinical performance, or third-party adoption. AI-assisted contributions and third-party code/data should be disclosed in a future citable release where applicable.
