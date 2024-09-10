# SMUG: Towards Robust MRI Reconstruction by Smoothed Unrolling (TMI 2023)

Repository with code to reproduce the results for SMUG in [our paper](https://arxiv.org/abs/2303.12735) and our upcoming journal version


Directory Structure
- `data/`: Contains datasets for MRI reconstruction tasks.
- `models/`: Model architectures for SMUG and related experiments.
- `options/`: Configuration files for different experiments.
- `util/`: Utility scripts for data handling and model evaluation.

## Overview
This repository provides code to reproduce the results from the **SMUG** method for robust MRI reconstruction, as described in our paper. SMUG systematically integrates Regularization by Smoothing (RS) with MoDL using a deep unrolled architecture. It addresses the instabilities of MoDL by optimizing where to apply RS in the unrolled architecture and introduces a novel unrolling loss to enhance training efficiency.

## Features
- **Robust MRI Reconstruction**: Enhances MoDL performance through systematic integration of RS.
- **Deep Unrolled Architecture**: Utilizes unrolling techniques for improved efficiency.
- **Instability Mitigation**: Significantly reduces three major types of instabilities in MoDL.

## Setup
1. Clone the repository:
```bash
git clone https://github.com/sjames40/SMUG_journal.git
```
2. Install the required dependencies:
```bash 
conda create --name SMUG
conda activate SMUG
pip install -r requirements.txt
```

3. Download the dataset from Dropbox: Data avaliable on https://www.dropbox.com/scl/fi/801dxovhbkp2bkl2krz5x/NEW_KSPACE.zip?rlkey=4u3b32f6c4pfujsv3kp7z5bdk&st=hwe9thrv&dl=0 

4. Refer to the `train_SMUG.py` script for training the SMUG model.
Additional scripts such as test.py and train_RSE2E.py provide testing and alternative training routines.

