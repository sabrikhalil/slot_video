# CONDITIONAL OBJECT-CENTRIC LEARNING FROM VIDEO

This repository implements the paper : "Conditional Object-Centric learning from video"

## Overview

This implementation extends Slot Attention with a predictor–corrector loop to capture temporal dynamics in video. It consists of:
- **Encoder:** CNN with positional embeddings and MLP for visual feature extraction.
- **Slot Attention (Corrector):** Refines slot representations using current frame features and previous slot states.
- **Slot Predictor:** Transformer-based module to propagate slot states over time.
- **Decoder:** Spatial broadcast decoder to reconstruct frames from slot representations.
- **Autoencoder:** Integrates all components for conditional video processing.

## Folder Structure

- **config/**: Configuration files.
- **data/**: Instructions or scripts for loading the CLEVRER video dataset.
- **models/**: Model implementations (encoder, slot attention, decoder, slot predictor, autoencoder, conditional module).
- **utils/**: Utility functions (e.g., positional encoding, visualization helpers).
- **experiments/**: Notebooks or scripts for evaluation and visualization.
- **train.py**: Training script.
- **evaluate.py**: Evaluation script.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/sabrikhalil/slot_video.git
cd slot_video
pip install -r requirements.txt
```

## Prepare Dataset 

Download dataset : 

```bash 
python data/download_dataset.py
```
Otherwise, download or place the CLEVRER video dataset into the data/clevrer/videos directory following the provided structure.

## Run Training 

```
python train.py 
```


