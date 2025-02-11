# FAIR-Chem / OpenCatalyst Gardens

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

OpenCatalyst Garden is a repository of state-of-the-art machine learning models developed by the FAIR-Chem team for catalyst discovery. These models are trained on the Open Catalyst Project (OC20) dataset and published  on [Garden](https://thegardens.ai), making them easily accessible and reproducible.

## Overview

The Open Catalyst Project (OC20) provides a large-scale dataset of DFT calculations for catalyst surface reactions. This repository contains Modal-based deployments of several models, enabling rapid structure optimization and energy/force predictions, which are essential for catalysis research.

## Quick Start

Follow these simple steps to get started locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Garden-AI/opencatalyst-garden.git
   cd opencatalyst-garden
   ```

2. **Install dependencies using `uv`:**  
   (Ensure you have Python 3.10 and that `uv` is installed.)
   ```bash
   uv pip install .
   ```

3. **Run the Modal app:**
   ```bash
   uv run modal run src/OC20_modal_app.py
   ```

## Available Models

- EquiformerV2
- GemNet-OC
- eSCN
- PaiNN
- SchNet
- DimeNet++
- SCN

## Available Tasks

1. **Structure to Energy and Forces (S2EF)**
   - Predicts atomic forces and total energy for a given structure.
   - Useful for understanding local atomic interactions.

2. **Initial Structure to Relaxed Energy (IS2RE)**
   - Predicts the relaxed energy of a structure without performing full relaxation.
   - Efficient for screening many candidate structures.

## Model Architectures

- **EquiformerV2**: Transformer-based model with E(3)-equivariant layers.
- **GemNet-OC**: Graph neural network optimized for OC20.
- **eSCN/SCN**: Spherical Channel Networks.
- **PaiNN**: Polarizable interaction neural network.
- **SchNet**: Continuous-filter convolutional neural network.
- **DimeNet++**: Directional message passing neural network.


## Modal App Definition

Check out the Modal App definition in [OC20_modal_app.py](src/OC20_modal_app.py) to see how the models
are loaded and configured to run on Modal's GPUs.


## Demo Notebook

For a detailed example of how to use these models for catalyst structure prediction and analysis, check out the [FairChem_OCP_Garden_Demo.ipynb](FairChem_OCP_Garden_Demo.ipynb) notebook.

## Contributing

Contributions are welcome! If you have suggestions or spot issues, please feel free to submit a Pull Request or open an issue.

## Acknowledgments

- The [Open Catalyst Project](https://opencatalystproject.org) team for the dataset and baseline models.
- The [FAIR-Chem](https://fair-chem.github.io/) team for model development and training.
