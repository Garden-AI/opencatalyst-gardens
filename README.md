# OpenCatalyst Garden

A collection of state-of-the-art machine learning models by the FAIR-Chem team, trained on the Open Catalyst Project's datasets, published on [Garden](https://thegardens.ai) using Modal.

## Overview

The Open Catalyst Project (OC20) is a large-scale dataset of DFT calculations for catalyst surface reactions. This repository provides Modal-based deployments of state-of-the-art models trained on this data to accelerate catalyst discovery.

The models are published on Garden, making them easily accessible and reproducible. This repo defines the Modal App that can uploaded to Gardena and published.

### Available Models

- EquiformerV2
- GemNet-OC
- eSCN
- PaiNN
- SchNet
- DimeNet++
- SCN

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Garden-AI/opencatalyst-garden.git
cd opencatalyst-garden
```

2. Install dependencies using `uv`:
```bash
uv pip install .
```

3. Run the modal app:
```bash
uv run modal run src/OC20_modal_app.py
```

### Available Tasks

1. **Structure to Energy and Forces (S2EF)**
   - Predicts atomic forces and total energy for a given structure
   - Useful for understanding local atomic interactions

2. **Initial Structure to Relaxed Energy (IS2RE)**
   - Predicts the relaxed energy of a structure without performing full relaxation
   - Efficient for screening many candidate structures

### Model Architectures

- **EquiformerV2**: Transformer-based model with E(3)-equivariant layers
- **GemNet-OC**: Graph neural network optimized for OC20
- **eSCN/SCN**: Spherical Channel Networks
- **PaiNN**: Polarizable interaction neural network
- **SchNet**: Continuous-filter convolutional neural network
- **DimeNet++**: Directional message passing neural network

## Demo Notebook

Check out `FairChem_OCP_Garden_Demo.ipynb` for a detailed example of how to use these models for catalyst structure prediction and analysis after being publised to Garden.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The [Open Catalyst Project](https://opencatalystproject.org) team for the dataset and baseline models
- The [FAIR-Chem](https://fair-chem.github.io/) project team for model development and training