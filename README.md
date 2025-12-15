# A Novel Approach to GNN Explainability: Distilling Knowledge with Inter-Layer Alignment

## Overview

This repository contains the official implementation of the paper **"A Novel Approach to GNN Explainability: Distilling Knowledge with Inter-Layer Alignment"**.

We propose a novel Graph Neural Network (GNN) explainability method that transfers knowledge from complex GNN teacher models to simple linear student models through knowledge distillation with inter-layer alignment, enabling efficient and interpretable graph node classification.

## Abstract

Graph Neural Networks (GNNs) have made significant strides in the analysis and modeling of complex networkdata, particularly excelling in graph and node classification tasks.However, the ”black-box” nature of GNNs impedes user understanding and trust, thereby restricting their broader application.This challenge has spurred a growing focus on demystifyingGNNs to make their decision-making processes more transparent.Traditional methods for explaining GNNs often rely on selectingsubgraphs and employing combinatorial optimization to generateunderstandable outputs. However, these methods are closelylinked to the inherent complexity of GNNs, leading to higher explanation costs. To address this issue, we introduce a lower complexity proxy model to explain GNNs. Our approach leverages knowledge distillation with inter-layer alignment, specifically targeting the challenge of over-smoothing and its detrimental impact on model explanation. Initially, we distill critical insights from complex GNN models into a more manageable proxy model. We then apply an inter-layer alignment-based distillation technique to ensure alignment between the proxy and the original model, facilitating the extraction of node or edge-level explanations within the proxy framework. We theoretically prove that the explanations derived from the proxy model are faithfulto both the proxy and the original model. Additionally, we showthat the upper bound of unfaithfulness between the proxy andthe original model remains consistent when the distillation erroris infinitesimal. This inter-layer alignment knowledge distillationtechnique enables the proxy model to retain the knowledgelearning and topological representation capabilities of the original model to the greatest extent. Experimental evaluations onnumerous real-world datasets confirm the effectiveness of ourmethod, demonstrating robust performance.

### Key Contributions

- **Inter-Layer Alignment Distillation**: Aligning feature representations between teacher and student models at each layer for more effective knowledge transfer
- **Linear Student Model**: Employing an SGC-based linear model as the student, which is inherently interpretable
- **Efficient Explanation Generation**: Leveraging the linear properties of the student model to rapidly generate node explanations through feature perturbation

## Environment Setup

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- PyTorch Geometric >= 2.0
- NumPy
- scikit-learn
- tqdm

### Installation

```bash
# Clone the repository
git clone https://github.com/ZXXSTUDENTGITHUB/DILA.git
cd DILA

# Create a virtual environment (recommended)
conda create -n gnn-explain python=3.8
conda activate gnn-explain

# Install PyTorch (choose according to your CUDA version)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install numpy scikit-learn tqdm pandas
```

## Quick Start

### Run Complete Experiments

```bash
python train.py
```

This command will run the complete experiment pipeline on all synthetic datasets (syn1-syn6), including:
1. Training the teacher model (GCN)
2. Knowledge distillation to train the student model
3. Generating node explanations
4. Evaluating explanation quality

### Custom Configuration

You can modify the following parameters in the `main()` function of `train.py`:

```python
# Experiment configuration
datasets = ["syn1", "syn2", "syn3", "syn4", "syn5", "syn6"]  # Dataset list
num_epochs = 200          # Number of training epochs
hidden_dim = 64           # Hidden layer dimension
num_layers = 3            # Number of GNN layers
alpha = 0.5               # Distillation loss weight (KL divergence vs Cross-entropy)
temperature = 3.0         # Distillation temperature
iterations = 3            # Number of experiment repetitions
```

## Datasets

### Synthetic Datasets

| Dataset | Description | #Nodes | Ground Truth |
|---------|-------------|--------|--------------|
| BA-Shapes | BA graph + House motif | 700 | 5-node house |
| BA-Community | Dual BA community graph | 1400 | 5-node house |
| Tree-Cycles | 8-level tree + Cycle motif | 1020 | 9-node cycle |
| Tree-Grid | 8-level tree + Grid motif | 871 | 6-node grid |
| BA-Bottle | BA graph + Bottle motif | 1231 | 9-node bottle |
| BA-2Motifs | BA graph + Dual House motif | 700 | 5-node house |

### Real-World Datasets

| Dataset | Description |
|---------|-------------|
| Bitcoin-Alpha | Bitcoin trust network |
| Bitcoin-OTC | Bitcoin OTC trading network |

### Data Format

Each dataset contains three `.npy` files:
- `*_A.npy`: Adjacency matrix
- `*_X.npy`: Node feature matrix
- `*_L.npy`: Node labels

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{zhang2024novel,
  title={A Novel Approach to GNN Explainability: Distilling Knowledge with Inter-Layer Alignment},
  author={Zhang, Xiaoxia and Liu, Xingyu and Wang, Guoyin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

## Authors

- **Xiaoxia Zhang**
- **Xingyu Liu**
- **Guoyin Wang**

## Acknowledgements

We thank all the researchers and developers who contributed to this project.
