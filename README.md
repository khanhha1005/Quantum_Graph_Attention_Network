# QGATN — Quantum Graph Attention Network for Smart Contract Vulnerability Detection

This repository implements an **optimized Quantum Graph Attention Network (QGATN)** for **smart contract vulnerability detection**.  
QGATN combines:

- **Node-wise Variational Quantum Circuit (VQC)** for nonlinear node feature transformation, and
- **Quantum Attention Circuit** to compute **edge-wise attention logits** for attention-weighted message passing,

built with **PyTorch**, **PyTorch Geometric**, and **PennyLane** .

---

## Highlights

- **Quantum-enhanced message passing**
  - Node embedding via **AngleEmbedding + (RY/RZ) + CZ ring entanglement + Pauli-Z readout**
  - Edge attention via **(RY/RZ + CNOT ladder + readout)** → scalar attention logit per edge
- **Parameter-efficient and configurable**
  - Qubits: **4 / 8 / 16**
  - Quantum depth: **1 / 3 / 5** (grid search ready)
- **Training optimizations**
  - Class weighting for imbalance (`pos_weight`)
  - OneCycleLR scheduler
  - Gradient clipping
  - TensorBoard logging
  - Early stopping
  - Optional focal loss (disabled by default)
---

## Training Dataset

The models are trained on **smart contract vulnerability detection datasets** (binary classification):

- **Reentrancy**: Detecting reentrancy vulnerabilities  
- **Integer Overflow/Underflow**: Detecting arithmetic overflow/underflow vulnerabilities  
- **Timestamp Dependency**: Detecting timestamp-dependent vulnerabilities  
- **Delegatecall**: Detecting unsafe `delegatecall` usage  

---

## Data Preparation

The training data in `train_data/` is obtained by running the **graph construction** and **graph normalization** steps from the **GNNSCVulDetector** pipeline.

### To prepare the training data

1. Clone the **GNNSCVulDetector** repository  
2. Follow their instructions to run:
   - graph construction (Solidity → execution-aware graph)
   - graph normalization (merge/redirect auxiliary and fallback semantics)
3. Put the exported JSON files into `train_data/` following this structure:

```

train_data/
├── reentrancy/
│   ├── train.json
│   └── valid.json
├── integeroverflow/
│   ├── train.json
│   └── valid.json
├── timestamp/
│   ├── train.json
│   └── valid.json
└── delegatecall/
├── train.json
└── valid.json

````

---

## Data Format

Each dataset contains:

- **Graph-structured code representations**
  - Nodes represent code elements (statements, expressions, variables, calls, etc.)
  - Edges represent relationships (control flow / data flow / call & fallback semantics)
- **Binary labels**
  - `targets = 1`: vulnerable
  - `targets = 0`: benign
- **JSON format**
  - Each JSON file contains a **list** of graphs with:
    - `node_features`: feature vectors per node
    - `graph`: edge list in format `[source, edge_type, target]`
    - `targets`: binary label (0 or 1)

Example sample:
```json
{
  "targets": 1,
  "contract_name": "example.sol",
  "graph": [[0, 9, 1], [1, 2, 3], [3, 1, 4]],
  "node_features": [
    [0, 1, 0, 0, ...],
    [1, 0, 0, 1, ...]
  ]
}
````

**Note:** The current dataloader keeps `edge_type` for compatibility, but the model uses only `(source, target)` as `edge_index`.

---

## Installation

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install dependencies

Minimal dependencies:

* `torch`
* `torch_geometric`
* `pennylane`
* `numpy`, `scikit-learn`, `tqdm`, `tensorboard`

Example:

```bash
pip install torch
pip install torch-geometric
pip install pennylane
pip install numpy scikit-learn tqdm tensorboard
```

## Quick Start (Training)

Run the grid-search training script:

```bash
python train_quantum_models.py
```

By default, it runs a grid search across:

* Vulnerability tasks: `integeroverflow`, `reentrancy`, `timestamp`, `delegatecall`
* Attention model: `default`
* Quantum depth: `[1, 3, 5]`
* Qubits: `[4, 8, 16]`

Key defaults (see `train_quantum_models.py`):

* `batch_size = 32`
* `max_epochs = 30`
* `learning_rate = 5e-4`
* `use_scheduler = True` (OneCycleLR)
* `use_class_weights = True`

---

## Outputs & Logs

All experiment outputs are saved to:

```
training_results_1/
  <vuln>_HEA_L<layers>_Q<qubits>_<timestamp>/
    ├── best_model.pt
    ├── model.pt
    ├── history.json
    ├── config.json
    └── logs/              # TensorBoard logs
  summary_<timestamp>.json
  results_<timestamp>.csv
```

### TensorBoard

```bash
tensorboard --logdir training_results_1
```
---

## Customization

Edit experiment grid in `train_quantum_models.py`:

* `layer_depths = [1, 3, 5]`
* `qubit_options = [4, 8, 16]`
* `vulnerability_types = [...]`

Other useful knobs:

* `config['max_epochs']`
* `config['batch_size']` (quantum simulation can be slow)
* `config['use_focal_loss'] = True`
* `config['grad_clip']`
