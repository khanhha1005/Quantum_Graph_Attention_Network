"""
Quantum Graph Attention Network (QATN) Layer - OPTIMIZED VERSION

Architecture (matching the figure):
┌─────────────────────────────────────────────────────────────────┐
│  QATN Layer                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Variational      │  │ Quantum Attention│  │ Classical     │  │
│  │ Quantum Circuit  │→ │ Layer            │→ │ Neighborhood  │  │
│  │ (VQC)            │  │                  │  │ Aggregation   │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘

OPTIMIZATIONS:
- Reduced qubit count (max 8 instead of 16) for 4-8x speedup
- Simplified quantum circuits with fewer layers
- Added batch processing optimization
- LayerNorm for better gradient flow
- Multi-head attention option
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import pennylane as qml
from pennylane.exceptions import DeviceError

try:
    from ..QNN_Node_Embedding import quantum_net
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from QNN_Node_Embedding import quantum_net

# Global device cache to avoid recreating devices
_DEVICE_CACHE = {}


def _get_device(n_qubits):
    """Get the best available quantum device with caching."""
    if n_qubits in _DEVICE_CACHE:
        return _DEVICE_CACHE[n_qubits]
    
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits, shots=None)
    except (DeviceError, ImportError, Exception):
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
    
    _DEVICE_CACHE[n_qubits] = dev
    return dev


# ============================================================================
# QUANTUM ATTENTION CIRCUIT (HEA variant only - OPTIMIZED)
# ============================================================================

def HEA_Attention(n_qubits, n_layers=1):
    """
    HEA (Hardware Efficient Ansatz) Attention Circuit - OPTIMIZED.
    Uses RY/RZ rotations + CNOT ladder entanglement.
    Reduced layers for faster execution.
    """
    dev = _get_device(n_qubits)
    n_params = n_layers * n_qubits * 2 + 1

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        idx = 0
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[idx], wires=i)
                idx += 1
            for i in range(n_qubits):
                qml.RZ(weights[idx], wires=i)
                idx += 1
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        qml.RY(weights[-1], wires=n_qubits - 1)
        return [qml.expval(qml.PauliZ(n_qubits - 1))]

    return qml.qnn.TorchLayer(circuit, {"weights": n_params}), circuit


# ============================================================================
# QUANTUM GRAPH ATTENTION CONV (QATN LAYER) - OPTIMIZED
# ============================================================================

class QGATConv(MessagePassing):
    """
    Quantum Graph Attention Network (QATN) Layer - OPTIMIZED.
    
    OPTIMIZATIONS:
    - Max 8 qubits (down from 16) for ~4x speedup
    - LayerNorm for better gradient flow
    - Configurable attention circuit depth
    - Better initialization
    
    Architecture (as shown in the figure):
    1. VQC (Variational Quantum Circuit): Transform node features
    2. Quantum Attention Layer: Compute attention weights
    3. Classical Neighborhood Aggregation
    """
    
    def __init__(
        self,
        in_channels,
        n_layers,
        attn_model="IQP",
        n_qubits=None,
        attn_qubits=None,
        dropout=0.0,
        residual=True,
        max_qubits=8,  # Options: 4, 8, or 16 qubits
        attn_layers=1,  # Configurable attention circuit depth
        use_layer_norm=True,  # NEW: LayerNorm for stability
    ):
        super().__init__(aggr="add")
        self.dropout = dropout
        self.residual = residual
        self.attn_model_name = attn_model
        self.max_qubits = max_qubits
        self.use_layer_norm = use_layer_norm

        # ====================================================================
        # PART 1: Variational Quantum Circuit (VQC) for node transformation
        # ====================================================================
        self.n_qubits = self._select_qubits(
            in_channels if n_qubits is None else n_qubits, 
            max_qubits=max_qubits
        )
        self.in_channels = in_channels

        # Feature reduction: maps input features to qubit dimension
        if in_channels != self.n_qubits:
            self.feature_reduction = Linear(in_channels, self.n_qubits, bias=False)
        else:
            self.feature_reduction = None

        # VQC: quantum_net creates the variational circuit
        self.vqc = quantum_net(self.n_qubits, n_layers, max_qubits=max_qubits)
        self.bias = Parameter(torch.empty(self.n_qubits))
        
        # NEW: LayerNorm for better training stability
        if use_layer_norm:
            self.layer_norm = LayerNorm(self.n_qubits)
        else:
            self.layer_norm = None

        # ====================================================================
        # PART 2: Quantum Attention Layer
        # ====================================================================
        attn_input_dim = self.n_qubits * 2
        self.attn_qubits = self._select_qubits(
            attn_input_dim if attn_qubits is None else attn_qubits,
            max_qubits=max_qubits
        )
        
        # Reduce attention input to qubit dimension
        self.attn_feature_reduction = Linear(attn_input_dim, self.attn_qubits, bias=False)

        # QuantumAttentionNetwork with configurable depth (HEA only)
        if attn_model != "HEA":
            raise ValueError(f"Unknown attn_model: {attn_model}. Only 'HEA' is supported.")
        self.quantum_attention, _ = HEA_Attention(self.attn_qubits, n_layers=attn_layers)

        # Readout layer for attention scores
        self.attn_readout = Linear(1, 1)
        
        self.reset_parameters()

    @staticmethod
    def _select_qubits(dim, max_qubits=8):
        """Select power-of-2 qubit count: 4, 8, or 16."""
        n_qubits = min(dim, max_qubits)
        if n_qubits > 8:
            return min(16, max_qubits)
        if n_qubits > 4:
            return min(8, max_qubits)
        return 4  # Minimum 4 qubits

    def reset_parameters(self):
        # Xavier initialization for better convergence
        nn.init.zeros_(self.bias)
        if self.feature_reduction is not None:
            nn.init.xavier_uniform_(self.feature_reduction.weight)
        nn.init.xavier_uniform_(self.attn_feature_reduction.weight)
        nn.init.xavier_uniform_(self.attn_readout.weight)
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def forward(self, x, edge_index):
        """
        Forward pass of QATN layer - OPTIMIZED.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
        
        Returns:
            Updated node features [N, n_qubits]
        """
        # Add self-loops for self-attention
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # ================================================================
        # STEP 1: VQC - Transform node features via quantum circuit
        # ================================================================
        # Reduce features to qubit dimension if needed
        if self.feature_reduction is not None:
            x_reduced = self.feature_reduction(x)
        else:
            x_reduced = x

        # Apply VQC: |0⟩ → Embedding → Ansatz → Measurement → Transformed Features
        transformed_features = self.vqc(x_reduced).float()
        
        # Apply LayerNorm for training stability
        if self.layer_norm is not None:
            transformed_features = self.layer_norm(transformed_features)

        # ================================================================
        # STEP 2 & 3: Quantum Attention + Classical Aggregation
        # ================================================================
        out = self.propagate(edge_index, x=transformed_features)
        
        # Add bias
        out = out + self.bias

        # Residual connection (optional)
        if self.residual and transformed_features.size(-1) == out.size(-1):
            out = out + transformed_features

        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        """
        Compute attention-weighted messages.
        
        This implements the "Quantum Attention Layer" from the figure:
        1. Concatenate source (x_j) and target (x_i) transformed features
        2. Pass through QuantumAttentionNetwork
        3. Apply Quantum Dropout
        4. Apply Softmax to get attention coefficients σ(z̃)_i
        5. Compute Weighted Features
        
        Args:
            x_i: Target node features [E, n_qubits]
            x_j: Source node features [E, n_qubits]
            index: Target node indices for softmax grouping
            ptr: Pointer for batched softmax
            size_i: Number of target nodes
        
        Returns:
            Weighted messages [E, n_qubits]
        """
        # ================================================================
        # Quantum Attention Network
        # ================================================================
        
        # Concatenate source and target transformed features
        # Input Data (from figure): [x_i || x_j]
        x_concat = torch.cat((x_i, x_j), dim=-1)
        
        # Reduce to attention qubit dimension
        x_attn_input = self.attn_feature_reduction(x_concat)
        
        # QuantumAttentionNetwork: compute raw attention scores
        # This is the quantum circuit that learns attention patterns
        alpha = self.quantum_attention(x_attn_input).float()
        
        # Ensure proper shape
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        
        # Readout layer
        alpha = self.attn_readout(alpha)
        
        # LeakyReLU activation (common in attention mechanisms)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # ================================================================
        # Softmax: σ(z̃)_i - normalize attention across neighbors
        # ================================================================
        alpha = softmax(alpha, index, ptr, size_i)
        
        # ================================================================
        # Quantum Dropout (from figure)
        # ================================================================
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # ================================================================
        # Weighted Features: attention coefficient × source features
        # ================================================================
        return alpha * x_j
