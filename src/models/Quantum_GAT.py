import torch
from torch.nn import Module, ModuleList, Linear, LeakyReLU, Dropout, LayerNorm
from torch_geometric.nn import global_mean_pool

try:
    from .QATN_Layers import QGATConv
except ImportError:
    from QATN_Layers import QGATConv


class QGAT(Module):
    """
    Quantum Graph Attention Network (QATN) - OPTIMIZED VERSION
    
    Quantum attention mechanism:
    - HEA: Hardware Efficient Ansatz (RY/RZ + CNOT ladder)
    
    OPTIMIZATIONS:
    - Configurable max_qubits (default 8 for speed)
    - LayerNorm for training stability
    - Better initialization
    
    Architecture:
    Input → [QATN Layer x L] → Global Mean Pool → Linear Classifier → Output
    """
    def __init__(
        self,
        input_dims,
        q_depths,
        output_dims,
        attn_model="HEA",
        activ_fn=LeakyReLU(0.2),
        dropout=0.2,
        readout=False,
        max_qubits=8,  # Configurable: 4, 8, or 16 qubits
        use_layer_norm=True,  # NEW: LayerNorm for stability
    ):
        super().__init__()
        layers = []
        current_dim = input_dims

        for q_depth in q_depths:
            layer = QGATConv(
                in_channels=current_dim,
                n_layers=q_depth,
                attn_model=attn_model,
                dropout=dropout,
                residual=True,
                max_qubits=max_qubits,  # Pass max_qubits
                use_layer_norm=use_layer_norm,
            )
            layers.append(layer)
            current_dim = layer.n_qubits

        self.layers = ModuleList(layers)
        self.embedding_dim = current_dim
        self.activ_fn = activ_fn
        self.dropout = Dropout(p=dropout)
        
        # Output LayerNorm for stability
        if use_layer_norm:
            self.output_norm = LayerNorm(self.embedding_dim)
        else:
            self.output_norm = None

        if readout:
            self.readout = Linear(1, 1)
        else:
            self.readout = None

        # Linear classifier
        self.classifier = Linear(self.embedding_dim, output_dims)

    def forward(self, x, edge_index, batch):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.activ_fn(h)
            h = self.dropout(h)

        h = global_mean_pool(h, batch)
        
        # Apply output normalization
        if self.output_norm is not None:
            h = self.output_norm(h)
        
        h = self.classifier(h)

        if self.readout is not None:
            h = self.readout(h)

        return h
