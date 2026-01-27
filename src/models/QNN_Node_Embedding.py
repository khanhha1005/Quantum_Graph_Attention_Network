import pennylane as qml
from pennylane.exceptions import DeviceError
import torch
import numpy as np

# Global device cache to avoid recreating devices
_VQC_DEVICE_CACHE = {}


def _select_qubits(dim, max_qubits=8):
    """Select power-of-2 qubit count: 4, 8, or 16."""
    n_qubits = min(dim, max_qubits)
    if n_qubits > 8:
        return min(16, max_qubits)
    if n_qubits > 4:
        return min(8, max_qubits)
    return 4  # Minimum 4 qubits


def quantum_net(n_qubits, n_layers, device_name=None, max_qubits=8):
    """
    Quantum variational circuit for node embedding - OPTIMIZED.
    
    Args:
        n_qubits: Number of qubits (will be capped by max_qubits)
        n_layers: Number of variational layers
        device_name: Quantum device name (default: "lightning.gpu" with fallback to "lightning.qubit")
        max_qubits: Maximum number of qubits (default: 8 for speed)
    
    Returns:
        A PyTorch layer (qml.qnn.TorchLayer) that can process node embeddings
    """
    # Cap qubits for speed
    actual_qubits = _select_qubits(n_qubits, max_qubits)
    
    # Use cached device or create new one
    cache_key = (actual_qubits, device_name)
    if cache_key in _VQC_DEVICE_CACHE:
        dev = _VQC_DEVICE_CACHE[cache_key]
    else:
        if device_name is None:
            try:
                dev = qml.device("lightning.gpu", wires=actual_qubits, shots=None)
                device_name = "lightning.gpu"
            except (DeviceError, ImportError, Exception):
                dev = qml.device("lightning.qubit", wires=actual_qubits, shots=None)
                device_name = "lightning.qubit"
        else:
            dev = qml.device(device_name, wires=actual_qubits, shots=None)
        _VQC_DEVICE_CACHE[cache_key] = dev
    
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def quantum_circuit(inputs, q_weights):
        """
        Quantum variational circuit for node embedding - OPTIMIZED.
        
        Args:
            inputs: Node features [batch_size, actual_qubits]
            q_weights: Trainable quantum weights [n_layers, actual_qubits, 2]
        
        Returns:
            Expectation values for each qubit [batch_size, actual_qubits]
        """
        # Embed features using angle embedding
        qml.AngleEmbedding(inputs, wires=range(actual_qubits), rotation="Y")
        
        # Apply variational layers
        for layer in range(n_layers):
            # Apply rotations to each qubit
            for qubit in range(actual_qubits):
                qml.RY(q_weights[layer, qubit, 0], wires=qubit)
                qml.RZ(q_weights[layer, qubit, 1], wires=qubit)
            
            # Entangling layer - ring topology
            for qubit in range(actual_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
            # Connect last qubit to first for ring topology
            if actual_qubits > 2:
                qml.CZ(wires=[actual_qubits - 1, 0])
        
        # Measure expectation values in Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(actual_qubits)]
    
    # Create weight shape: [n_layers, actual_qubits, 2]
    weight_shapes = {"q_weights": (n_layers, actual_qubits, 2)}
    
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

