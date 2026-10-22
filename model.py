"""
QSNN Model — Hybrid Quantum-Spiking Neural Network.

Architecture:
    ConvSNN (snnTorch)  →  QuantumLayer (Qiskit)  →  ClassicalANN (PyTorch)

Components:
    1. ConvSNN:      Conv→Leaky→Pool ×2, then FC→Leaky ×3, outputs 4 neurons
    2. QuantumLayer: 4-qubit circuit with Ry encoding, mixed rotations, CNOT
    3. ClassicalANN: FC with BatchNorm + Sigmoid, outputs 10 classes
    4. QSNN:         Wrapper that orchestrates phases (bypass / quantum)
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils as snn_utils

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector


# ═════════════════════════════════════════════════════════════
# 1. Convolutional Spiking Neural Network Encoder
# ═════════════════════════════════════════════════════════════

class ConvSNN(nn.Module):
    """Convolutional SNN encoder using Leaky Integrate-and-Fire neurons.

    Architecture:
        Conv2d(3→32, 3×3)  → Leaky(β=0.9)  → MaxPool2d(2)
        Conv2d(32→64, 3×3) → Leaky(β=0.9)  → MaxPool2d(2)
        Flatten
        Linear(4096→128)   → Leaky(β=0.9)
        Linear(128→64)     → Leaky(β=0.9)
        Linear(64→4)       → Leaky(β=0.9, output=True)

    The network processes rate-encoded spike trains over multiple time steps
    and returns the mean membrane potential of the output layer, producing a
    continuous 4-dimensional feature vector per sample.

    Args:
        beta (float): Membrane potential decay rate for LIF neurons.
        num_steps (int): Number of simulation time steps.
    """

    def __init__(self, beta=0.9, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()

        # ── Convolutional block 1 ──
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              init_hidden=True)
        self.pool1 = nn.MaxPool2d(2)

        # ── Convolutional block 2 ──
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)

        # ── Fully connected SNN layers ──
        # After 2× pool on 32×32 input: 32→16→8, so 64 × 8 × 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              init_hidden=True)

        self.fc2 = nn.Linear(128, 64)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              init_hidden=True)

        self.fc3 = nn.Linear(64, 4)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              init_hidden=True, output=True)

    def forward(self, spike_data):
        """Process rate-encoded spike trains through the SNN.

        Args:
            spike_data: Tensor of shape (num_steps, B, 3, 32, 32)
                        containing binary spike trains.

        Returns:
            features: Tensor of shape (B, 4) — mean membrane potentials
                      of the output layer across all time steps.
        """
        # Reset all LIF neuron membrane potentials
        snn_utils.reset(self)

        mem_rec = []
        num_steps = spike_data.shape[0]

        for step in range(num_steps):
            x = spike_data[step]

            # Conv block 1: Conv → Leaky → Pool
            x = self.conv1(x)
            x = self.lif1(x)
            x = self.pool1(x)

            # Conv block 2: Conv → Leaky → Pool
            x = self.conv2(x)
            x = self.lif2(x)
            x = self.pool2(x)

            # Flatten spatial dims
            x = x.flatten(1)

            # FC block 1
            x = self.fc1(x)
            x = self.lif3(x)

            # FC block 2
            x = self.fc2(x)
            x = self.lif4(x)

            # FC block 3 — output layer (returns spk, mem)
            x = self.fc3(x)
            spk, mem = self.lif5(x)
            mem_rec.append(mem)

        # Average membrane potential across time steps → continuous features
        features = torch.stack(mem_rec).mean(dim=0)  # (B, 4)
        return features


# ═════════════════════════════════════════════════════════════
# 2. Quantum Layer (4-Qubit Parameterised Circuit)
# ═════════════════════════════════════════════════════════════

def _build_quantum_circuit():
    """Construct the 4-qubit parameterised quantum circuit.

    Circuit structure:
        ┌────────────────────────┐
        │  Angle Encoding (Ry)   │  ← 4 input params from ConvSNN
        │  q0: Ry(x₀)           │
        │  q1: Ry(x₁)           │
        │  q2: Ry(x₂)           │
        │  q3: Ry(x₃)           │
        ├────────────────────────┤
        │  Trainable Rotations   │  ← 4 trainable weight params
        │  q0: Rx(θ₀)           │
        │  q1: Rz(θ₁)           │
        │  q2: Ry(θ₂)           │
        │  q3: Rz(θ₃)           │
        ├────────────────────────┤
        │  Entanglement          │
        │  CNOT(q0, q1)         │
        │  CNOT(q2, q3)         │
        └────────────────────────┘

    Returns:
        qc: QuantumCircuit
        input_params: ParameterVector of input parameters (length 4)
        weight_params: ParameterVector of trainable parameters (length 4)
    """
    input_params = ParameterVector('x', 4)
    weight_params = ParameterVector('w', 4)

    qc = QuantumCircuit(4)

    # ── Angle encoding with Ry gates ──
    for i in range(4):
        qc.ry(input_params[i], i)

    # ── Trainable rotation gates (mixed types for expressivity) ──
    qc.rx(weight_params[0], 0)   # Rx on qubit 0
    qc.rz(weight_params[1], 1)   # Rz on qubit 1
    qc.ry(weight_params[2], 2)   # Ry on qubit 2
    qc.rz(weight_params[3], 3)   # Rz on qubit 3

    # ── Entanglement via CNOT gates ──
    qc.cx(0, 1)   # CNOT: control=q0, target=q1
    qc.cx(2, 3)   # CNOT: control=q2, target=q3

    return qc, input_params, weight_params


class QuantumLayer(nn.Module):
    """4-qubit quantum circuit layer wrapped as a PyTorch module.

    Uses Qiskit's SamplerQNN to simulate the circuit and TorchConnector
    to bridge it into PyTorch's autograd graph.

    Input:  (B, 4)  — classical features from ConvSNN
    Output: (B, 16) — measurement probabilities over 2⁴ basis states
    """

    def __init__(self):
        super().__init__()
        qc, input_params, weight_params = _build_quantum_circuit()

        qnn = SamplerQNN(
            circuit=qc,
            input_params=list(input_params),
            weight_params=list(weight_params),
            input_gradients=True,   # Enable backprop through inputs
        )

        self.qnn = TorchConnector(qnn)

    def forward(self, x):
        """Run the quantum circuit on a batch of 4-dim feature vectors.

        Args:
            x: Tensor of shape (B, 4) with continuous classical values.

        Returns:
            Tensor of shape (B, 16) — probabilities for each of the
            16 basis states (|0000⟩ through |1111⟩).
        """
        return self.qnn(x)


# ═════════════════════════════════════════════════════════════
# 3. Classical ANN Head
# ═════════════════════════════════════════════════════════════

class ClassicalANN(nn.Module):
    """Fully connected ANN classification head.

    Architecture:
        Linear(in→64)  → BatchNorm1d → Sigmoid → Dropout
        Linear(64→32)  → BatchNorm1d → Sigmoid → Dropout
        Linear(32→10)  (raw logits — softmax is in CrossEntropyLoss)

    Args:
        input_dim (int): Dimensionality of input features.
                         4 for bypass mode, 16 for quantum mode.
        num_classes (int): Number of output classes (10 for CIFAR-10).
        dropout (float): Dropout probability for regularisation.
    """

    def __init__(self, input_dim, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(dropout),

            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """Forward pass through the ANN head.

        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            Tensor of shape (B, num_classes) — raw logits.
        """
        return self.net(x)


# ═════════════════════════════════════════════════════════════
# 4. Full QSNN Model (Wrapper)
# ═════════════════════════════════════════════════════════════

class QSNN(nn.Module):
    """Hybrid Quantum-Spiking Neural Network.

    Combines ConvSNN → QuantumLayer → ClassicalANN.

    Two operating modes:
        - **Bypass** (Phase 1): ConvSNN(4) → ANN_bypass(4→10)
          Quantum layer is skipped; trains Conv+ANN only.
        - **Quantum** (Phase 2): ConvSNN(4) → Quantum(4→16) → ANN_quantum(16→10)
          Full hybrid pipeline with trainable quantum circuit.

    Args:
        beta (float): LIF neuron decay rate.
        num_steps (int): SNN simulation time steps.
        num_classes (int): Output classes.
        dropout (float): ANN dropout rate.
    """

    def __init__(self, beta=0.9, num_steps=25, num_classes=10, dropout=0.3):
        super().__init__()

        self.conv_snn = ConvSNN(beta=beta, num_steps=num_steps)
        self.quantum_layer = QuantumLayer()

        # Two separate ANN heads for the two phases
        self.ann_bypass = ClassicalANN(input_dim=4, num_classes=num_classes,
                                       dropout=dropout)
        self.ann_quantum = ClassicalANN(input_dim=16, num_classes=num_classes,
                                        dropout=dropout)

    def forward(self, spike_data, use_quantum=False):
        """Forward pass through the full QSNN pipeline.

        Args:
            spike_data: Tensor of shape (num_steps, B, 3, 32, 32)
                        — rate-encoded spike trains.
            use_quantum: If True, route features through the quantum layer.

        Returns:
            Tensor of shape (B, 10) — class logits.
        """
        # ── Stage 1: Convolutional SNN encoder ──
        features = self.conv_snn(spike_data)  # (B, 4)

        if use_quantum:
            # ── Stage 2: Quantum circuit ──
            # TorchConnector runs on CPU; move features appropriately
            device = features.device
            features_cpu = features.to('cpu')
            q_out = self.quantum_layer(features_cpu)  # (B, 16) on CPU
            q_out = q_out.to(device)

            # ── Stage 3: Classical ANN (quantum head) ──
            return self.ann_quantum(q_out)
        else:
            # ── Bypass: skip quantum, use direct ANN head ──
            return self.ann_bypass(features)
