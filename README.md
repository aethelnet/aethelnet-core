# Aethelnet Core: Continuous-Time Liquid Graph Neural Networks

Aethelnet Core is the mathematical foundation of the Liquid Graph Neural Network (LGNN) architecture. It completely replaces the discrete-time, static weight matrices of traditional Deep Learning with a continuous-time, dynamic topological state machine governed by Ordinary Differential Equations (ODEs).

This repository contains the pure physics and mathematical engine. For the cybernetic daemon, P2P mesh network, and sensory ingestion layers, refer to the `aethelnet-node` repository.

## Theoretical Architecture

At its core, the Aethelnet LGNN is a non-Euclidean information ecosystem. The graph topology is entirely fluid: nodes organically drift, form bridges based on Hebbian resonance, and decay over time if unsupported by external sensory stimuli or internal recursive validation.

### 1. The ODE Topology Engine
The network state does not progress in discrete inference steps. Instead, the continuous flow of the node activation state $h(t)$ is solved using Neural ODEs:

$$ \frac{dh_i(t)}{dt} = -\tau_i h_i(t) + f\left( \sum_{j \in N(i)} W_{ij}(t) h_j(t) + I_i(t) \right) $$

Where:
*   **$\tau_i$**: The node-specific decay rate (thermodynamic cooling).
*   **$W_{ij}(t)$**: The dynamic synaptic weight bridging concept $i$ and $j$.
*   **$f(\cdot)$**: A non-linear squashing function (e.g., `tanh`).
*   **$I_i(t)$**: External sensory stimuli.

### 2. Hebbian Synaptic Plasticity
Synaptic connections ($W_{ij}$) between concept nodes are governed by continuous Hebbian learning rules. The evolution of the weights is mathematically defined as:

$$ \frac{dW_{ij}(t)}{dt} = \eta \cdot (h_i(t) \cdot h_j(t)) - \gamma \cdot W_{ij}(t) $$

Nodes that fire together synchronously strengthen their geometric bond ($\eta$). Unused edges are continuously penalized by the weight decay factor ($\gamma$) until they hit zero and are automatically pruned from the adjacency matrix, effectively achieving dynamic sparsity.

### 3. Physics Sharding (Stochastic Neighborhood Evolution)
A fully connected graph evolving via ODEs poses an $O(N^2)$ memory and computational bottleneck. Aethelnet Core circumvents this via "Physics Sharding." The ODE solver only evaluates active topological neighborhoods defined by geometric proximity (cosine similarity thresholding). Graph interactions thus scale linearly, allowing for millions of active nodes on standard hardware.

## Installation

Aethelnet Core is built on PyTorch and `torchdiffeq`. It is designed to be embedded into external execution layers as the primary cognitive engine.

```bash
git clone https://github.com/aethelnet/aethelnet-core.git
cd aethelnet-core
pip install -e .
```

## Basic Usage

```python
import torch
from aethelnet.liquid_graph import LiquidGraph

# Initialize the High-Dimensional Information Space
lgnn = LiquidGraph(hidden_dim=768, resonance_threshold=0.85)

# Seed independent observations (t=0)
lgnn.add_node("concept_a", torch.randn(768))
lgnn.add_node("concept_b", torch.randn(768))

# Run the continuous evolution physics engine (dt=1.0)
lgnn.evolve_topology(compute_time=1.0)
```

## Academic Research

This framework serves as a testbed for researchers exploring:
*   Continuous-time representation learning.
*   Dynamical systems in Graph Neural Networks.
*   Self-organizing, decentralized knowledge graphs.

If you are applying Neural ODEs or Liquid Time-Constant Networks to macroscopic P2P topologies, we welcome your contributions to the core physics engine.

*License: AGPL-3.0*
