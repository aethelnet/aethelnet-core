# Aethelnet Core 🧠💧

**Aethelnet Core** is the mathematical and logical foundation of the Liquid Graph Neural Network (LGNN). It provides a continuously evolving, non-Euclidean representation of concepts, beliefs, and observations using continuous-time Ordinary Differential Equations (ODEs).

## Architecture

At its heart, Aethelnet is an information ecosystem. It doesn't just store data; it allows data to organically grow, decay, and form emergent connections over time.

### Liquid Graph (`liquid_graph.py`)
- **Continuous ODE Solvers:** Uses `torchdiffeq` to simulate continuous flow.
- **Physics Sharding:** Solves the $O(N^2)$ memory bottleneck via Stochastic Neighborhood Evolution. Graph interactions scale linearly, allowing for millions of active nodes on consumer hardware.
- **Dynamic Topology:** Nodes naturally form bridges (Hebbian learning) when their multi-dimensional embeddings resonate, and sever bridges when they drift apart.

## Installation

Aethelnet Core is designed to be embedded into execution layers (like `aethelnet-node`).

```bash
pip install -e .
```

## Usage

```python
import torch
from aethelnet.liquid_graph import LiquidGraph

# Initialize the 768-D Information Space
lgnn = LiquidGraph(hidden_dim=768, resonance_threshold=0.85)

# Seed an observation
lgnn.add_node("concept_a", torch.randn(768))

# Run the continuous evolution physics engine
lgnn.evolve_topology(compute_time=1.0)
```

## Design Philosophy

- **Decentralized Truth:** There is no single master node. Truth emerges from the topological consensus of the graph.
- **Organic Decay:** Information that is not validated or reinforced slowly decays into noise, preventing context pollution.
