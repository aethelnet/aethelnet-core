import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, Any, List, Optional
from torchdiffeq import odeint
import logging
import time

logger = logging.getLogger("LGNN.Graph")

class LiquidGraph(nn.Module):
    """
    The Liquid Graph Neural Network (LGNN) topology.
    Manages a dynamic, self-evolving set of nodes and edges where:
    - Nodes are concepts containing continuous latent states (diffeq-driven).
    - Edges evolve dynamically based on Hebbian co-firing resonance.
    - Personas (sub-graphs) can be activated or deactivated dynamically.
    """
    def __init__(self, hidden_dim: int = 768, resonance_threshold: float = 0.6, decay_rate: float = 0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.resonance_threshold = resonance_threshold
        self.decay_rate = decay_rate
        
        # NetworkX representation for structural tracking
        self.nx_graph = nx.Graph()
        
        # Mapping from node_id -> node_embedding (torch parameter)
        self.nodes = nn.ParameterDict()
        
        self._id_map: Dict[str, str] = {}
        self._safe_map: Dict[str, str] = {}
        
        # Persona definitions: persona_name -> list of node_ids
        self.personas: Dict[str, List[str]] = {}
        # Active status: persona_name -> boolean
        self.active_personas: Dict[str, bool] = {}
        
        # Hardware speed tracker (virtual time / real time ratio)
        self.hardware_speed_factor: float = 1.0
        
        # Attention layer to compute dynamic resonance/alignment between nodes
        self.resonance_aligner = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Linear layer for flow dynamics
        self.flow_dynamics = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh() # Bound the flow to prevent ODE explosion
        )

    def _safe_id(self, node_id: str) -> str:
        """PyTorch ParameterDict keys cannot contain '.'."""
        if node_id in self._id_map:
            return self._id_map[node_id]
        import hashlib
        safe = "n_" + hashlib.md5(node_id.encode()).hexdigest()[:16]
        self._id_map[node_id] = safe
        self._safe_map[safe] = node_id
        return safe

    def _original_id(self, safe_id: str) -> str:
        return self._safe_map.get(safe_id, safe_id)

    def define_persona(self, persona_name: str, node_ids: List[str]):
        """
        Groups specific concept nodes under a cryptographic persona label.
        """
        self.personas[persona_name] = node_ids
        self.active_personas[persona_name] = False # Dormant by default
        logger.info(f"[LGNN] Defined persona '{persona_name}' with nodes {node_ids}.")

    def set_persona_active(self, persona_name: str, active: bool):
        """
        Activates or deactivates a persona sub-graph.
        """
        if persona_name in self.personas:
            self.active_personas[persona_name] = active
            status = "ACTIVE" if active else "DORMANT"
            logger.info(f"[LGNN] Persona '{persona_name}' is now {status}.")

    def get_node_mask(self, node_ids: List[str], device: torch.device, quarantined_nodes: List[str] = None) -> torch.Tensor:
        """
        Computes a binary mask tensor where 0.0 isolates inactive persona nodes and quarantined nodes.
        """
        if quarantined_nodes is None:
            quarantined_nodes = []
        mask = torch.ones(len(node_ids), 1, device=device)
        for i, nid in enumerate(node_ids):
            # Check if this node belongs to any inactive personas
            for persona_name, p_nodes in self.personas.items():
                if nid in p_nodes and not self.active_personas.get(persona_name, False):
                    mask[i] = 0.0 # Isolate dormant node
                    break
            
            # Check if node is quarantined
            if nid in quarantined_nodes:
                mask[i] = 0.0 # Isolate toxic/foreign nodes from continuous flow
        return mask

    def add_node(self, node_id: str, seed_embedding: torch.Tensor, connections: Optional[List[str]] = None):
        if not isinstance(seed_embedding, torch.Tensor):
            raise TypeError("seed_embedding must be a torch.Tensor")
        if seed_embedding.shape != (self.hidden_dim,):
            raise ValueError(f"Expected embedding shape ({self.hidden_dim},), got {seed_embedding.shape}")
        seed_embedding = seed_embedding.to(dtype=torch.float32)

        safe_id = self._safe_id(node_id)
        if safe_id in self.nodes:
            logger.warning(f"[LGNN] Node {node_id} already exists. Updating embedding.")
            
        self.nodes[safe_id] = nn.Parameter(seed_embedding.clone().detach())
        self.nx_graph.add_node(node_id)
        
        if connections:
            for target in connections:
                if self.nx_graph.has_node(target):
                    self.nx_graph.add_edge(node_id, target, weight=1.0)
        
        logger.info(f"[LGNN] Spawned node: '{node_id}' with hidden dim {self.hidden_dim}.")

    def remove_node(self, node_id: str):
        safe_id = self._safe_id(node_id)
        if safe_id in self.nodes:
            del self.nodes[safe_id]
            self.nx_graph.remove_node(node_id)
            # Remove from persona lists
            for p_name in list(self.personas.keys()):
                if node_id in self.personas[p_name]:
                    self.personas[p_name].remove(node_id)
            logger.info(f"[LGNN] Severed node: '{node_id}'.")
            del self._id_map[node_id]
            del self._safe_map[safe_id]

    def forward(self, t, latent_states: torch.Tensor) -> torch.Tensor:
        # Stabilize ODE gradients
        latent_states = torch.nan_to_num(latent_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get active node mask
        safe_ids = list(self.nodes.keys())
        node_ids = [self._original_id(sid) for sid in safe_ids]
        q_nodes = getattr(self, "current_quarantined_nodes", None)
        mask = self.get_node_mask(node_ids, latent_states.device, quarantined_nodes=q_nodes)
        
        # Apply mask to node states
        masked_states = latent_states * mask
        
        # 1. Local flow dynamics
        local_flow = self.flow_dynamics(masked_states) - self.decay_rate * masked_states
        
        # 2. Relational flow using Sparse Tensors!
        adj_matrix = getattr(self, "cached_adj_matrix", None)
        if adj_matrix is None:
            num_nodes = latent_states.size(0)
            indices = torch.arange(num_nodes, device=latent_states.device).unsqueeze(0).repeat(2, 1)
            values = torch.ones(num_nodes, device=latent_states.device)
            adj_matrix = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()

        # Efficient Sparse-Dense Matrix Multiplication (Graph Convolution)
        # We skip the heavy MultiheadAttention to achieve pure Sparse Tensor speed
        relational_flow = torch.sparse.mm(adj_matrix, masked_states)
        
        # Combined gradient: dx/dt
        grad = (local_flow + relational_flow * 0.1) * mask
        return torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)

    def evolve_topology(self, compute_time: float = 1.0, quarantined_nodes: List[str] = None):
        if not self.nodes:
            return
            
        safe_ids = list(self.nodes.keys())
        node_ids = [self._original_id(sid) for sid in safe_ids]
        initial_states = torch.stack([self.nodes[sid] for sid in safe_ids])
        
        # Build Sparse COO Tensor from NetworkX
        import scipy.sparse as sp
        import numpy as np
        
        if len(self.nx_graph.edges) > 0:
            adj_scipy = nx.to_scipy_sparse_array(self.nx_graph, nodelist=node_ids, format="coo", weight="weight")
            indices = torch.tensor(np.vstack((adj_scipy.row, adj_scipy.col)), dtype=torch.long)
            values = torch.tensor(adj_scipy.data, dtype=torch.float32)
            adj_matrix = torch.sparse_coo_tensor(indices, values, size=adj_scipy.shape).coalesce()
        else:
            num_nodes = len(node_ids)
            adj_matrix = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty(0), size=(num_nodes, num_nodes)).coalesce()
            
        self.cached_adj_matrix = adj_matrix.to(initial_states.device)
        self.current_quarantined_nodes = quarantined_nodes
        
        # Solve the ODE
        t_span = torch.tensor([0.0, compute_time])
        t0 = time.perf_counter()
        
        try:
            refined_states = odeint(self, initial_states, t_span, method='rk4', options={'step_size': 0.1})[-1]
            refined_states = torch.nan_to_num(refined_states, nan=0.0, posinf=1.0, neginf=-1.0)
        finally:
            self.current_quarantined_nodes = None
            self.cached_adj_matrix = None
        duration = time.perf_counter() - t0
        
        self.hardware_speed_factor = compute_time / (duration + 1e-8)
        logger.info(f"[LGNN] Evolved topology. Wall-clock: {duration:.4f}s | Virtual: {compute_time}s | Speed Factor: {self.hardware_speed_factor:.2f}x")
        
        # Update node parameters
        with torch.no_grad():
            for i, sid in enumerate(safe_ids):
                self.nodes[sid].copy_(refined_states[i])
                
        # --- Fast Vectorized Hebbian Edge Evolution ---
        normalized_states = refined_states / (refined_states.norm(dim=-1, keepdim=True) + 1e-8)
        similarity_matrix = torch.matmul(normalized_states, normalized_states.T)
        
        mask = self.get_node_mask(node_ids, refined_states.device, quarantined_nodes=quarantined_nodes).squeeze(1)
        valid_mask = torch.outer(mask, mask)
        valid_mask.fill_diagonal_(0.0) # No self-edges
        
        similarity_matrix = similarity_matrix * valid_mask
        
        # Fast bulk operations instead of O(N^2) python loop
        high_sim_indices = torch.nonzero(similarity_matrix >= self.resonance_threshold, as_tuple=False)
        high_sim_indices = high_sim_indices[high_sim_indices[:, 0] < high_sim_indices[:, 1]] # Undirected
        
        for idx in range(high_sim_indices.size(0)):
            i, j = high_sim_indices[idx].tolist()
            u, v = node_ids[i], node_ids[j]
            similarity = float(similarity_matrix[i, j].item())
            if self.nx_graph.has_edge(u, v):
                current_weight = self.nx_graph[u][v].get('weight', 1.0)
                self.nx_graph[u][v]['weight'] = min(current_weight + 0.1, 1.0)
            else:
                self.nx_graph.add_edge(u, v, weight=0.5)
                logger.info(f"[LGNN] Spawned bridge '{u}' <-> '{v}' (Sim: {similarity:.2f})")
                
        # Fast bulk decay of existing edges
        edges = list(self.nx_graph.edges(data=True))
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        for u, v, data in edges:
            if u not in node_id_to_idx or v not in node_id_to_idx:
                continue
            i, j = node_id_to_idx[u], node_id_to_idx[v]
            if mask[i] == 0.0 or mask[j] == 0.0:
                continue
            if similarity_matrix[i, j] < self.resonance_threshold:
                current_weight = data.get('weight', 1.0)
                new_weight = current_weight - 0.05
                if new_weight <= 0.0:
                    self.nx_graph.remove_edge(u, v)
                else:
                    self.nx_graph[u][v]['weight'] = new_weight
