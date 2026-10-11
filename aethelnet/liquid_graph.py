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
    def __init__(self, hidden_dim: int = 768, resonance_threshold: float = 0.65, decay_rate: float = 0.05, tether_strength: float = 0.50, shard_size: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.resonance_threshold = resonance_threshold
        self.decay_rate = decay_rate
        self.tether_strength = tether_strength
        self.shard_size = shard_size
        
        # NetworkX representation for structural tracking
        self.nx_graph = nx.Graph()
        
        # Mapping from node_id -> node_embedding (torch parameter)
        self.nodes = nn.ParameterDict()
        # Semantic Anchors: safe_id -> original grounding embedding (Elastic Tether ground truth)
        self.anchors: Dict[str, torch.Tensor] = {}
        
        self._id_map: Dict[str, str] = {}
        self._safe_map: Dict[str, str] = {}
        
        # Persona definitions: persona_name -> list of node_ids
        self.personas: Dict[str, List[str]] = {}
        # Active status: persona_name -> boolean
        self.active_personas: Dict[str, bool] = {}
        
        # Track node movement (historical variance) to assign Shielded (Axiom) Nodes
        self.node_variances: Dict[str, float] = {}
        
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
            logger.debug(f"[LGNN] Node {node_id} already exists. Updating embedding.")
            
        normed = torch.nn.functional.normalize(seed_embedding, p=2, dim=0)
        self.nodes[safe_id] = nn.Parameter(normed.clone().detach())
        self.anchors[safe_id] = normed.clone().detach()
        self.nx_graph.add_node(node_id)
        
        if connections:
            for target in connections:
                if self.nx_graph.has_node(target):
                    self.nx_graph.add_edge(node_id, target, weight=1.0)
        
        logger.debug(f"[LGNN] Spawned node: '{node_id}' with hidden dim {self.hidden_dim}.")

    def remove_node(self, node_id: str):
        safe_id = self._safe_id(node_id)
        if safe_id in self.nodes:
            del self.nodes[safe_id]
            if safe_id in self.anchors:
                del self.anchors[safe_id]
            self.nx_graph.remove_node(node_id)
            # Remove from persona lists
            for p_name in list(self.personas.keys()):
                if node_id in self.personas[p_name]:
                    self.personas[p_name].remove(node_id)
            logger.info(f"[LGNN] Severed node: '{node_id}'.")
            del self._id_map[node_id]
            del self._safe_map[safe_id]

    def set_anchor(self, node_id: str, anchor_embedding: torch.Tensor):
        """Updates the semantic ground-truth anchor for a node (e.g. after note editing)."""
        safe_id = self._safe_id(node_id)
        normed = torch.nn.functional.normalize(anchor_embedding.to(dtype=torch.float32), p=2, dim=0)
        self.anchors[safe_id] = normed.clone().detach()

    def get_node_strain(self, node_id: str) -> float:
        """
        Computes the elastic tether strain (Hookean tension) between the dynamic state
        and its original semantic anchor. Values range from 0.0 (rest) to ~2.0 (maximum deflection).
        """
        safe_id = self._safe_id(node_id)
        if safe_id not in self.nodes:
            return 0.0
        current_state = self.nodes[safe_id].detach()
        if safe_id not in self.anchors:
            return 0.0
        anchor_state = self.anchors[safe_id].to(current_state.device)
        return float(torch.norm(current_state - anchor_state).item())

    def get_node_embedding(self, node_id: str) -> Optional[torch.Tensor]:
        """Returns the detached embedding tensor for a given node_id, or None if not present."""
        safe_id = self._safe_id(node_id)
        if safe_id not in self.nodes:
            return None
        return self.nodes[safe_id].detach().clone()

    def get_node_energy(self, node_id: str) -> float:
        """Computes the activation energy (L2 norm) of the concept node."""
        emb = self.get_node_embedding(node_id)
        if emb is None:
            return 0.0
        return float(torch.norm(emb).item())

    def get_node_resonance(self, node_a: str, node_b: str) -> float:
        """Computes cosine similarity between two concept nodes in latent space."""
        emb_a = self.get_node_embedding(node_a)
        emb_b = self.get_node_embedding(node_b)
        if emb_a is None or emb_b is None:
            return 0.0
        norm_a = emb_a / (emb_a.norm() + 1e-8)
        norm_b = emb_b / (emb_b.norm() + 1e-8)
        return float(torch.dot(norm_a, norm_b).item())

    def get_top_resonant_nodes(
        self,
        query: Any,
        k: int = 5,
        persona: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns top-k most resonant nodes relative to a query node_id (str) or vector (Tensor)."""
        if not self.nodes:
            return []

        if isinstance(query, str):
            query_emb = self.get_node_embedding(query)
            if query_emb is None:
                return []
            exclude_id = query
        elif isinstance(query, torch.Tensor):
            query_emb = query.detach().clone()
            exclude_id = None
        else:
            raise TypeError("query must be a node_id string or a torch.Tensor")

        norm_query = query_emb / (query_emb.norm() + 1e-8)

        # Filter by active persona if requested
        eligible_node_ids = []
        for sid in self.nodes.keys():
            nid = self._original_id(sid)
            if nid == exclude_id:
                continue
            if persona and persona in self.personas:
                if nid not in self.personas[persona]:
                    continue
            eligible_node_ids.append(nid)

        if not eligible_node_ids:
            return []

        candidates = torch.stack([self.nodes[self._safe_id(nid)] for nid in eligible_node_ids])
        norm_candidates = candidates / (candidates.norm(dim=-1, keepdim=True) + 1e-8)
        sims = torch.matmul(norm_candidates, norm_query.to(candidates.device))

        top_indices = torch.topk(sims, min(k, len(eligible_node_ids))).indices.tolist()
        results = []
        for idx in top_indices:
            nid = eligible_node_ids[idx]
            score = float(sims[idx].item())
            results.append({
                "node_id": nid,
                "score": score,
                "energy": float(torch.norm(candidates[idx]).item()),
                "variance": self.node_variances.get(nid, 0.0),
                "is_shielded": self.nx_graph.nodes[nid].get("is_shielded", False) if self.nx_graph.has_node(nid) else False
            })
        return results

    def get_node_conviction(self, node_id: str) -> float:
        """[Deprecated] Returns energy activation mapped via tanh."""
        import math
        safe_id = self._safe_id(node_id)
        if safe_id not in self.nodes:
            return 0.0
        
        with torch.no_grad():
            state_tensor = self.nodes[safe_id]
            raw_score = state_tensor.mean().item()
            conviction = math.tanh(raw_score * 10.0)
            
        return conviction

    def forward(self, t, latent_states: torch.Tensor) -> torch.Tensor:
        # Stabilize ODE gradients
        latent_states = torch.nan_to_num(latent_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get active node mask
        node_ids = getattr(self, "cached_node_ids", None)
        if node_ids is None:
            safe_ids = list(self.nodes.keys())
            node_ids = [self._original_id(sid) for sid in safe_ids]
            
        # The mask is constant during one solve; evolve_topology computes it once
        mask = getattr(self, "cached_mask", None)
        if mask is None:
            q_nodes = getattr(self, "current_quarantined_nodes", None)
            mask = self.get_node_mask(node_ids, latent_states.device, quarantined_nodes=q_nodes)

        # Apply mask to node states
        masked_states = latent_states * mask

        # 1. Relational flow using normalized Graph Laplacian diffusion
        # L = D^-1 * A - I (continuous Laplacian diffusion), row-normalized to prevent
        # unbounded energy amplification
        norm_adj = getattr(self, "cached_norm_adj", None)
        if norm_adj is not None:
            # Sparse D^-1 A precomputed once per evolve_topology call: memory scales with edges, not N^2
            relational_flow = torch.sparse.mm(norm_adj, masked_states) - masked_states
        else:
            adj_matrix = getattr(self, "cached_adj_matrix", None)
            if adj_matrix is None:
                num_nodes = latent_states.size(0)
                adj_matrix = torch.eye(num_nodes, device=latent_states.device, dtype=torch.float32)
            deg = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
            norm_adj = adj_matrix / deg
            relational_flow = torch.matmul(norm_adj, masked_states) - masked_states
        
        # Shielded Nodes Flag Injection
        # Axiom nodes (shielded) are mathematically frozen (dx/dt = 0)
        shield_mask = getattr(self, "cached_shield_mask", None)
        if shield_mask is None:
            shield_mask = torch.ones_like(mask)
        
        # 2. Hookean Elastic Tether Restoration
        anchor_matrix = getattr(self, "cached_anchor_matrix", None)
        if anchor_matrix is not None:
            tether_force = -self.tether_strength * (masked_states - anchor_matrix)
        else:
            tether_force = torch.zeros_like(masked_states)

        # Pure Harmonic Graph Heat Equation with Hookean Dirichlet Potential:
        # dx/dt = alpha * (D^-1 A - I) x - lambda * (x - x_0)
        # Eliminates artificial MLP attractor collapse and strictly preserves conceptual distinctness.
        grad = (relational_flow * 0.35 + tether_force) * mask * shield_mask
        grad = torch.clamp(grad, -2.0, 2.0)
        return torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)

    def evolve_topology(self, compute_time: float = 1.0, quarantined_nodes: List[str] = None):
        if not self.nodes:
            return {
                "new_edges": [], "strengthened_edges": [],
                "decayed_edges": [], "pruned_edges": [],
                "active_nodes": 0, "active_edges": 0,
                "mean_tether_strain": 0.0, "max_strain_node": None, "max_strain_val": 0.0
            }
            
        safe_ids = list(self.nodes.keys())
        node_ids = [self._original_id(sid) for sid in safe_ids]
        # Node states are nn.Parameters, but nothing ever backpropagates through the solve.
        # Detaching (and solving under no_grad below) stops autograd from keeping every RK4
        # intermediate alive, which made peak memory ~40x the size of the adjacency.
        # NOTE (Architectural Handover): If End-to-End Neural ODE Training with backpropagation
        # through the solve is ever needed, DO NOT re-enable naive autograd BPTT. Instead,
        # use the Adjoint Sensitivity Method (torchdiffeq.odeint_adjoint) to preserve O(1) memory!
        # AGENT-NOTE(claude, 2026-10-11): Agreed. Caveat: the adjoint re-integrates backwards in time,
        # and these dynamics are contractive forwards (|eigenvalues| ~0.5-1.2), so error grows roughly
        # like exp(1.2 * T) on the way back. For short fixed-step RK4 solves, torch.utils.checkpoint
        # gives exact gradients and may be simpler. See docs/IDEAS.md #8.
        initial_states = torch.stack([self.nodes[sid].detach() for sid in safe_ids])
        num_nodes = len(node_ids)

        # Build Sparse COO Tensor from NetworkX and row-normalize it once (D^-1 A)
        import numpy as np

        if len(self.nx_graph.edges) > 0:
            adj_scipy = nx.to_scipy_sparse_array(self.nx_graph, nodelist=node_ids, format="coo", weight="weight")
            rows = torch.as_tensor(adj_scipy.row, dtype=torch.long)
            cols = torch.as_tensor(adj_scipy.col, dtype=torch.long)
            vals = torch.as_tensor(adj_scipy.data, dtype=torch.float32)
        else:
            rows = cols = torch.empty(0, dtype=torch.long)
            vals = torch.empty(0, dtype=torch.float32)
        deg = torch.zeros(num_nodes, dtype=torch.float32).index_add_(0, rows, vals).clamp(min=1.0)
        norm_adj = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals / deg[rows], (num_nodes, num_nodes), check_invariants=True
        ).coalesce()

        self.cached_norm_adj = norm_adj.to(initial_states.device)
        self.current_quarantined_nodes = quarantined_nodes
        self.cached_node_ids = node_ids
        self.cached_mask = self.get_node_mask(node_ids, initial_states.device, quarantined_nodes=quarantined_nodes)

        # Cache anchor matrix for Elastic Tether
        anchors_list = []
        for sid in safe_ids:
            if sid in self.anchors:
                anchors_list.append(self.anchors[sid].to(initial_states.device))
            else:
                anchors_list.append(self.nodes[sid].detach().clone().to(initial_states.device))
        self.cached_anchor_matrix = torch.stack(anchors_list)
        
        # Determine top 5 stable nodes to shield
        stable_nodes = set()
        if len(self.node_variances) >= 5:
            sorted_nodes = sorted(self.node_variances.items(), key=lambda x: x[1])
            stable_nodes = {nid for nid, var in sorted_nodes[:5]}
            
        shield_mask = torch.ones((len(node_ids), 1), device=initial_states.device, dtype=torch.float32)
        for i, nid in enumerate(node_ids):
            is_shield = (nid in stable_nodes)
            self.nx_graph.nodes[nid]['is_shielded'] = is_shield
            if is_shield:
                shield_mask[i] = 0.0
                logger.debug(f"[LGNN] Shielding Axiom Node: {nid}")
        
        self.cached_shield_mask = shield_mask
        
        # Solve the ODE
        t_span = torch.tensor([0.0, compute_time])
        t0 = time.perf_counter()
        
        try:
            with torch.no_grad():
                refined_states = odeint(self, initial_states, t_span, method='rk4', options={'step_size': 0.1})[-1]
                refined_states = torch.nan_to_num(refined_states, nan=0.0, posinf=1.0, neginf=-1.0)
                refined_states = torch.nn.functional.normalize(refined_states, p=2, dim=-1)
        finally:
            self.current_quarantined_nodes = None
            self.cached_norm_adj = None
            self.cached_mask = None
            self.cached_node_ids = None
            self.cached_shield_mask = None
            self.cached_anchor_matrix = None
        duration = time.perf_counter() - t0
        
        self.hardware_speed_factor = compute_time / (duration + 1e-8)
        logger.debug(f"[LGNN] Evolved topology. Wall-clock: {duration:.4f}s | Virtual: {compute_time}s | Speed Factor: {self.hardware_speed_factor:.2f}x")
        
        # Update node parameters and historical variance
        with torch.no_grad():
            movements = torch.norm(refined_states - initial_states, dim=1)
            for i, sid in enumerate(safe_ids):
                if sid not in self.nodes:
                    continue
                self.nodes[sid].copy_(refined_states[i])
                
                # Update variance
                orig_id = node_ids[i]
                mov = movements[i].item()
                if orig_id in self.node_variances:
                    self.node_variances[orig_id] = 0.9 * self.node_variances[orig_id] + 0.1 * mov
                else:
                    self.node_variances[orig_id] = mov
                
        # --- Vectorized Hebbian Edge Evolution ---
        new_edges = []
        strengthened_edges = []
        decayed_edges = []
        pruned_edges = []

        normalized_states = refined_states
        mask = self.get_node_mask(node_ids, refined_states.device, quarantined_nodes=quarantined_nodes).squeeze(1)
        
        num_nodes = normalized_states.size(0)
        SHARD_SIZE = getattr(self, "shard_size", 10000)
        
        if num_nodes > SHARD_SIZE:
            # PHYSICS SHARDING
            logger.info(f"[LGNN] Applying Physics Sharding: Selecting random {SHARD_SIZE}-node neighborhood.")
            shard_indices = torch.randperm(num_nodes, device=refined_states.device)[:SHARD_SIZE]
            shard_states = normalized_states[shard_indices]
            shard_mask = mask[shard_indices]
            
            similarity_matrix = torch.matmul(shard_states, shard_states.T)
            valid_mask = torch.outer(shard_mask, shard_mask)
            valid_mask.fill_diagonal_(0.0) # No self-edges
            similarity_matrix = similarity_matrix * valid_mask
            
            high_sim_indices = torch.nonzero(similarity_matrix >= self.resonance_threshold, as_tuple=False)
            high_sim_indices = high_sim_indices[high_sim_indices[:, 0] < high_sim_indices[:, 1]] # Undirected
            
            for idx in range(high_sim_indices.size(0)):
                i_local, j_local = high_sim_indices[idx].tolist()
                i = shard_indices[i_local].item()
                j = shard_indices[j_local].item()
                u, v = node_ids[i], node_ids[j]
                similarity = float(similarity_matrix[i_local, j_local].item())
                if self.nx_graph.has_edge(u, v):
                    current_weight = float(self.nx_graph[u][v].get('weight', 1.0))
                    new_w = min(current_weight + 0.05, 1.0)
                    self.nx_graph[u][v]['weight'] = new_w
                    strengthened_edges.append((u, v, round(new_w, 3)))
                else:
                    self.nx_graph.add_edge(u, v, weight=similarity, label="hebbian")
                    new_edges.append((u, v, round(similarity, 3)))
                    logger.debug(f"[LGNN] Shard Spawned bridge '{u}' <-> '{v}' (Sim: {similarity:.2f})")
        else:
            # O(N^2) Full Matrix for small/medium graphs
            similarity_matrix = torch.matmul(normalized_states, normalized_states.T)
            valid_mask = torch.outer(mask, mask)
            valid_mask.fill_diagonal_(0.0) # No self-edges
            similarity_matrix = similarity_matrix * valid_mask
            
            high_sim_indices = torch.nonzero(similarity_matrix >= self.resonance_threshold, as_tuple=False)
            high_sim_indices = high_sim_indices[high_sim_indices[:, 0] < high_sim_indices[:, 1]] # Undirected
            
            for idx in range(high_sim_indices.size(0)):
                i, j = high_sim_indices[idx].tolist()
                u, v = node_ids[i], node_ids[j]
                similarity = float(similarity_matrix[i, j].item())
                if self.nx_graph.has_edge(u, v):
                    current_weight = float(self.nx_graph[u][v].get('weight', 1.0))
                    new_w = min(current_weight + 0.05, 1.0)
                    self.nx_graph[u][v]['weight'] = new_w
                    strengthened_edges.append((u, v, round(new_w, 3)))
                else:
                    self.nx_graph.add_edge(u, v, weight=similarity, label="hebbian")
                    new_edges.append((u, v, round(similarity, 3)))
                    logger.debug(f"[LGNN] Spawned bridge '{u}' <-> '{v}' (Sim: {similarity:.2f})")
                
        # Synaptic decay of existing dynamic edges
        edges = list(self.nx_graph.edges(data=True))
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        if num_nodes > SHARD_SIZE:
            shard_idx_map = {shard_indices[k].item(): k for k in range(min(SHARD_SIZE, num_nodes))}
            for u, v, data in edges:
                if data.get('is_manual') or data.get('label') in ('wikilink', 'wiki_link'):
                    continue
                if u not in node_id_to_idx or v not in node_id_to_idx:
                    continue
                i, j = node_id_to_idx[u], node_id_to_idx[v]
                if i in shard_idx_map and j in shard_idx_map:
                    loc_i, loc_j = shard_idx_map[i], shard_idx_map[j]
                    if similarity_matrix[loc_i, loc_j] < self.resonance_threshold:
                        current_weight = float(data.get('weight', 1.0))
                        new_weight = current_weight - self.decay_rate
                        if new_weight <= 0.05:
                            self.nx_graph.remove_edge(u, v)
                            pruned_edges.append((u, v))
                        else:
                            self.nx_graph[u][v]['weight'] = round(new_weight, 3)
                            decayed_edges.append((u, v, round(new_weight, 3)))
        else:
            for u, v, data in edges:
                if data.get('is_manual') or data.get('label') in ('wikilink', 'wiki_link'):
                    continue
                if u not in node_id_to_idx or v not in node_id_to_idx:
                    continue
                i, j = node_id_to_idx[u], node_id_to_idx[v]
                if mask[i] == 0.0 or mask[j] == 0.0:
                    continue
                if similarity_matrix[i, j] < self.resonance_threshold:
                    current_weight = float(data.get('weight', 1.0))
                    new_weight = current_weight - self.decay_rate
                    if new_weight <= 0.05:
                        self.nx_graph.remove_edge(u, v)
                        pruned_edges.append((u, v))
                    else:
                        self.nx_graph[u][v]['weight'] = round(new_weight, 3)
                        decayed_edges.append((u, v, round(new_weight, 3)))

        strains = [self.get_node_strain(nid) for nid in node_ids]
        mean_strain = float(sum(strains) / len(strains)) if strains else 0.0
        max_strain_idx = int(np.argmax(strains)) if strains else 0
        max_strain_node = node_ids[max_strain_idx] if strains else None
        max_strain_val = float(strains[max_strain_idx]) if strains else 0.0

        return {
            "new_edges": new_edges,
            "strengthened_edges": strengthened_edges,
            "decayed_edges": decayed_edges,
            "pruned_edges": pruned_edges,
            "active_nodes": num_nodes,
            "active_edges": self.nx_graph.number_of_edges(),
            "mean_tether_strain": round(mean_strain, 4),
            "max_strain_node": max_strain_node,
            "max_strain_val": round(max_strain_val, 4)
        }

    def get_resonance(self, node_a: str, node_b: str) -> float:
        """Calculates exact cosine resonance between two nodes on the unit sphere."""
        safe_a = self._safe_id(node_a)
        safe_b = self._safe_id(node_b)
        if safe_a not in self.nodes or safe_b not in self.nodes:
            return 0.0
        va = torch.nn.functional.normalize(self.nodes[safe_a], p=2, dim=-1)
        vb = torch.nn.functional.normalize(self.nodes[safe_b], p=2, dim=-1)
        return float(torch.dot(va, vb).item())

    def save_state(self, filepath: str):
        """
        Serializes complete graph state, topology, weights, and parameters to disk.
        """
        import os
        edges_data = []
        for u, v, d in self.nx_graph.edges(data=True):
            edges_data.append({
                "u": u,
                "v": v,
                "weight": float(d.get("weight", 1.0)),
                "attrs": {k: v_val for k, v_val in d.items() if k != "weight"}
            })

        nodes_tensors = {}
        nodes_attrs = {}
        for safe_id, param in self.nodes.items():
            orig_id = self._original_id(safe_id)
            nodes_tensors[orig_id] = param.detach().cpu().clone()
            if self.nx_graph.has_node(orig_id):
                nodes_attrs[orig_id] = dict(self.nx_graph.nodes[orig_id])

        state = {
            "version": "1.1.0",
            "hidden_dim": self.hidden_dim,
            "resonance_threshold": self.resonance_threshold,
            "decay_rate": self.decay_rate,
            "shard_size": getattr(self, "shard_size", 10000),
            "nodes_tensors": nodes_tensors,
            "nodes_attrs": nodes_attrs,
            "edges": edges_data,
            "personas": self.personas,
            "active_personas": self.active_personas,
            "node_variances": self.node_variances,
            "_id_map": self._id_map,
            "_safe_map": self._safe_map,
            "flow_dynamics_state": self.flow_dynamics.state_dict(),
            "resonance_aligner_state": self.resonance_aligner.state_dict(),
            "hardware_speed_factor": self.hardware_speed_factor
        }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(state, filepath)
        logger.info(f"[LGNN] Saved graph state ({len(nodes_tensors)} nodes, {len(edges_data)} edges) to '{filepath}'.")

    def load_state(self, filepath: str, map_location=None):
        """
        Loads and reconstructs complete graph state, topology, and parameters from disk.
        """
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph checkpoint file not found: {filepath}")

        state = torch.load(filepath, map_location=map_location or torch.device("cpu"), weights_only=False)

        checkpoint_dim = state.get("hidden_dim", self.hidden_dim)
        if checkpoint_dim != self.hidden_dim:
            self.hidden_dim = checkpoint_dim
            self.flow_dynamics = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Tanh()
            )
            heads = 8 if self.hidden_dim % 8 == 0 else (4 if self.hidden_dim % 4 == 0 else 1)
            self.resonance_aligner = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=heads,
                batch_first=True
            )

        self.resonance_threshold = state.get("resonance_threshold", self.resonance_threshold)
        self.decay_rate = state.get("decay_rate", self.decay_rate)
        self.shard_size = state.get("shard_size", getattr(self, "shard_size", 10000))
        self.hardware_speed_factor = state.get("hardware_speed_factor", 1.0)

        # Clear current graph
        self.nx_graph.clear()
        self.nodes = nn.ParameterDict()
        self._id_map = state.get("_id_map", {})
        self._safe_map = state.get("_safe_map", {})
        self.personas = state.get("personas", {})
        self.active_personas = state.get("active_personas", {})
        self.node_variances = state.get("node_variances", {})

        # Restore sub-modules
        if "flow_dynamics_state" in state:
            self.flow_dynamics.load_state_dict(state["flow_dynamics_state"])
        if "resonance_aligner_state" in state:
            self.resonance_aligner.load_state_dict(state["resonance_aligner_state"])

        # Restore nodes
        nodes_tensors = state.get("nodes_tensors", {})
        nodes_attrs = state.get("nodes_attrs", {})
        for orig_id, tensor in nodes_tensors.items():
            safe_id = self._safe_id(orig_id)
            self.nodes[safe_id] = nn.Parameter(tensor.to(dtype=torch.float32))
            attrs = nodes_attrs.get(orig_id, {})
            self.nx_graph.add_node(orig_id, **attrs)

        # Restore edges
        edges_data = state.get("edges", [])
        for e in edges_data:
            u, v = e["u"], e["v"]
            weight = e.get("weight", 1.0)
            attrs = e.get("attrs", {})
            self.nx_graph.add_edge(u, v, weight=weight, **attrs)

        logger.info(f"[LGNN] Loaded graph state ({len(self.nodes)} nodes, {len(self.nx_graph.edges)} edges) from '{filepath}'.")
