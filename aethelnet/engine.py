import asyncio
import logging
import torch
import os
from fastapi import FastAPI, WebSocket
from typing import Dict, Any, List, Optional

from .liquid_graph import LiquidGraph

logger = logging.getLogger("Aethelnet.Engine")

class AethelEngine:
    """
    The orchestrator for the Liquid Graph Neural Network.
    Manages the continuous ignition loop and serves as the Swarm Node backend.
    """
    def __init__(self, hidden_dim: int = 768):
        self.graph = LiquidGraph(hidden_dim=hidden_dim)
        self.is_running = False
        self.connected_clients: List[WebSocket] = []
        self.graph_lock = asyncio.Lock()

    def get_symbol_conviction(self, symbol: str) -> float:
        """[Deprecated] Compatibility shim returning node conviction."""
        return self.graph.get_node_conviction(symbol)

    def get_concept_energy(self, concept_id: str) -> float:
        """Reads concept activation energy (L2 norm) from the graph."""
        return self.graph.get_node_energy(concept_id)

    def query_resonance(self, query_id_or_tensor: Any, k: int = 5, persona: Optional[str] = None) -> List[Dict[str, Any]]:
        """Finds top-k most resonant concepts in the topological space."""
        return self.graph.get_top_resonant_nodes(query_id_or_tensor, k=k, persona=persona)

    def save_graph(self, filepath: str):
        """Persists entire graph topology and weights to disk."""
        self.graph.save_state(filepath)

    def load_graph(self, filepath: str):
        """Restores graph topology and weights from disk."""
        self.graph.load_state(filepath)

    async def ignition_loop(self, tick_interval: float = 1.0):
        """
        The continuous background loop that evolves the graph over time.
        This represents the 'Flow' or 'Meditation' state of the node.
        """
        self.is_running = True
        logger.info("[AethelEngine] Ignition sequence started. Entering continuous ODE flow.")
        
        while self.is_running:
            try:
                # Evolve topology through Neural ODEs asynchronously to prevent blocking the event loop
                loop = asyncio.get_event_loop()
                async with self.graph_lock:
                    await loop.run_in_executor(None, lambda: self.graph.evolve_topology(compute_time=1.0))
                
                # Broadcast the new state to all connected swarm clients
                await self.broadcast_state()
                
            except Exception as e:
                logger.error(f"[AethelEngine] Error in ignition loop: {e}")
                
            await asyncio.sleep(tick_interval)

    async def broadcast_state(self):
        """
        Splatter the distilled 'Senf' (state) to the connected Swarm.
        """
        if not self.connected_clients:
            return
            
        async with self.graph_lock:
            nodes_data = [{"id": n} for n in self.graph.nx_graph.nodes()]
            links_data = [
                {"source": u, "target": v, "weight": float(d.get("weight", 0.0))}
                for u, v, d in self.graph.nx_graph.edges(data=True)
            ]

        state_summary = {
            "type": "network_layout",
            "nodes": nodes_data,
            "links": links_data,
            "metrics": {
                "node_count": len(nodes_data),
                "edge_count": len(links_data),
                "speed_factor": self.graph.hardware_speed_factor
            }
        }
        
        dead_clients = []
        for ws in self.connected_clients:
            try:
                await ws.send_json(state_summary)
            except Exception:
                dead_clients.append(ws)
                
        for ws in dead_clients:
            if ws in self.connected_clients:
                self.connected_clients.remove(ws)

# --- Minimal FastAPI Server Implementation ---

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the continuous evolution in the background
    task = asyncio.create_task(engine.ignition_loop())
    yield
    engine.is_running = False
    task.cancel()

app = FastAPI(title="Aethelnet Swarm Node", lifespan=lifespan)
engine = AethelEngine()

@app.websocket("/ws/swarm")
async def swarm_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the Swarm App (Federated Learning).
    Receives connections from mobile clients or other nodes.
    """
    await websocket.accept()
    engine.connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Here we would merge remote tensors from the Swarm
            pass
    except Exception:
        if websocket in engine.connected_clients:
            engine.connected_clients.remove(websocket)

from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-Auth-Token", auto_error=False)

AUTH_TOKEN = os.environ.get("AETHELNET_AUTH_TOKEN")

@app.post("/api/inject")
async def inject_node(
    node_id: str,
    embedding: Optional[List[float]] = None,
    api_key: str = Security(api_key_header)
):
    """
    Inject a new concept into the graph.
    Accepts explicit embedding or seeds deterministically from the concept's SHA256.
    """
    if AUTH_TOKEN and api_key != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
        
    async with engine.graph_lock:
        if embedding is not None:
            if len(embedding) != engine.graph.hidden_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected embedding dimension {engine.graph.hidden_dim}, got {len(embedding)}"
                )
            seed_emb = torch.tensor(embedding, dtype=torch.float32)
        else:
            # Deterministic hash projection prevents random noise contamination
            import hashlib
            seed_bytes = hashlib.sha256(node_id.encode("utf-8")).digest()
            seed_int = int.from_bytes(seed_bytes[:8], byteorder="big")
            g = torch.Generator().manual_seed(seed_int)
            seed_emb = torch.randn(engine.graph.hidden_dim, generator=g)

        seed_emb = seed_emb / (seed_emb.norm() + 1e-8)
        engine.graph.add_node(node_id, seed_emb)
    return {"status": "injected", "node_id": node_id}

@app.post("/api/graph/save")
async def save_graph_checkpoint(filepath: str, api_key: str = Security(api_key_header)):
    if AUTH_TOKEN and api_key != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    async with engine.graph_lock:
        engine.save_graph(filepath)
    return {"status": "saved", "filepath": filepath}

@app.post("/api/graph/load")
async def load_graph_checkpoint(filepath: str, api_key: str = Security(api_key_header)):
    if AUTH_TOKEN and api_key != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    async with engine.graph_lock:
        engine.load_graph(filepath)
    return {"status": "loaded", "filepath": filepath}

@app.get("/api/graph/query")
async def query_graph_resonance(query: str, k: int = 5, persona: Optional[str] = None):
    async with engine.graph_lock:
        matches = engine.query_resonance(query, k=k, persona=persona)
    return {"query": query, "matches": matches}
