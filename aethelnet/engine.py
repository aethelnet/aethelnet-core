import asyncio
import logging
import torch
import os
from fastapi import FastAPI, WebSocket
from typing import Dict, Any, List

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
        """
        Schnittstelle für die BrainEngineFull. 
        Liest die aktuelle Trading-Conviction für ein Symbol aus dem Graph aus.
        """
        return self.graph.get_node_conviction(symbol)

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

api_key_header = APIKeyHeader(name="X-Auth-Token")

AUTH_TOKEN = os.environ.get("AETHELNET_AUTH_TOKEN")
if not AUTH_TOKEN:
    raise RuntimeError("AETHELNET_AUTH_TOKEN environment variable is required")

@app.post("/api/inject")
async def inject_node(node_id: str, api_key: str = Security(api_key_header)):
    """
    Inject a new concept into the graph.
    """
    if api_key != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
        
    async with engine.graph_lock:
        seed_emb = torch.randn(engine.graph.hidden_dim)
        seed_emb = seed_emb / (seed_emb.norm() + 1e-8)
        engine.graph.add_node(node_id, seed_emb)
    return {"status": "injected", "node_id": node_id}
