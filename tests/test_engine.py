import os
import tempfile
import unittest
import torch
from starlette.testclient import TestClient
from aethelnet.engine import AethelEngine, app


class TestEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AethelEngine(hidden_dim=32)
        self.client = TestClient(app)

    def test_deterministic_injection_and_energy(self):
        # Inject without vector: deterministic hash seeding
        seed_bytes = b"DeterministicConcept"
        import hashlib
        h = hashlib.sha256(seed_bytes).digest()
        seed_int = int.from_bytes(h[:8], byteorder="big")
        g = torch.Generator().manual_seed(seed_int)
        expected_v = torch.randn(32, generator=g)
        expected_v = expected_v / (expected_v.norm() + 1e-8)

        self.engine.graph.add_node("DeterministicConcept", expected_v)
        energy = self.engine.get_concept_energy("DeterministicConcept")
        self.assertAlmostEqual(energy, 1.0, places=4)

        # Query resonance
        matches = self.engine.query_resonance("DeterministicConcept", k=1)
        # Should be empty since it excludes itself
        self.assertEqual(len(matches), 0)

        # Add second node and query
        v2 = expected_v.clone()
        v2[0] += 0.01
        v2 = v2 / v2.norm()
        self.engine.graph.add_node("NeighborConcept", v2)

        matches = self.engine.query_resonance("DeterministicConcept", k=5)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["node_id"], "NeighborConcept")
        self.assertGreater(matches[0]["score"], 0.99)

    def test_engine_save_and_load(self):
        v = torch.randn(32)
        v = v / v.norm()
        self.engine.graph.add_node("NodeX", v)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.engine.save_graph(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            new_engine = AethelEngine(hidden_dim=32)
            new_engine.load_graph(tmp_path)
            self.assertIn("NodeX", new_engine.graph.nx_graph)
            emb = new_engine.graph.get_node_embedding("NodeX")
            self.assertIsNotNone(emb)
            self.assertTrue(torch.allclose(emb, v))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_fastapi_endpoints(self):
        # Inject node via API
        resp = self.client.post("/api/inject?node_id=ApiConcept")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "injected")
        self.assertEqual(data["node_id"], "ApiConcept")

        # Query via API
        resp2 = self.client.get("/api/graph/query?query=ApiConcept")
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertEqual(data2["query"], "ApiConcept")
        self.assertIsInstance(data2["matches"], list)


if __name__ == "__main__":
    unittest.main()
