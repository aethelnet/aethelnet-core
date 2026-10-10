import os
import tempfile
import unittest
import torch
from aethelnet.liquid_graph import LiquidGraph


class TestLiquidGraph(unittest.TestCase):

    def test_node_lifecycle(self):
        graph = LiquidGraph(hidden_dim=32)
        v1 = torch.randn(32)
        v1 = v1 / v1.norm()

        # Add node
        graph.add_node("Alpha.Concept", v1)
        self.assertIn("Alpha.Concept", graph.nx_graph)
        emb = graph.get_node_embedding("Alpha.Concept")
        self.assertIsNotNone(emb)
        self.assertTrue(torch.allclose(emb, v1))
        self.assertGreater(graph.get_node_energy("Alpha.Concept"), 0.99)

        # Remove node
        graph.remove_node("Alpha.Concept")
        self.assertNotIn("Alpha.Concept", graph.nx_graph)
        self.assertIsNone(graph.get_node_embedding("Alpha.Concept"))

    def test_topological_queries(self):
        graph = LiquidGraph(hidden_dim=16)
        base = torch.zeros(16)
        base[0] = 1.0

        similar = torch.zeros(16)
        similar[0] = 0.9
        similar[1] = 0.1
        similar = similar / similar.norm()

        orthogonal = torch.zeros(16)
        orthogonal[5] = 1.0

        graph.add_node("Base", base)
        graph.add_node("Similar", similar)
        graph.add_node("Orthogonal", orthogonal)

        res_sim = graph.get_node_resonance("Base", "Similar")
        res_ortho = graph.get_node_resonance("Base", "Orthogonal")

        self.assertGreater(res_sim, 0.85)
        self.assertLess(abs(res_ortho), 1e-4)

        top_matches = graph.get_top_resonant_nodes("Base", k=2)
        self.assertEqual(len(top_matches), 2)
        self.assertEqual(top_matches[0]["node_id"], "Similar")
        self.assertGreater(top_matches[0]["score"], 0.85)

    def test_hebbian_edge_evolution_and_decay(self):
        graph = LiquidGraph(hidden_dim=16, resonance_threshold=0.5, decay_rate=0.01)

        # Two almost identical vectors
        v1 = torch.zeros(16)
        v1[0] = 1.0
        v2 = torch.zeros(16)
        v2[0] = 0.99
        v2[1] = 0.05
        v2 = v2 / v2.norm()

        # One orthogonal vector
        v3 = torch.zeros(16)
        v3[8] = 1.0

        graph.add_node("A", v1)
        graph.add_node("B", v2)
        graph.add_node("C", v3)

        # Evolve topology
        graph.evolve_topology(compute_time=0.5)

        # Hebbian rule should connect A and B because cosine similarity >= 0.5
        self.assertTrue(graph.nx_graph.has_edge("A", "B"), "Expected edge between A and B")
        self.assertFalse(graph.nx_graph.has_edge("A", "C"), "Did not expect edge between orthogonal A and C")

    def test_persona_masking(self):
        graph = LiquidGraph(hidden_dim=16)
        v = torch.randn(16)
        v = v / v.norm()

        graph.add_node("PublicNode", v.clone())
        graph.add_node("SecretNode", v.clone())

        graph.define_persona("ShadowGuild", ["SecretNode"])
        # By default, defined persona is dormant (inactive)
        self.assertFalse(graph.active_personas["ShadowGuild"])

        mask = graph.get_node_mask(["PublicNode", "SecretNode"], torch.device("cpu"))
        self.assertEqual(mask[0].item(), 1.0, "PublicNode should be active (mask=1.0)")
        self.assertEqual(mask[1].item(), 0.0, "SecretNode should be isolated (mask=0.0)")

        # Activate persona
        graph.set_persona_active("ShadowGuild", True)
        mask_active = graph.get_node_mask(["PublicNode", "SecretNode"], torch.device("cpu"))
        self.assertEqual(mask_active[1].item(), 1.0, "SecretNode should be active after persona activation")

    def test_physics_sharding(self):
        # Test with shard_size=4 on 10 nodes to trigger physics sharding branch
        graph = LiquidGraph(hidden_dim=8, shard_size=4, resonance_threshold=0.3)
        for i in range(10):
            v = torch.randn(8)
            v = v / v.norm()
            graph.add_node(f"Node_{i}", v)

        # Should run physics sharding without error
        graph.evolve_topology(compute_time=0.2)
        self.assertGreater(graph.hardware_speed_factor, 0.0)

    def test_persistence_roundtrip(self):
        graph = LiquidGraph(hidden_dim=32, resonance_threshold=0.25, decay_rate=0.03, shard_size=50)

        v1 = torch.randn(32)
        v1 = v1 / v1.norm()
        v2 = torch.randn(32)
        v2 = v2 / v2.norm()

        graph.add_node("Genesis.Core", v1)
        graph.add_node("Apex.Mind", v2, connections=["Genesis.Core"])
        graph.define_persona("Architect", ["Genesis.Core"])
        graph.set_persona_active("Architect", True)
        graph.node_variances["Genesis.Core"] = 0.042

        # Run one step
        graph.evolve_topology(compute_time=0.2)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            save_path = tmp.name

        try:
            # Save state
            graph.save_state(save_path)
            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)

            # Create fresh instance and load
            loaded_graph = LiquidGraph(hidden_dim=16) # intentionally wrong dim to verify load overrides
            loaded_graph.load_state(save_path)

            self.assertEqual(loaded_graph.hidden_dim, 32)
            self.assertEqual(loaded_graph.resonance_threshold, 0.25)
            self.assertEqual(loaded_graph.decay_rate, 0.03)
            self.assertEqual(loaded_graph.shard_size, 50)

            # Verify nodes and embeddings
            self.assertIn("Genesis.Core", loaded_graph.nx_graph)
            self.assertIn("Apex.Mind", loaded_graph.nx_graph)
            emb1 = loaded_graph.get_node_embedding("Genesis.Core")
            self.assertIsNotNone(emb1)
            self.assertTrue(torch.allclose(emb1, graph.get_node_embedding("Genesis.Core")))

            # Verify personas and variances
            self.assertEqual(loaded_graph.personas.get("Architect"), ["Genesis.Core"])
            self.assertTrue(loaded_graph.active_personas.get("Architect"))
            self.assertAlmostEqual(
                loaded_graph.node_variances.get("Genesis.Core", 0.0),
                graph.node_variances.get("Genesis.Core", 0.0),
                places=5
            )

            # Verify edge
            self.assertTrue(loaded_graph.nx_graph.has_edge("Genesis.Core", "Apex.Mind"))

            # Verify loaded graph can continue evolving seamlessly
            loaded_graph.evolve_topology(compute_time=0.2)
            self.assertGreater(loaded_graph.hardware_speed_factor, 0.0)

        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


def _random_graph(n=120, dim=32, n_edges=400, seed=0):
    gen = torch.Generator().manual_seed(seed)
    X = torch.nn.functional.normalize(torch.randn(n, dim, generator=gen), dim=1)
    edges = set()
    while len(edges) < n_edges:
        u, v = torch.randint(0, n, (2,), generator=gen).tolist()
        if u != v:
            edges.add((min(u, v), max(u, v)))
    graph = LiquidGraph(hidden_dim=dim, resonance_threshold=1.01)
    for i in range(n):
        graph.add_node(f"n{i}", X[i])
    for u, v in edges:
        graph.nx_graph.add_edge(f"n{u}", f"n{v}", weight=1.0)
    return graph, X, edges


class TestEvolutionDynamics(unittest.TestCase):

    def test_evolution_matches_reference_dynamics(self):
        """Pins the current dynamics dx/dt = 0.35 (D^-1 A - I) x - tether (x - x0).

        Its exact solution is computed in closed form here. If the model is changed on
        purpose (e.g. made nonlinear), update this reference in the same PR.
        """
        graph, X, edges = _random_graph()
        n, T, alpha, lam = X.shape[0], 1.0, 0.35, graph.tether_strength
        graph.evolve_topology(compute_time=T)
        out = torch.stack([graph.get_node_embedding(f"n{i}") for i in range(n)])

        A = torch.zeros(n, n)
        for u, v in edges:
            A[u, v] = A[v, u] = 1.0
        I = torch.eye(n)
        M = alpha * (A / A.sum(1, keepdim=True).clamp(min=1.0) - I) - lam * I
        E = torch.matrix_exp(M * T)
        ref = torch.nn.functional.normalize(E @ X + torch.linalg.solve(M, (E - I) @ (lam * X)), dim=1)
        self.assertLess(float((out - ref).abs().max()), 1e-5)

    def test_solve_does_not_record_autograd_graph(self):
        """Nothing backpropagates through evolve_topology; recording the graph cost ~40x memory."""
        graph, _, _ = _random_graph(n=40, n_edges=80)
        grad_enabled_in_forward = []
        original_forward = graph.forward

        def spy(t, x):
            grad_enabled_in_forward.append(torch.is_grad_enabled() or x.requires_grad)
            return original_forward(t, x)

        graph.forward = spy
        graph.evolve_topology(compute_time=0.3)
        self.assertTrue(grad_enabled_in_forward)
        self.assertFalse(any(grad_enabled_in_forward))

    def test_isolated_nodes_and_empty_graph(self):
        graph = LiquidGraph(hidden_dim=16, resonance_threshold=1.01)
        v = torch.nn.functional.normalize(torch.randn(16), dim=0)
        graph.add_node("solo", v)
        graph.evolve_topology(compute_time=0.5)
        # An isolated node only feels the tether, so its direction is unchanged
        self.assertTrue(torch.allclose(graph.get_node_embedding("solo"), v, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
