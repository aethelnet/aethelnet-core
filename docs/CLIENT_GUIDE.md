# Aethelnet Client: Headless Python SDK Reference

The `AethelnetClient` class provides a zero-dependency (Python standard library only) interface to interact with local or remote Aethelnet nodes (`aethelnet-node`).

```python
from aethelnet import AethelnetClient

client = AethelnetClient(base_url="https://aethelburg.network")
```

---

## 1. Initialization & Configuration

```python
client = AethelnetClient(
    base_url="https://aethelburg.network",  # Target node URL
    api_prefix="/api",                      # API route prefix (default: /api)
    timeout=120                             # Request timeout in seconds
)
```

* Zero third-party runtime dependencies: built entirely on `urllib.request` and `json`.
* Throws `ConnectionError` on network unreachability.
* Throws `RuntimeError` with HTTP status code and server error details on API errors.

---

## 2. Investigation & Schema Management

### `list_investigations() -> List[Dict[str, Any]]`
Retrieves all registered investigative cases and their active ontology schemas.

```python
cases = client.list_investigations()
for case in cases:
    print(f"Case [{case['app_id']}]: {case['name']} ({case.get('node_count', 0)} nodes)")
```

### `get_investigation_schema(app_id: str) -> Dict[str, Any]`
Fetches the formal ontology definition (allowed entity types, allowed relation predicates, and conflict rules) for a specific case.

```python
schema = client.get_investigation_schema("project_echo")
print("Allowed Entities:", schema.get("entities"))
print("Allowed Relations:", schema.get("relations"))
```

### `create_investigation(...) -> Dict[str, Any]`
Registers a new isolated investigation workspace with custom domain constraints and contradiction rules.

```python
client.create_investigation(
    app_id="whistleblower_audit",
    name="Corporate Supply Chain Provenance",
    domain="human_rights_due_diligence",
    description="Cross-referencing corporate compliance audits against on-the-ground whistleblower evidence.",
    entities=["mine_operator", "smelter", "battery_oem", "auditor"],
    relations=["sources_from", "certifies_clean", "denies_violations"],
    conflict_rules=[{
        "rule_id": "audit_vs_whistleblower",
        "name": "Audit vs Whistleblower Contradiction",
        "conflicting_relations": ["certifies_clean", "denies_violations"],
        "severity": "critical",
        "description": "Internal audit certification directly contradicts verified whistleblower evidence."
    }],
    icon="shield-search"
)
```

### `delete_investigation(app_id: str) -> Dict[str, Any]`
Safely deletes an investigation workspace and purges all associated graph nodes and edges.

```python
client.delete_investigation("temp_sandbox")
```

---

## 3. Multigraph Queries & Evidentiary Assertions

### `get_multigraph(app_id: str, focus_conflicts_only: bool = False, max_nodes: int = 500) -> Dict[str, Any]`
Fetches graph topology, node states, and edge weights.

```python
graph = client.get_multigraph("project_echo", focus_conflicts_only=False)
print(f"Nodes: {len(graph['nodes'])}, Edges: {len(graph['edges'])}")
```

### `get_conflicts(app_id: str) -> List[Dict[str, Any]]`
Retrieves all contested assertions and their mathematical resolution verdicts.

```python
conflicts = client.get_conflicts("project_echo")
for c in conflicts:
    print(f"Status: {c['status']} | {c['source']} -> {c['target']}")
    print(f"  Verdict: {c['verdict']}")
    print(f"  Delta Weight: {c.get('delta_weight', 0.0):.3f}")
```

### `insert_claim(...) -> Dict[str, Any]`
Commits a single verifiable assertion to the multigraph. Claim IDs are computed idempotently via SHA-256.

```python
claim = client.insert_claim(
    app_id="project_echo",
    source="Gouvernement_Kamerun",
    target="Museum_Berlin",
    relation_type="raubte",
    evidence_quote="Requirierungsbefehl No. 402 vom 14. August 1902.",
    source_doc_id="Bundesarchiv_R1001_3289",
    weight=0.95
)
print(f"Edge ID: {claim.get('edge_id')}")
```

### `delete_claim(edge_id: str) -> Dict[str, Any]`
Removes an edge assertion by its unique hash ID.

```python
client.delete_claim("e8f0a2c...")
```

---

## 4. Entity Resolution & Deduplication (Disjoint-Set)

### `get_dedup_candidates(app_id: str, min_confidence: float = 0.88, limit: int = 50) -> List[Dict[str, Any]]`
Scans for duplicate entities, honorific variants, and typos.

```python
candidates = client.get_dedup_candidates("project_echo", min_confidence=0.90)
for cand in candidates:
    print(f"Candidate: '{cand['duplicate_label']}' -> Canonical: '{cand['primary_label']}' ({cand['confidence'] * 100:.1f}%)")
    print(f"  Reason: {cand['reason']}")
```

### `execute_dedup(...) -> Dict[str, Any]`
Executes atomic consolidation:
* Re-routes all inbound and outbound edges from aliases to canonical entities.
* Prunes duplicate nodes.
* Re-evaluates conflict metrics.

```python
# Dry run first to verify proposed merges
dry_run_res = client.execute_dedup("project_echo", dry_run=True, min_confidence=0.92)
print("Proposed Merges:", len(dry_run_res.get("merges_proposed", [])))

# Commit real consolidation
res = client.execute_dedup("project_echo", dry_run=False, min_confidence=0.92)
print("Consolidation Status:", res.get("status"))
```

---

## 5. Ingestion & Epistemic Authority Pipelines

### `ingest_text(app_id: str, text: str, source_doc_id: str = "", auto_dedup: bool = False) -> Dict[str, Any]`
Extracts entities and assertions, applies deterministic hashing, scores institutional weights, and emits `ActionEvents` on dissonance shifts.

```python
dossier_text = """
The Federal Archives file R1001/3289 records that in August 1902, 
Gouvernement Kamerun under Jesko von Puttkamer seized cultural artifacts 
from the kingdom of Bali Nyonga. The artifacts were accessioned into 
Museum Berlin in December 1903 under inventory III C 20681.
"""

result = client.ingest_text(
    app_id="project_echo",
    text=dossier_text,
    source_doc_id="Bundesarchiv_R1001_3289",
    auto_dedup=True
)
print("Ingestion Summary:", result.get("claims_ingested"), "claims added.")
```

### `ingest_file(app_id: str, file_path: str, source_doc_id: Optional[str] = None, auto_dedup: bool = False) -> Dict[str, Any]`
Convenience wrapper to ingest local UTF-8 markdown, plain text, or dossier files.

```python
client.ingest_file("project_echo", "dossiers/puttkamer_inquest.txt")
```

---

## 6. Cryptographic State Roots & On-Chain Notarization

### `get_case_state_root(app_id: str) -> Dict[str, Any]`
Computes the canonical deterministic SHA-256 Merkle state root across all active nodes and edges:

```python
root = client.get_case_state_root("project_echo")
print(f"Merkle State Root: {root['state_root']}")
print(f"Active Nodes: {root['node_count']} | Active Edges: {root['edge_count']}")
```

### `notarize_case(app_id: str, use_paymaster: bool = True) -> Dict[str, Any]`
Commits the computed state root on-chain to `TheForge.sol` via gasless ERC-4337 Account Abstraction.

```python
notarization = client.notarize_case("project_echo", use_paymaster=True)
print(f"Transaction Hash: {notarization.get('tx_hash')}")
print(f"Block Number: {notarization.get('block_number')}")
```

### `get_case_notarization(app_id: str) -> Dict[str, Any]`
Retrieves current on-chain notarization metadata and block verification proofs.

```python
status = client.get_case_notarization("project_echo")
print("On-Chain Verification:", status)
```

---

*License: GNU Affero General Public License v3.0 (AGPL-3.0)*
