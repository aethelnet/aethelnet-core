"""
=============================================================================
AETHELNET HEADLESS PYTHON SDK CLIENT
=============================================================================
Universal headless client providing programmatic access to the Aethelnet
Forensic Multigraph, Entity Resolution Engine, Epistemic Weighting, and
Autonomous Investigation Builder.

Example usage:
    from aethelnet import AethelnetClient

    client = AethelnetClient(base_url="https://aethelburg.network")
    
    # 1. Inspect existing cases & conflicts
    cases = client.list_investigations()
    conflicts = client.get_conflicts("project_echo")
    
    # 2. Check for duplicate entities (Shadow Museum corpus)
    candidates = client.get_dedup_candidates("project_echo", min_confidence=0.90)
    
    # 3. Create a custom forensic case
    client.create_investigation(
        app_id="supply_chain_leak",
        name="Cobalt Supply Chain Provenance",
        domain="human_rights_due_diligence",
        entities=["mine_operator", "smelter", "battery_oem", "trader"],
        relations=["sources_from", "certifies_clean", "denies_child_labor"],
        conflict_rules=[{
            "rule_id": "audit_vs_whistleblower",
            "name": "Audit vs Whistleblower Contradiction",
            "conflicting_relations": ["certifies_clean", "denies_child_labor"],
            "severity": "critical",
            "description": "Internal audit contradicts investigative reporter findings."
        }]
    )
=============================================================================
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional


class AethelnetClient:
    """
    Headless HTTP Client for Aethelnet Causal Nodes & Multigraphs.
    Zero external dependencies (uses standard library urllib).
    """
    def __init__(self, base_url: str = "https://aethelburg.network", api_prefix: str = "/api", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_prefix = api_prefix.rstrip("/")
        self.endpoint = f"{self.base_url}{self.api_prefix}"
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.endpoint}{path}"
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if query:
                url = f"{url}?{query}"

        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Aethelnet-SDK-Python/2.0"
        }

        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("detail", str(e))
            except Exception:
                msg = f"{e}: {err_body}"
            raise RuntimeError(f"Aethelnet API Error [{e.code}] on {method} {path}: {msg}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Could not connect to Aethelnet node at {self.base_url}: {e.reason}")

    # =========================================================================
    # 1. INVESTIGATIONS & SCHEMAS
    # =========================================================================

    def list_investigations(self) -> List[Dict[str, Any]]:
        """Returns all registered forensic investigation cases and their schemas."""
        res = self._request("GET", "/ontology/apps")
        return res.get("apps", [])

    def get_investigation_schema(self, app_id: str) -> Dict[str, Any]:
        """Fetches the formal ontology schema for a specific case."""
        res = self._request("GET", f"/ontology/apps/{app_id}")
        return res.get("app", {})

    def create_investigation(
        self,
        app_id: str,
        name: str,
        domain: str = "forensics",
        entities: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        conflict_rules: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        icon: str = "shield-search"
    ) -> Dict[str, Any]:
        """Registers a custom domain ontology and workspace."""
        payload = {
            "app_id": app_id,
            "name": name,
            "domain": domain,
            "description": description,
            "icon": icon,
            "entities": entities or ["entity", "actor", "artifact"],
            "relations": relations or ["relates_to", "opposes", "claims"],
            "conflict_rules": conflict_rules or []
        }
        return self._request("POST", "/ontology/apps", payload=payload)

    def delete_investigation(self, app_id: str) -> Dict[str, Any]:
        """Safely removes a custom investigation and wipes its multigraph edges."""
        return self._request("DELETE", f"/ontology/apps/{app_id}")

    # =========================================================================
    # 2. MULTIGRAPH & EVIDENCE LEDGER
    # =========================================================================

    def get_multigraph(self, app_id: str, focus_conflicts_only: bool = False, max_nodes: int = 500) -> Dict[str, Any]:
        """Fetches the active multigraph nodes, edges, and conflict verdicts."""
        params = {"focus_conflicts_only": focus_conflicts_only, "max_nodes": max_nodes}
        return self._request("GET", f"/ontology/multigraph/{app_id}", params=params)

    def get_conflicts(self, app_id: str) -> List[Dict[str, Any]]:
        """Fetches all active contested claims and their mathematical resolution verdicts."""
        res = self._request("GET", f"/ontology/multigraph/{app_id}/conflicts")
        return res.get("conflicts", [])

    def insert_claim(
        self,
        app_id: str,
        source: str,
        target: str,
        relation_type: str,
        evidence_quote: str = "",
        source_doc_id: str = "",
        weight: float = 0.55
    ) -> Dict[str, Any]:
        """Commits a verifiable claim / relationship edge to the case ledger."""
        payload = {
            "app_id": app_id,
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "weight": weight,
            "evidence_quote": evidence_quote,
            "source_doc_id": source_doc_id
        }
        return self._request("POST", "/ontology/multigraph/edge", payload=payload)

    def delete_claim(self, edge_id: str) -> Dict[str, Any]:
        """Removes a claim / edge from the multigraph."""
        return self._request("DELETE", f"/ontology/multigraph/edge/{edge_id}")

    def clear_canvas(self, app_id: str) -> Dict[str, Any]:
        """Purges all edges and nodes for an investigation for a clean slate."""
        return self._request("DELETE", f"/ontology/multigraph/{app_id}")

    # =========================================================================
    # 3. ENTITY RESOLUTION & DEDUPLICATION (SHADOW MUSEUM TESTBED)
    # =========================================================================

    def get_dedup_candidates(self, app_id: str, min_confidence: float = 0.88, limit: int = 50) -> List[Dict[str, Any]]:
        """Scans for potential duplicate entities (aliases, honorific variants, typos)."""
        params = {"threshold": min_confidence, "limit": limit}
        res = self._request("GET", f"/ontology/multigraph/{app_id}/dedup_candidates", params=params)
        return res.get("candidates", [])

    def execute_dedup(
        self,
        app_id: str,
        dry_run: bool = True,
        min_confidence: float = 0.90,
        merges: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes atomic batch node merging with disjoint-set clustering:
        - Reroutes inbound and outbound edges from duplicates to canonical primary.
        - Prunes duplicate nodes.
        - Automatically re-evaluates conflicts and recalculates epistemic weights.
        """
        payload = {
            "dry_run": dry_run,
            "min_confidence": min_confidence,
            "merges": merges
        }
        return self._request("POST", f"/ontology/multigraph/{app_id}/dedup_batch_execute", payload=payload)

    # =========================================================================
    # 4. EPISTEMIC EDGE WEIGHTING
    # =========================================================================

    def recalculate_weights(self, app_id: str) -> Dict[str, Any]:
        """Triggers whole-graph epistemic weight recalculation and dissonance resolution."""
        return self._request("POST", f"/ontology/multigraph/{app_id}/weights_recalculate")

    def ingest_text(
        self,
        app_id: str,
        text: str,
        source_doc_id: str = "",
        auto_dedup: bool = False
    ) -> Dict[str, Any]:
        """
        Zero-Shot Structured Ingestion & Reactive Pipeline Hook:
        Extracts entities and relations, applies deterministic idempotent claim IDs,
        evaluates epistemic authority weights, and emits ActionEvents on dissonance shifts.
        """
        payload = {
            "app_id": app_id,
            "text": text,
            "source_doc_id": source_doc_id,
            "auto_dedup": auto_dedup
        }
        return self._request("POST", "/ontology/ingest/text", payload=payload)

    def ingest_file(
        self,
        app_id: str,
        file_path: str,
        source_doc_id: Optional[str] = None,
        auto_dedup: bool = False
    ) -> Dict[str, Any]:
        """
        Ingests a local text, markdown, or JSON dossier file into the multigraph.
        """
        import os
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dossier file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        doc_id = source_doc_id or os.path.basename(file_path)
        return self.ingest_text(
            app_id=app_id,
            text=content,
            source_doc_id=doc_id,
            auto_dedup=auto_dedup
        )

    def get_case_state_root(self, app_id: str) -> Dict[str, Any]:
        """
        Computes the canonical deterministic SHA-256 state root for a case multigraph.
        """
        return self._request("GET", f"/ontology/multigraph/{app_id}/state_root")

    def notarize_case(self, app_id: str, use_paymaster: bool = True) -> Dict[str, Any]:
        """
        Notarizes the case multigraph state root on-chain to TheForge smart contract
        via gasless Account Abstraction (ERC-4337).
        """
        return self._request("POST", f"/ontology/multigraph/{app_id}/notarize?use_paymaster={str(use_paymaster).lower()}")

    def get_case_notarization(self, app_id: str) -> Dict[str, Any]:
        """
        Retrieves the on-chain notarization status and cryptographic verification for a case.
        """
        return self._request("GET", f"/ontology/multigraph/{app_id}/notarization")


