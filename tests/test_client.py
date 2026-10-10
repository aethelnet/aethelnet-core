import json
import os
import unittest
from unittest.mock import patch, MagicMock

from aethelnet.client import AethelnetClient


def _fake_response(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestClientAuth(unittest.TestCase):
    def _sent_request(self, client: AethelnetClient, call):
        with patch("urllib.request.urlopen", return_value=_fake_response({"apps": []})) as urlopen:
            call(client)
        return urlopen.call_args[0][0]

    def test_token_argument_sent_as_bearer(self):
        client = AethelnetClient(base_url="http://localhost:8000", token="abc123")
        req = self._sent_request(client, lambda c: c.list_investigations())
        self.assertEqual(req.get_header("Authorization"), "Bearer abc123")

    def test_token_from_environment(self):
        with patch.dict(os.environ, {"AETHELNET_AUTH_TOKEN": "from-env"}):
            client = AethelnetClient(base_url="http://localhost:8000")
        req = self._sent_request(client, lambda c: c.delete_claim("e1"))
        self.assertEqual(req.get_header("Authorization"), "Bearer from-env")
        self.assertEqual(req.get_method(), "DELETE")

    def test_no_token_no_header(self):
        with patch.dict(os.environ, {"AETHELNET_AUTH_TOKEN": ""}):
            client = AethelnetClient(base_url="http://localhost:8000")
        req = self._sent_request(client, lambda c: c.list_investigations())
        self.assertIsNone(req.get_header("Authorization"))

    def test_query_params_encoded(self):
        client = AethelnetClient(base_url="http://localhost:8000", token="")
        req = self._sent_request(client, lambda c: c.get_dedup_candidates("case_a", min_confidence=0.9, limit=5))
        self.assertIn("min_confidence=0.9", req.full_url)
        self.assertIn("/ontology/multigraph/case_a/dedup_candidates", req.full_url)


if __name__ == "__main__":
    unittest.main()
