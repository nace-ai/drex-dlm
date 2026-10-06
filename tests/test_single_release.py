"""Weight-free tests of the published single-decision API and model license."""
import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference
from serve import make_handler
from kev.api import SystemOneRequest, to_record


REQUEST = {"state": "ticket", "questions": {"level": {"type": "score", "criteria": ["Low", "High"]}}}


class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(range(len(text))))


class Scorer:
    def encode(self, tok, record, **kwargs):
        return {"ids": [0, 1, 2], "record": record}

    def probs(self, enc):
        return [torch.tensor([0.25, 0.75]) for _ in enc["record"]["questions"]]


class SingleReleaseTests(unittest.TestCase):
    def test_ordered_legend_and_default_model(self):
        parsed = SystemOneRequest.model_validate(REQUEST)
        self.assertEqual(parsed.model, "drex-dlm")
        self.assertEqual(to_record(parsed)[1][0]["legend"], ["Low", "High"])

    def test_reject_batch_wrapper(self):
        for body in ({"requests": [REQUEST]}, {**REQUEST, "requests": [REQUEST]}):
            with self.assertRaises(ValueError):
                inference.decide(Tokenizer(), Scorer(), body)

    def test_incomplete_and_nonfinite_readouts(self):
        scorer = Scorer()
        for rows in ([], [torch.tensor([float("nan"), 0.0])], [torch.tensor([1.0])]):
            with mock.patch.object(scorer, "probs", return_value=rows):
                with self.assertRaisesRegex(RuntimeError, "readout"):
                    inference.decide(Tokenizer(), scorer, REQUEST)

    def test_http_contract_and_internal_errors(self):
        scorer = Scorer()
        handler = make_handler(Tokenizer(), scorer, "drex-dlm")
        handler.log_message = lambda *args: None
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for body, expected in ((REQUEST, 200), ({"requests": [REQUEST]}, 400),
                                       ({**REQUEST, "state": float("nan")}, 400),
                                       ({"state": "x", "questions": {}}, 400)):
                    with self.subTest(body=body):
                        conn = http.client.HTTPConnection(*server.server_address, timeout=5)
                        try:
                            conn.request("POST", "/v1/systemone", json.dumps(body), {"Content-Type": "application/json"})
                            response = conn.getresponse()
                            self.assertEqual(response.status, expected)
                            response.read()
                        finally:
                            conn.close()
                with mock.patch.object(scorer, "probs", side_effect=ValueError("private detail")):
                    conn = http.client.HTTPConnection(*server.server_address, timeout=5)
                    try:
                        conn.request("POST", "/v1/systemone", json.dumps(REQUEST))
                        response = conn.getresponse()
                        self.assertEqual(response.status, 500)
                        self.assertEqual(json.loads(response.read()), {"error": "internal scoring error"})
                    finally:
                        conn.close()
            finally:
                server.shutdown()
                thread.join()

    def test_model_card_and_notice_license(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "README.md").read_text().startswith("---\nlicense: cc-by-nc-4.0\n"))
        self.assertIn("CC BY-NC 4.0", (root / "MODEL_LICENSE.md").read_text())
        self.assertIn("not MIT", (root / "LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
