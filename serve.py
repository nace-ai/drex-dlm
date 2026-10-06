"""Local System One server for Drex DLM.

    python serve.py --port 8000

POST /v1/systemone accepts ``model``, ``state``, and named ``choice``, ``noul``,
or ``score`` questions. This process scores the pointer head in this directory.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from inference import decide, load


def reject_nonfinite_json(value):
    raise ValueError(f"non-finite JSON number: {value}")


def make_handler(tok, model, model_name):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print("%s %s" % (self.address_string(), fmt % args))

        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"status": "ok", "model": model_name})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.close_connection = True
                self._send(400, {"error": "Content-Length must be an integer"})
                return
            if length <= 0 or length > 1_000_000:
                self.close_connection = True
                self._send(400, {"error": "request body must be between 1 byte and 1 MB"})
                return
            try:
                request = json.loads(self.rfile.read(length), parse_constant=reject_nonfinite_json)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                if "model" in request and not isinstance(request["model"], str):
                    raise ValueError("model must be a string")
                with lock:
                    result = decide(tok, model, request)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:
                self.log_error("System One scoring failed: %r", exc)
                self._send(500, {"error": "internal scoring error"})
                return
            self._send(200, {"model": request.get("model") or model_name, **result})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve Drex DLM at POST /v1/systemone.")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", default="drex-dlm")
    args = parser.parse_args()
    tok, model = load(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tok, model, args.name))
    print(f"Drex DLM listening on http://{args.host}:{args.port}/v1/systemone")
    server.serve_forever()


if __name__ == "__main__":
    main()
