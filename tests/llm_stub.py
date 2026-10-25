"""A tiny fake Ollama server for tests. It answers from a script, on 127.0.0.1."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def ok_reply(text: str, done_reason: str = "stop"):
    body = {"message": {"role": "assistant", "content": text}, "done": True,
            "done_reason": done_reason}
    return (200, body, 0)


class StubOllama:
    """`script` is a list of (status, body, delay_seconds). The last entry repeats forever."""

    def __init__(self, script=None, tags=()):
        self.script = list(script or [ok_reply("ok")])
        self.tags = list(tags)
        self.chat_requests: list[dict] = []
        self.url = ""
        self._server = None

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up (timeout test)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": n} for n in outer.tags]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.chat_requests.append(json.loads(self.rfile.read(length) or b"{}"))
                script = outer.script
                status, body, delay = script.pop(0) if len(script) > 1 else script[0]
                if delay:
                    time.sleep(delay)
                self._send(status, body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()