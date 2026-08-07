"""Serve the annotation viewer over HTTP with server-side verdict storage.

``polke adjudicate <annotations>`` builds the viewer page and serves it, so
adjudication can happen from another machine (typically through an SSH
tunnel). Every verdict the viewer records is PUT back to the server and
written to ``verdicts.json`` next to the records — the browser's
localStorage is only a cache, and ``polke score`` reads the file directly.

Endpoints:
    GET  /               the viewer page (built once at startup)
    GET  /verdicts.json  the stored verdicts ({"verdicts": {}} when empty)
    PUT  /verdicts.json  replace the stored verdicts (atomic write)
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_server(html: str, verdicts_file: Path,
                host: str, port: int) -> ThreadingHTTPServer:
    page = html.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes,
                  ctype: str = "application/json; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path in ("/", "/view.html", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/verdicts.json":
                if verdicts_file.exists():
                    self._send(200, verdicts_file.read_bytes())
                else:
                    self._send(200, b'{"verdicts": {}}')
            else:
                self._send(404, b'{"error": "not found"}')

        def do_PUT(self):  # noqa: N802
            if self.path.split("?", 1)[0] != "/verdicts.json":
                self._send(404, b'{"error": "not found"}')
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(n).decode("utf-8"))
                verdicts = data.get("verdicts")
                if not isinstance(verdicts, dict):
                    raise ValueError('body must be {"verdicts": {...}}')
            except (ValueError, UnicodeDecodeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode())
                return
            tmp = verdicts_file.with_name(verdicts_file.name + ".tmp")
            tmp.write_text(
                json.dumps({"tool": "polke adjudicate", "verdicts": verdicts},
                           ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8")
            tmp.replace(verdicts_file)          # atomic on POSIX
            self._send(200, b'{"ok": true}')

        do_POST = do_PUT

        def log_message(self, fmt, *args):      # quiet server
            pass

    return ThreadingHTTPServer((host, port), Handler)
