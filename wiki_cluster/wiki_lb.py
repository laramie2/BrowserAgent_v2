#!/usr/bin/env python3
import argparse
import http.client
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class BackendPool:
    def __init__(self, ports):
        self._ports = list(ports)
        self._active = {port: 0 for port in self._ports}
        self._cursor = 0
        self._lock = threading.Lock()

    def ordered_candidates(self):
        with self._lock:
            indexed = []
            total = len(self._ports)
            for offset in range(total):
                idx = (self._cursor + offset) % total
                port = self._ports[idx]
                indexed.append((self._active[port], offset, idx, port))
            indexed.sort()
            self._cursor = (self._cursor + 1) % total
            return [port for _, _, _, port in indexed]

    def acquire(self, port):
        with self._lock:
            self._active[port] += 1

    def release(self, port):
        with self._lock:
            self._active[port] = max(0, self._active[port] - 1)

    def snapshot(self):
        with self._lock:
            return dict(self._active)


class WikiProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "wiki-native-lb/1.0"

    def do_HEAD(self):
        self._proxy()

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def _proxy(self):
        if self.path == "/_wiki_cluster_status":
            body = "\n".join(
                f"127.0.0.1:{port} active={active}"
                for port, active in sorted(self.server.pool.snapshot().items())
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return

        last_error = None
        for port in self.server.pool.ordered_candidates():
            self.server.pool.acquire(port)
            try:
                self._proxy_to_backend(port)
                return
            except Exception as exc:
                last_error = exc
                self.log_error("backend 127.0.0.1:%s failed: %s", port, exc)
            finally:
                self.server.pool.release(port)

        message = f"All wiki backends failed: {last_error}\n".encode("utf-8")
        self.send_response(502)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(message)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(message)

    def _proxy_to_backend(self, port):
        body = None
        content_length = self.headers.get("Content-Length")
        if content_length:
            body = self.rfile.read(int(content_length))

        headers = {}
        for key, value in self.headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                headers[key] = value
        headers["Host"] = f"127.0.0.1:{port}"

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()

            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.end_headers()

            if self.command != "HEAD":
                while True:
                    chunk = response.read(1024 * 64)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        finally:
            conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Small local load balancer for multiple kiwix-serve processes.")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=22015)
    parser.add_argument("--backends", required=True, help="Comma-separated backend ports, for example: 22115,22116")
    return parser.parse_args()


def main():
    args = parse_args()
    ports = [int(port.strip()) for port in args.backends.split(",") if port.strip()]
    if not ports:
        raise SystemExit("No backend ports configured.")

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), WikiProxyHandler)
    server.pool = BackendPool(ports)

    def shutdown(signum, frame):
        del signum, frame
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"wiki native load balancer listening on {args.listen_host}:{args.listen_port}", flush=True)
    print("backends: " + ", ".join(f"127.0.0.1:{port}" for port in ports), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
