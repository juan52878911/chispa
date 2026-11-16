"""Servidor OpenAI falso de logprobs para la prueba en seco de r10_t3.sh (CHISPA_LOCAL=1): sin GPU ni modelo.

    python3 r10_t3_vllm_falso.py PUERTO

GET /v1/models y POST /v1/chat/completions: lee las etiquetas del último mensaje de usuario y devuelve un top 20 determinista (hash del contenido)
con las primeras etiquetas; con `prompt_logprobs` devuelve el logprob de la etiqueta final. Solo biblioteca estándar.
"""
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def responder(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.responder({"data": [{"id": "falso"}]})

    def do_POST(self):
        c = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = c["messages"]
        ultimo = next(m for m in reversed(msgs) if m["role"] == "user")["content"]
        etq = [o["etiqueta"] for o in json.loads(ultimo)["opciones"]]
        h = int(hashlib.sha256(ultimo.encode()).hexdigest(), 16)
        if "prompt_logprobs" in c:
            cod = msgs[-1]["content"]
            return self.responder({"choices": [{"message": {"content": cod}}], "prompt_logprobs": [None, {"1": {"logprob": -3.0, "decoded_token": cod}}]})
        top = [{"token": e, "logprob": -0.1 - ((h >> (3 * i)) % 50) / 10.0} for i, e in enumerate(etq[:12])]
        self.responder({"choices": [{"message": {"content": top[0]["token"]}, "logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "top_logprobs": top}]}}]})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
