"""Servidor vLLM falso para probar techo_local en seco (sin GPU ni modelo): chat y completions (harmony), top 20 determinista por hash del caso.

    python3 vllm_falso.py PUERTO [MS_POR_PETICION]

GET /v1/models; POST /v1/chat/completions (como r10_t3_vllm_falso) y POST /v1/completions (prompt harmony: lee el último mensaje de usuario). Con `prompt_logprobs` devuelve
el logprob de la etiqueta final. Solo biblioteca estándar.
"""
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def top_de(ultimo, etq):
    h = int(hashlib.sha256(ultimo.encode()).hexdigest(), 16)
    return [{"token": e, "logprob": -0.1 - ((h >> (3 * i)) % 50) / 10.0} for i, e in enumerate(etq[:12])]


class H(BaseHTTPRequestHandler):
    ms = 0

    def log_message(self, *a):
        pass

    def responder(self, obj, codigo=200):
        b = json.dumps(obj).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.responder({"data": [{"id": "falso"}]})

    def do_POST(self):
        c = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.ms:
            time.sleep(self.ms / 1000)
        if self.path.endswith("/v1/completions"):
            return self.completions(c)
        msgs = c["messages"]
        ultimo = next(m for m in reversed(msgs) if m["role"] == "user")["content"]
        etq = [o["etiqueta"] for o in json.loads(ultimo)["opciones"]]
        if "prompt_logprobs" in c:
            cod = msgs[-1]["content"]
            return self.responder({"choices": [{"message": {"content": cod}}], "prompt_logprobs": [None, {"1": {"logprob": -3.0, "decoded_token": cod}}]})
        top = top_de(ultimo, etq)
        self.responder({"choices": [{"message": {"content": top[0]["token"]}, "logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "top_logprobs": top}]}}]})

    def completions(self, c):
        p = c["prompt"]
        if "<|start|>developer<|message|>" not in p or ("prompt_logprobs" not in c and not p.endswith("<|channel|>final<|message|>")):
            return self.responder({"error": "prompt harmony inesperado"}, 400)
        ultimo = p.split("<|start|>user<|message|>")[-1].split("<|end|>")[0]
        etq = [o["etiqueta"] for o in json.loads(ultimo)["opciones"]]
        if "prompt_logprobs" in c:
            cod = p.rsplit("<|message|>", 1)[-1]
            return self.responder({"choices": [{"text": "", "prompt_logprobs": [None, {"1": {"logprob": -3.0, "decoded_token": cod}}]}]})
        top = top_de(ultimo, etq)
        self.responder({"choices": [{"text": top[0]["token"], "logprobs": {"tokens": [top[0]["token"]], "token_logprobs": [top[0]["logprob"]], "top_logprobs": [{t["token"]: t["logprob"] for t in top}]}}]})


def servir(puerto, ms=0):
    H.ms = ms
    return ThreadingHTTPServer(("127.0.0.1", puerto), H)


if __name__ == "__main__":
    servir(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0).serve_forever()
