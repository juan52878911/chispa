"""vLLM falso para la prueba en seco de R9-e2e (solo biblioteca estándar): /v1/models y /v1/chat/completions con logprobs y prompt_logprobs.

    python3 r9_vllm_falso.py --puerto 8765 [--ms 5]

Llamada principal (logprobs): devuelve un top 20 con las primeras 5 etiquetas de la fila (la favorita, elegida por hash del texto, también
con su variante " X") y deja fuera el resto, para que el servidor Go use el relleno por prompt_logprobs. Llamada de relleno (prompt_logprobs):
devuelve el logprob de la etiqueta con la que termina la conversación. No es un modelo: solo prueba la mecánica y los formatos.
"""
import argparse
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
CUENTA = {"principal": 0, "relleno": 0}
CERROJO = threading.Lock()


def etiquetas(msgs):
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    return re.findall(r'"etiqueta": "(.)"', u), u


def logprob(u, e):
    h = int(hashlib.sha256((u + e).encode()).hexdigest()[:8], 16)
    return -0.05 - (h % 4000) / 1000.0   # entre -0,05 y -4,05


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, cod=200):
        b = json.dumps(obj).encode()
        self.send_response(cod)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json({"data": [{"id": "qwen3-4b"}]})

    def do_POST(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        time.sleep(ARGS.ms / 1000)
        msgs = cuerpo["messages"]
        etq, u = etiquetas(msgs)
        if "prompt_logprobs" in cuerpo:
            with CERROJO:
                CUENTA["relleno"] += 1
            cod = msgs[-1]["content"]
            return self._json({"choices": [{"message": {"content": ""}}], "prompt_logprobs": [
                None, {"1": {"logprob": -0.1, "decoded_token": "<|im_start|>"}}, {"2": {"logprob": logprob(u, cod) - 2, "decoded_token": cod}}]})
        with CERROJO:
            CUENTA["principal"] += 1
        top = []
        for e in sorted(etq, key=lambda e: logprob(u, e), reverse=True)[:5]:
            top.append({"token": e, "logprob": logprob(u, e)})
        if top:
            top.append({"token": " " + top[0]["token"], "logprob": top[0]["logprob"] - 1.0})
        top.append({"token": "Respuesta", "logprob": -6.0})
        self._json({"choices": [{"message": {"content": top[0]["token"]}, "logprobs": {"content": [{"token": top[0]["token"], "top_logprobs": top}]}}]})


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--ms", type=float, default=5)
    ARGS = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", ARGS.puerto), H).serve_forever()


if __name__ == "__main__":
    main()
