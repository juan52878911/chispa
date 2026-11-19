"""vLLM falso para la prueba en seco de W1-fs8 (w1_fs8_local.sh): como r10_t3_vllm_falso.py, pero el top 20 sale del hash de TODA la conversación (sistema, demostraciones y caso),
no solo del último mensaje: cualquier diferencia entre el prompt del servidor Go y el de llm_logprobs.py (demostraciones, orden, formato) cambia las probabilidades y se ve en la medida.

    python3 w1_vllm_falso.py PUERTO
"""
import hashlib
import json
import sys
from http.server import ThreadingHTTPServer

import r10_t3_vllm_falso as base


class H(base.H):
    def do_POST(self):
        c = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = c["messages"]
        ultimo = next(m for m in reversed(msgs) if m["role"] == "user")["content"]
        etq = [o["etiqueta"] for o in json.loads(ultimo)["opciones"]]
        h = int(hashlib.sha256(json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), 16)
        if "prompt_logprobs" in c:
            cod = msgs[-1]["content"]
            return self.responder({"choices": [{"message": {"content": cod}}], "prompt_logprobs": [None, {"1": {"logprob": -3.0, "decoded_token": cod}}]})
        top = [{"token": e, "logprob": -0.1 - ((h >> (3 * i)) % 50) / 10.0} for i, e in enumerate(etq[:12])]
        self.responder({"choices": [{"message": {"content": top[0]["token"]}, "logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "top_logprobs": top}]}}]})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
