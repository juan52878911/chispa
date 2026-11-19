"""W1: bancos FABRICADOS para la prueba en seco del ud (sin ONNX de Qwen3-Embedding ni GPU): a partir de `soporte.jsonl` de w1_filas, escribe por tarea `banco-<tarea>.npy`
(vectores inventados de 8 dimensiones, normalizados, deterministas), `.jsonl`, `.ejemplos.jsonl` (con `llm_logprobs.elegir_ejemplos`, que es lo que reproduce chispa-banco-qwen en Go)
y `bancos.json`. Solo sirve para probar que las tuberías empalman; las cifras de verdad salen de los bancos de chispa-banco-qwen.

    python3 -m model.alt.w1_banco_falso --soporte ent/soporte.jsonl --salida DIR [--k 8]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from model.alt import llm_logprobs as lp

COLS = ("id", "pregunta", "tipo", "texto", "opciones", "etiqueta")


def vector(texto, dim=8):
    h = hashlib.sha256(texto.encode("utf-8")).digest()
    v = np.frombuffer((h * ((dim * 4) // len(h) + 1))[:dim * 4], dtype=np.uint8).astype(np.float32)[:dim] - 127.5
    return (v / np.linalg.norm(v)).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--soporte", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args(argv)
    por = {}
    for l in open(a.soporte, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            por.setdefault(r["tarea"], []).append(r)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for t, rs in sorted(por.items()):
        base = str(sal / ("banco-" + t))
        np.save(base + ".npy", np.stack([vector(r["texto"]) for r in rs]))
        with open(base + ".jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({"etiqueta": r["etiqueta"], "es_oro": True}, ensure_ascii=False) + "\n")
        ejs = lp.elegir_ejemplos(rs, a.k, 20261115)
        with open(base + ".ejemplos.jsonl", "w", encoding="utf-8") as f:
            for e in ejs:
                f.write(json.dumps({c: e[c] for c in COLS}, ensure_ascii=False) + "\n")
        resumen[t] = {"filas": len(rs), "dim": 8, "ejemplos_fs": len(ejs)}
    (sal / "bancos.json").write_text(json.dumps(resumen, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(resumen))


if __name__ == "__main__":
    main()
