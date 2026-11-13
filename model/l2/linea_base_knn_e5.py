"""Línea base de las ablaciones de L2: e5 con media de tokens (sin ranuras ni torre) y kNN-32 sobre el MISMO soporte que L2 con k.

    python -m model.l2.linea_base_knn_e5 --jevbench RUTA --salida DIR [--k 8 32] [--vecinos 32] [--temp 0.05] [--device cpu]

Para cada pliegue de JevBench (mismo `pliegues` que `model.evaluacion.jevbench_l2`: soporte de k por opción, sin el caso ni sus
paráfrasis) vota entre los `vecinos` más cercanos del soporte por coseno de los vectores de e5 (media de tokens con máscara de la
última capa, normalizada, con el prefijo «query: » como los mensajes de L2), con pesos exp(sim/temp). Escribe
`DIR/pred/knn_e5_k<k>.jsonl` con `{id, probs, ms, soporte_n}`, el mismo formato que L2, y la exactitud global por k.
Un caso sin soporte (o sin soporte de alguna opción) reparte la probabilidad que no tiene soporte de forma uniforme: no se inventa una clase.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.evaluacion import chispa_eval as CE
from model.evaluacion.jevbench import cargar_jevbench, pliegues

VECINOS = 32
TEMP = 0.05


@torch.no_grad()
def vectores(enc, textos, lote=64):
    """(n,384) normalizados: media de la última capa de e5 sobre los tokens de «query: » + texto."""
    out = []
    for i in range(0, len(textos), lote):
        H, m = enc.tokens(textos[i:i + lote])
        mf = m.unsqueeze(-1).float().to(H.device)
        out.append(F.normalize((H * mf).sum(1) / mf.sum(1).clamp(min=1), dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 384)


def probs_knn(q, X, ys, opciones, vecinos=VECINOS, temp=TEMP):
    """q (d,), X (n,d) normalizados, ys (n,) etiquetas del soporte → {opción: p}. Sin soporte: uniforme."""
    p = dict.fromkeys(opciones, 0.0)
    if len(ys):
        s = (X @ q).numpy()
        idx = np.argsort(-s, kind="stable")[:vecinos]
        w = np.exp((s[idx] - s[idx].max()) / temp)
        for i, wi in zip(idx, w):
            if ys[i] in p:
                p[ys[i]] += float(wi)
    tot = sum(p.values())
    if tot <= 0:
        return {o: 1.0 / len(opciones) for o in opciones}
    return {o: v / tot for o, v in p.items()}


def predecir_knn(enc, filas, k, vecinos=VECINOS, temp=TEMP):
    """[{id, probs, ms, soporte_n}] en el orden de `filas` (que traen `esquema`, `grupo`, `etiqueta`, `texto`)."""
    por_id = {}
    for _, rs, sop in pliegues(filas, k):
        t0 = time.perf_counter()
        X = vectores(enc, [r["texto"] for r in sop])
        Q = vectores(enc, [r["texto"] for r in rs])
        ys = [r["etiqueta"] for r in sop]
        ms = (time.perf_counter() - t0) * 1000 / len(rs)
        for q, r in zip(Q, rs):
            por_id[r["id"]] = {"id": r["id"], "probs": probs_knn(q, X, ys, r["opciones"], vecinos, temp), "ms": round(ms, 3), "soporte_n": len(sop)}
    return [por_id[r["id"]] for r in filas]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--k", type=int, nargs="*", default=[8, 32])
    ap.add_argument("--vecinos", type=int, default=VECINOS)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--max-tok", type=int, default=256)
    ap.add_argument("--max-casos", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    from model.l2.modelo import Codificador
    filas = cargar_jevbench(a.jevbench)
    if a.max_casos:
        filas = filas[:a.max_casos]
    enc = Codificador(device=a.device, max_tok=a.max_tok)
    sal = Path(a.salida)
    (sal / "pred").mkdir(parents=True, exist_ok=True)
    for k in a.k:
        preds = predecir_knn(enc, filas, k, a.vecinos, a.temp)
        with open(sal / "pred" / ("knn_e5_k%d.jsonl" % k), "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        res = CE.evaluar(filas, [{x: p[x] for x in ("id", "probs", "ms")} for p in preds])
        print("== kNN-%d e5 (soporte k=%d): exactitud %.4f, macro-F1 %.4f" % (a.vecinos, k, res["global"]["exactitud"], res["global"]["macro_f1"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
