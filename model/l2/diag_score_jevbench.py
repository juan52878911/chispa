"""Diagnóstico de la cabeza de score sobre JevBench: ¿el signo de la similitud importa y dónde cae la mediana CORAL?

    python -m model.l2.diag_score_jevbench --ckpt mejor.pt --jevbench RUTA --salida DIR [--device cpu] [--max-casos N]

Sobre los casos `score` de JevBench (k = 0, sin soporte), con la cabeza del checkpoint (`coral1`, `coral`, `acumulados` o `grueso_fino`):
  (a) histograma de la etiqueta predicha (la mediana CORAL: nº de umbrales con P(y>j) > 0,5), junto al del oro, por número de niveles K;
  (b) exactitud con la similitud `s` tal cual y con `−s` (se invierte el signo de lo que entra a la cabeza). Si `−s` acierta más que `s`,
      la cabeza está orientada al revés; si ambas dan lo mismo que predecir siempre un nivel, la similitud no informa.
Referencia: exactitud de predecir siempre la moda del oro. Escribe `DIR/diag_score_jevbench.json` y `.md`. (Este fichero no sustituye a
`diag_score.py`, que diagnostica el entrenamiento con tasksource.)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from model.evaluacion.jevbench_informe import top1


@torch.no_grad()
def predecir(modelo, enc, filas, signo=1.0, lote=64):
    """{id: probs} del modelo sobre `filas` score con la entrada de la cabeza multiplicada por `signo`."""
    cabezas = list(modelo.coral.values())
    originales = [h.forward for h in cabezas]
    try:
        if signo != 1.0:
            for h, f in zip(cabezas, originales):
                h.forward = lambda x, f=f: f(signo * x)
        out = {}
        for i in range(0, len(filas), lote):
            fs = filas[i:i + lote]
            probs, _ = modelo(fs, enc, entrenar=False)
            for r, p in zip(fs, probs):
                p = p.double().cpu()
                out[r["id"]] = dict(zip(r["opciones"], (p / p.sum()).tolist()))
        return out
    finally:
        for h, f in zip(cabezas, originales):
            h.forward = f


def resumir(filas, preds_s, preds_menos):
    """Cifras por K y total a partir de las predicciones con s y con −s."""
    res = {"n": len(filas), "por_k": {}}
    por_k = {}
    for r in filas:
        por_k.setdefault(len(r["opciones"]), []).append(r)
    for k, rs in sorted(por_k.items()):
        oro = Counter(int(r["etiqueta"]) for r in rs)
        d = {"n": len(rs), "oro": {str(n): oro.get(n, 0) for n in range(1, k + 1)}, "moda_oro_acc": max(oro.values()) / len(rs)}
        for nombre, preds in (("s", preds_s), ("menos_s", preds_menos)):
            pred = [top1(preds[r["id"]], r["opciones"], "score") for r in rs]
            hist = Counter(int(p) for p in pred if p is not None)
            d[nombre] = {"acc": sum(p == r["etiqueta"] for p, r in zip(pred, rs)) / len(rs),
                         "mediana_predicha": {str(n): hist.get(n, 0) for n in range(1, k + 1)}}
        res["por_k"][str(k)] = d
    res["acc_s"] = sum(d["s"]["acc"] * d["n"] for d in res["por_k"].values()) / max(len(filas), 1)
    res["acc_menos_s"] = sum(d["menos_s"]["acc"] * d["n"] for d in res["por_k"].values()) / max(len(filas), 1)
    res["moda_oro_acc"] = sum(d["moda_oro_acc"] * d["n"] for d in res["por_k"].values()) / max(len(filas), 1)
    return res


def informe_md(res, cabeza):
    L = ["# Diagnóstico de score en JevBench (cabeza %s)" % cabeza, "",
         "%d casos score. Exactitud con `s`: **%.3f**; con `−s`: **%.3f**; predecir siempre la moda del oro: %.3f." % (res["n"], res["acc_s"], res["acc_menos_s"], res["moda_oro_acc"]), ""]
    for k, d in res["por_k"].items():
        niv = list(d["oro"])
        L += ["## K = %s (%d casos)" % (k, d["n"]), "", "| nivel | " + " | ".join(niv) + " |", "|---|" + "---|" * len(niv),
              "| oro | " + " | ".join(str(d["oro"][n]) for n in niv) + " |",
              "| mediana predicha con s | " + " | ".join(str(d["s"]["mediana_predicha"][n]) for n in niv) + " |",
              "| mediana predicha con −s | " + " | ".join(str(d["menos_s"]["mediana_predicha"][n]) for n in niv) + " |", "",
              "exactitud: s %.3f, −s %.3f, moda del oro %.3f" % (d["s"]["acc"], d["menos_s"]["acc"], d["moda_oro_acc"]), ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tok", type=int, default=0, help="0 = el del checkpoint")
    ap.add_argument("--max-casos", type=int, default=0)
    a = ap.parse_args(argv)
    from model.evaluacion.jevbench import cargar_jevbench
    from .evaluar import cargar_modelo
    from .modelo import MAX_TOK, Codificador
    filas = [r for r in cargar_jevbench(a.jevbench) if r["tipo"] == "score"]
    if a.max_casos:
        filas = filas[:a.max_casos]
    modelo, ck = cargar_modelo(a.ckpt, a.device)
    enc = Codificador(device=a.device, max_tok=a.max_tok or ck.get("max_tok", MAX_TOK))
    enc.cargar_enc(ck.get("enc"))
    res = resumir(filas, predecir(modelo, enc, filas), predecir(modelo, enc, filas, -1.0))
    res["cabeza_score"] = modelo.cabeza_score
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "diag_score_jevbench.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    md = informe_md(res, modelo.cabeza_score)
    (sal / "diag_score_jevbench.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
