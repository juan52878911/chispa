"""Producto: genera el banco kNN de UNA tarea para el servidor Go desde un jsonl de ejemplos de oro (y, opcional, un fondo etiquetado por el LLM).

    python3 -m model.alt.generar_banco --tarea NOMBRE --oro oro.jsonl --salida DIR [--fondo fondo.jsonl] [--k 8] [--semilla 20261115] [--cache DIR_CACHE_HUB] [--lote 64]

Entradas (jsonl, una fila por línea):
  --oro     ejemplos de oro de la tarea: {"id", "pregunta", "tipo", "texto", "opciones": [ids], "etiqueta": id correcto, "descripciones": {id: texto} (opcional)}.
            «Trae entre 8 y 32 ejemplos por clase»: es todo lo que el cliente aporta. Una fila con "split" distinto de "train" no cuenta como oro.
  --fondo   OPCIONAL: textos con la etiqueta que les puso el LLM ({"id"?, "texto", "etiqueta"}); entran al banco detrás del oro y votan en el kNN
            (con el oro a 0,90 de similitud o más, manda solo el oro). Sin él, el banco es solo oro. Para etiquetar textos sin etiqueta, pase
            `model/alt/llm_logprobs.py` por ellos y guarde el top-1 de cada fila como `etiqueta`.
Salida en DIR (lo que lee `knn.tareas.<t>.banco` del servidor):
  banco-<tarea>.npy             vectores e5 float32 (n × 384), oro primero y fondo después, normalizados;
  banco-<tarea>.jsonl           {"etiqueta", "es_oro"} por fila, en el mismo orden;
  banco-<tarea>.ejemplos.jsonl  los k ejemplos de oro del few-shot (fs8) de L3, elegidos con `llm_logprobs.elegir_ejemplos` (estratificados por clase, semilla fija):
                                el servidor los lee tal cual, de modo que el prompt es el de `llm_logprobs.py --ejemplos --k K` byte a byte;
  banco-<tarea>.meta.json       tamaños, e5 y versiones.
Los vectores son los del servidor: «query: » + texto (normalización de mayúsculas incluida), e5 sin ajustar, 256 tokens, media de tokens con máscara y norma L2,
con el `Codificador` y `vectores` de los bancos de R6/R8 (torch 2.5.1 y transformers 4.46.3, fijados: otras versiones mueven los vectores hasta 0,06).
Si la tarea no tiene oro (--oro vacío) se escribe un banco vacío: el servidor escala SIEMPRE a L3 (zero-shot, motivo sin_oro)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TORCH, TRANSFORMERS = "2.5.1", "4.46.3"


def leer(ruta):
    return [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()] if ruta else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tarea", required=True)
    ap.add_argument("--oro", required=True)
    ap.add_argument("--fondo")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--k", type=int, default=8, help="nº de ejemplos del few-shot de L3 (0 = sin fichero de ejemplos)")
    ap.add_argument("--semilla", type=int, default=20261115)
    ap.add_argument("--cache", help="caché del hub de HF (HF_HUB_CACHE) con models--intfloat--multilingual-e5-small; trabaja fuera de línea")
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--permitir-versiones", action="store_true", help="no exige torch/transformers fijados")
    a = ap.parse_args(argv)
    import os
    if a.cache:
        os.environ["HF_HUB_CACHE"] = a.cache
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    import transformers
    if not a.permitir_versiones and (not torch.__version__.startswith(TORCH) or transformers.__version__ != TRANSFORMERS):
        sys.exit("torch %s y transformers %s van fijados (los vectores de los bancos): hay %s y %s; --permitir-versiones para seguir" % (TORCH, TRANSFORMERS, torch.__version__, transformers.__version__))
    from model.alt import llm_logprobs as lp
    from model.l2.linea_base_knn_e5 import vectores
    from model.l2.modelo import Codificador
    oro = [r for r in leer(a.oro) if r.get("split", "train") == "train"]
    fondo = leer(a.fondo)
    ops = {tuple(r["opciones"]) for r in oro}
    for r in oro:
        if r["etiqueta"] not in r["opciones"]:
            sys.exit("%s: la etiqueta %r no es una de sus opciones" % (r.get("id"), r["etiqueta"]))
    textos = [r["texto"] for r in oro] + [r["texto"] for r in fondo]
    enc = Codificador(device="cpu", max_tok=256)
    X = vectores(enc, textos, lote=a.lote).numpy().astype(np.float32) if textos else np.zeros((0, 384), np.float32)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    base = sal / ("banco-%s" % a.tarea)
    np.save(str(base) + ".npy", X)
    with open(str(base) + ".jsonl", "w", encoding="utf-8") as f:
        for r in oro:
            f.write(json.dumps({"etiqueta": r["etiqueta"], "es_oro": True}, ensure_ascii=False) + "\n")
        for r in fondo:
            f.write(json.dumps({"etiqueta": r["etiqueta"], "es_oro": False}, ensure_ascii=False) + "\n")
    n_ej = 0
    if a.k and oro:
        ejs = lp.elegir_ejemplos(oro, a.k, a.semilla)
        cols = ("id", "pregunta", "tipo", "texto", "opciones", "etiqueta")
        with open(str(base) + ".ejemplos.jsonl", "w", encoding="utf-8") as f:
            for e in ejs:
                f.write(json.dumps({k: e[k] for k in cols}, ensure_ascii=False) + "\n")
        n_ej = len(ejs)
    meta = {"tarea": a.tarea, "n_oro": len(oro), "n_fondo": len(fondo), "dim": int(X.shape[1]), "ejemplos_fs": n_ej, "semilla": a.semilla,
            "clases_oro": {c: sum(1 for r in oro if r["etiqueta"] == c) for c in sorted({r["etiqueta"] for r in oro})},
            "e5": "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3", "torch": torch.__version__, "transformers": transformers.__version__,
            "opciones_distintas": len(ops)}
    (sal / ("banco-%s.meta.json" % a.tarea)).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
