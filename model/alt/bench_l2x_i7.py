"""Banco de latencia de L2x (y del cross-encoder, para comparar) para el i7: solo numpy + onnxruntime, sin torch ni el repo.

    taskset -c 0-3 python3 bench_l2x_i7.py --onnx l2x_fp32.onnx l2x_int8.onnx [ce_int8.onnx] [--hilos 4] [--opciones 2 4 8 16]
                                           [--n 200] [--texto 128] [--tok-opcion 8] [--tok-pregunta 16] --salida bench.json

Por cada modelo y cada número de opciones K mide p50, p99 y media (ms) de UNA petición con ids sintéticos (el coste no depende del contenido):
  - L2x (el grafo tiene `opcion_mask`): una secuencia de 4 + pregunta + K·(1 + tok_opcion) + texto tokens, y la variante `total128`, con la
    secuencia entera de 128 tokens (texto = lo que sobre).
  - cruzado (solo `input_ids` y `attention_mask`): un lote de K secuencias de `--texto` tokens, una por opción (peor caso, sin relleno).
Escribe la CPU, los hilos y la afinidad (para saber si el i7 estaba libre) junto con las cifras. Pasa si p50 ≤ 60 ms con ≤ 4 opciones.
"""
import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import onnxruntime as ort


def cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            nombre = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        nombre = platform.processor()
    try:
        carga = os.getloadavg()
    except OSError:
        carga = None
    return {"cpu": nombre, "nucleos_visibles": os.cpu_count(), "afinidad": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "carga_1_5_15": carga, "onnxruntime": ort.__version__}


def sesion(ruta, hilos):
    o = ort.SessionOptions()
    o.intra_op_num_threads = hilos
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(ruta, o, providers=["CPUExecutionProvider"])


def entradas_l2x(rng, k, total, tok_op, tok_preg, vocab=250000):
    """ids/máscaras de una secuencia con `k` opciones; si `total` es None el texto mide lo que pida quien llama (se pasa en `total`)."""
    ids = rng.randint(5, vocab, (1, total)).astype(np.int64)
    om = np.zeros((1, k, total), dtype=np.float32)
    p = 2 + tok_preg
    for j in range(k):
        om[0, j, p:p + 1 + tok_op] = 1.0
        p += 1 + tok_op
    tm = np.zeros((1, total), dtype=np.float32)
    tm[0, p + 1:total - 1] = 1.0
    return {"input_ids": ids, "attention_mask": np.ones_like(ids), "opcion_mask": om, "texto_mask": tm}


def medir(s, f, n, calentar=10):
    ms = []
    for i in range(calentar + n):
        t0 = time.perf_counter()
        s.run(None, f())
        if i >= calentar:
            ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return {"p50_ms": round(statistics.median(ms), 2), "p99_ms": round(ms[min(len(ms) - 1, int(round(0.99 * (len(ms) - 1))))], 2), "media_ms": round(statistics.mean(ms), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--opciones", type=int, nargs="*", default=[2, 4, 8, 16])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--texto", type=int, default=128)
    ap.add_argument("--tok-opcion", type=int, default=8)
    ap.add_argument("--tok-pregunta", type=int, default=16)
    ap.add_argument("--salida")
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    res = {"entorno": cpu(), "hilos": a.hilos, "n": a.n, "texto": a.texto, "tok_opcion": a.tok_opcion, "tok_pregunta": a.tok_pregunta, "modelos": {}}
    for ruta in a.onnx:
        s = sesion(ruta, a.hilos)
        nombres = {i.name for i in s.get_inputs()}
        r = {}
        for k in a.opciones:
            if "opcion_mask" in nombres:
                tot = 4 + a.tok_pregunta + k * (1 + a.tok_opcion) + a.texto
                r["texto%d/%d" % (a.texto, k)] = dict(medir(s, lambda: entradas_l2x(rng, k, tot, a.tok_opcion, a.tok_pregunta), a.n), tokens=tot)
                if 128 > 4 + a.tok_pregunta + k * (1 + a.tok_opcion) + 8:
                    r["total128/%d" % k] = dict(medir(s, lambda: entradas_l2x(rng, k, 128, a.tok_opcion, a.tok_pregunta), a.n), tokens=128)
            else:
                def f():
                    ids = rng.randint(5, 250000, (k, a.texto)).astype(np.int64)
                    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}
                r["lote%dx%d" % (k, a.texto)] = dict(medir(s, f, a.n), tokens=k * a.texto)
        res["modelos"][os.path.basename(ruta)] = r
        print(os.path.basename(ruta), json.dumps(r))
    if a.salida:
        with open(a.salida, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
            fh.write("\n")


if __name__ == "__main__":
    main()
