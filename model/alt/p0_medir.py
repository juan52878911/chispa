"""P0/P2: predicciones de L2 (M1), del cross-encoder del hito 4 (PyTorch fp32, ONNX fp32 e int8) y de kNN-e5 sobre JevBench y la batería
no vista; y el informe pareado (exactitud + IC95 bootstrap, McNemar exacto).

    python -m model.alt.p0_medir pred --modelo l2|ce|ce-fp32-onnx|ce-int8-onnx|knn --bateria DIR [--jevbench FILAS.jsonl] --salida DIR
                                      [--m1 m1.pt] [--ckpt DIR_CE] [--onnx F.onnx] [--hilos 8] [--max-filas N]
    python -m model.alt.p0_medir informe --pred DIR --bateria DIR --jevbench FILAS.jsonl --salida informe.json [--llm l3.jsonl]

`pred` escribe DIR/<modelo>/<conjunto>.jsonl con {id, probs}; el conjunto de JevBench se llama `jevbench`. La exactitud es la del
top-1 de `jevbench_informe.top1` (en score, la mediana). IC95: bootstrap percentil de 2000 remuestreos, semilla 0 (`model.l2.largos.ic95`).
McNemar: prueba exacta binomial bilateral sobre los pares discordantes. kNN-e5: k ejemplos por clase del soporte (split train de la
batería), 32 vecinos, temperatura 0,05, vectores de e5 con media de tokens (`linea_base_knn_e5`).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path


def leer(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def escribir(ruta, preds):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps({"id": p["id"], "probs": p["probs"]}, ensure_ascii=False) + "\n")


def cargar_bateria(directorio, max_filas=None):
    """{conjunto: (soporte, evaluación)}"""
    out = {}
    for f in sorted(Path(directorio).glob("*.jsonl")):
        rs = leer(f)
        ev = [r for r in rs if r["split"] == "test"][:max_filas]
        out[f.stem] = ([r for r in rs if r["split"] == "train"], ev)
    return out


# ---------------------------------------------------------------- predictores
def pred_l2(m1, conjuntos, hilos):
    import torch
    torch.set_num_threads(hilos)
    from model.l2.evaluar import cargar_modelo, predecir
    from model.l2.modelo import MAX_TOK, Codificador
    modelo, ck = cargar_modelo(m1)
    enc = Codificador(device="cpu", max_tok=ck.get("max_tok", MAX_TOK))   # como model.evaluacion.jevbench_l2
    return {c: [{"id": p["id"], "probs": p["probs"]} for p in predecir(modelo, enc, ev, lote=64)] for c, (_, ev) in conjuntos.items()}


def pred_ce(ckpt, conjuntos, hilos, l2x=False, dispositivo="cpu"):
    import torch
    torch.set_num_threads(hilos)
    from .evaluar import predecir
    from .l2x import L2x
    from .modelo import CrossEncoder
    m = (L2x if l2x else CrossEncoder).cargar(ckpt, dispositivo)
    return {c: predecir(m, ev, lote=16) for c, (_, ev) in conjuntos.items()}


def pred_onnx(onnx, ckpt, conjuntos, hilos, l2x=False):
    if l2x:
        from .l2x_onnx import OrtL2x
        m = OrtL2x(onnx, ckpt, hilos)
    else:
        from .modelo import trozos
        from .onnx_ce import OrtCE
        m = OrtCE(onnx, ckpt, hilos)
        m.partir = lambda fs, mt: trozos(fs, mt, m.largo)
    out = {}
    for c, (_, ev) in conjuntos.items():
        ps = []
        for i in range(0, len(ev), 16):
            for fs in m.partir(ev[i:i + 16], 8192):
                ps += [{"id": r["id"], "probs": p} for r, p in zip(fs, m.probs(fs))]
        out[c] = ps
    return out


def pred_knn(conjuntos, hilos):
    import torch
    torch.set_num_threads(hilos)
    from model.l2.linea_base_knn_e5 import probs_knn, vectores
    from model.l2.modelo import Codificador
    enc = Codificador(device="cpu", max_tok=256)   # como linea_base_knn_e5
    out = {}
    for c, (sop, ev) in conjuntos.items():
        if c == "jevbench" or not sop:
            continue
        X = vectores(enc, [r["texto"] for r in sop])
        Q = vectores(enc, [r["texto"] for r in ev])
        ys = [r["etiqueta"] for r in sop]
        out[c] = [{"id": r["id"], "probs": probs_knn(q, X, ys, r["opciones"])} for q, r in zip(Q, ev)]
    return out


# ---------------------------------------------------------------- estadística
def top1(r, probs):
    from model.evaluacion.jevbench_informe import top1 as t
    return t(probs, r["opciones"], r["tipo"])


def aciertos(filas, preds):
    por = {p["id"]: p["probs"] for p in preds}
    return [1.0 if r["id"] in por and top1(r, por[r["id"]]) == r["etiqueta"] else 0.0 for r in filas]


def macro_f1(filas, preds):
    por = {p["id"]: p["probs"] for p in preds}
    ys = [r["etiqueta"] for r in filas]
    yp = [top1(r, por[r["id"]]) if r["id"] in por else None for r in filas]
    f = []
    for c in sorted(set(ys)):
        tp = sum(1 for a, b in zip(ys, yp) if a == c and b == c)
        fp = sum(1 for a, b in zip(ys, yp) if a != c and b == c)
        fn = sum(1 for a, b in zip(ys, yp) if a == c and b != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / len(f)


def mcnemar(a, b):
    """(solo a acierta, solo b acierta, p exacto bilateral)"""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def _ic95(ok, remuestreos=2000, semilla=0):
    """(media, bajo, alto): bootstrap percentil sobre casos; el mismo algoritmo que `model.l2.largos.ic95` (sin importar torch)."""
    import numpy as np
    ok = np.asarray(ok, dtype=float)
    medias = ok[np.random.RandomState(semilla).randint(0, len(ok), (remuestreos, len(ok)))].mean(1)
    return float(ok.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def ic(x):
    m, lo, hi = _ic95(x)
    return {"acc": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def ic_dif(a, b):
    m, lo, hi = _ic95([x - y for x, y in zip(a, b)])
    return {"dif": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def leer_llm(ruta, filas, permisivo):
    """Predicciones del LLM de `llm_jevbench.py`: estricta (respuesta inválida = fallo, como L3) o permisiva (probs parciales
    renormalizadas, ids ausentes = 0)."""
    out, invalidas = [], 0
    for r, l in zip(filas, [json.loads(x) for x in open(ruta, encoding="utf-8") if x.strip()]):
        assert r["id"] == l["id"]
        probs = l["probs"]
        if probs is None and permisivo:
            try:
                i, j = l["raw"].find("{"), l["raw"].rfind("}")
                pr = json.loads(l["raw"][i:j + 1])["probs"]
                pr = {k: float(v) for k, v in pr.items() if k in r["opciones"] and isinstance(v, (int, float)) and v >= 0}
                s = sum(pr.values())
                probs = {o: pr.get(o, 0.0) / s for o in r["opciones"]} if s > 0 else None
            except (ValueError, KeyError, AttributeError, TypeError):
                probs = None
        if probs is None:
            invalidas += 1
            probs = {o: 0.0 for o in r["opciones"]}
            probs["__sin_respuesta__"] = 1.0
        out.append({"id": r["id"], "probs": probs})
    return out, invalidas


def informe(a):
    pred = Path(a.pred)
    jev = leer(a.jevbench)
    bat = cargar_bateria(a.bateria)
    filas = {c: ev for c, (_, ev) in bat.items()}
    filas["jevbench"] = jev
    modelos = sorted(d.name for d in pred.iterdir() if d.is_dir())
    res = {"modelos": modelos, "conjuntos": {}, "pareado": {}}
    P = {m: {c: leer(pred / m / (c + ".jsonl")) for c in filas if (pred / m / (c + ".jsonl")).exists()} for m in modelos}
    A = {m: {c: aciertos(filas[c], P[m][c]) for c in P[m]} for m in modelos}
    for m in modelos:
        res["conjuntos"][m] = {}
        for c in P[m]:
            res["conjuntos"][m][c] = dict(n=len(filas[c]), **ic(A[m][c]), macro_f1=round(macro_f1(filas[c], P[m][c]), 4))
        bc = [c for c in P[m] if c != "jevbench"]
        if bc:
            todo = [x for c in bc for x in A[m][c]]
            res["conjuntos"][m]["BATERIA(agrupada)"] = dict(n=len(todo), **ic(todo))
            res["conjuntos"][m]["BATERIA(macro de tareas)"] = {"acc": round(sum(res["conjuntos"][m][c]["acc"] for c in bc) / len(bc), 4),
                                                              "macro_f1": round(sum(res["conjuntos"][m][c]["macro_f1"] for c in bc) / len(bc), 4), "tareas": len(bc)}
    if a.llm:
        for nombre, perm in (("llm-estricto", False), ("llm-permisivo", True)):
            ps, inval = leer_llm(a.llm, jev, perm)
            A[nombre] = {"jevbench": aciertos(jev, ps)}
            res["conjuntos"][nombre] = {"jevbench": dict(n=len(jev), invalidas=inval, **ic(A[nombre]["jevbench"]), macro_f1=round(macro_f1(jev, ps), 4))}
    pares = [tuple(x.split(":")) for x in a.pares]
    for x, y in pares:
        d = {}
        cs = sorted(set(A.get(x, {})) & set(A.get(y, {})))
        for c in cs:
            n01, n10, p = mcnemar(A[x][c], A[y][c])
            d[c] = dict(n=len(A[x][c]), **{"solo_" + x: n01, "solo_" + y: n10}, p_mcnemar=round(p, 4), **ic_dif(A[x][c], A[y][c]))
        bc = [c for c in cs if c != "jevbench"]
        if bc:
            ax, ay = [v for c in bc for v in A[x][c]], [v for c in bc for v in A[y][c]]
            n01, n10, p = mcnemar(ax, ay)
            d["BATERIA(agrupada)"] = dict(n=len(ax), **{"solo_" + x: n01, "solo_" + y: n10}, p_mcnemar=round(p, 6), **ic_dif(ax, ay))
        res["pareado"]["%s - %s" % (x, y)] = d
    Path(a.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pred")
    p.add_argument("--modelo", required=True, choices=["l2", "ce", "ce-fp32-onnx", "ce-int8-onnx", "knn", "l2x", "l2x-fp32-onnx", "l2x-int8-onnx"])
    p.add_argument("--bateria", default="")
    p.add_argument("--jevbench")
    p.add_argument("--salida", required=True)
    p.add_argument("--m1")
    p.add_argument("--ckpt")
    p.add_argument("--onnx")
    p.add_argument("--hilos", type=int, default=8)
    p.add_argument("--max-filas", type=int)
    p.add_argument("--dispositivo", default="cpu")
    p.add_argument("--sin-bateria", action="store_true")
    i = sub.add_parser("informe")
    i.add_argument("--pred", required=True)
    i.add_argument("--bateria", required=True)
    i.add_argument("--jevbench", required=True)
    i.add_argument("--salida", required=True)
    i.add_argument("--llm")
    i.add_argument("--pares", nargs="*", default=["l2:ce", "ce-fp32-onnx:ce-int8-onnx", "ce:ce-int8-onnx"])
    a = ap.parse_args(argv)
    if a.cmd == "informe":
        return informe(a) or 0
    conj = {} if a.sin_bateria else cargar_bateria(a.bateria, a.max_filas)
    if a.jevbench:
        if Path(a.jevbench).is_dir():   # clon de JevBench: se convierte con el mismo cargador del arnés
            from model.evaluacion.jevbench import cargar_jevbench
            conj["jevbench"] = ([], cargar_jevbench(a.jevbench)[:a.max_filas])
        else:
            conj["jevbench"] = ([], leer(a.jevbench)[:a.max_filas])
    t0 = time.time()
    if a.modelo == "l2":
        out = pred_l2(a.m1, conj, a.hilos)
    elif a.modelo in ("ce", "l2x"):
        out = pred_ce(a.ckpt, conj, a.hilos, l2x=a.modelo == "l2x", dispositivo=a.dispositivo)
    elif a.modelo == "knn":
        out = pred_knn(conj, a.hilos)
    else:
        out = pred_onnx(a.onnx, a.ckpt, conj, a.hilos, l2x=a.modelo.startswith("l2x"))
    for c, ps in out.items():
        escribir(Path(a.salida) / a.modelo / (c + ".jsonl"), ps)
        print(a.modelo, c, len(ps), flush=True)
    print("segundos", round(time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
