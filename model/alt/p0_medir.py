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


def leer_jev(ruta):
    """Filas de JevBench: un .jsonl ya convertido o el clon (datasets/public), que se convierte con el cargador del arnés."""
    if Path(ruta).is_dir():
        from model.evaluacion.jevbench import cargar_jevbench
        return cargar_jevbench(ruta)
    return leer(ruta)


def escribir(ruta, preds):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps({"id": p["id"], "probs": p["probs"]}, ensure_ascii=False) + "\n")


def cargar_bateria(directorio, max_filas=None):
    """{conjunto: (soporte, evaluación)}"""
    out = {}
    for f in sorted(Path(directorio).glob("*.jsonl")):
        if f.name.startswith("."):  # ._* de macOS (AppleDouble) que GNU tar saca de un tarball hecho en el Mac
            continue
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
    jev = leer_jev(a.jevbench)
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


# ---------------------------------------------------------------- P3: informe de L2x contra la batería
def auroc(conf, ok):
    """Área bajo la curva ROC de `conf` para separar aciertos (1) de fallos (0); empates con rango medio. None si falta una clase."""
    import numpy as np
    conf, ok = np.asarray(conf, dtype=float), np.asarray(ok, dtype=int)
    n1, n0 = int(ok.sum()), int((1 - ok).sum())
    if n1 == 0 or n0 == 0:
        return None
    orden = np.argsort(conf, kind="stable")
    rangos = np.empty(len(conf))
    c = conf[orden]
    i = 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[i]:
            j += 1
        rangos[orden[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((rangos[ok == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def base_top1(a):
    """Guarda para cada modelo de `--pred` {id: [top1, probabilidad máxima]}: lo que necesita `informe-l2x` para comparar sin las predicciones enteras."""
    pred = Path(a.pred)
    filas = {c: ev for c, (_, ev) in cargar_bateria(a.bateria).items()}
    filas["jevbench"] = leer_jev(a.jevbench)
    out = {}
    for d in sorted(x for x in pred.iterdir() if x.is_dir()):
        out[d.name] = {}
        for c, fs in filas.items():
            f = d / (c + ".jsonl")
            if not f.exists():
                continue
            por = {p["id"]: p["probs"] for p in leer(f)}
            for r in fs:
                pr = por.get(r["id"])
                if pr is not None:
                    out[d.name][r["id"]] = [top1(r, pr), round(max(pr.values()), 4)]
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in out.items()})


def informe_l2x(a):
    """Informe de paso de P3: la batería es la vara (agrupada y por tarea), hatecheck-es aparte (la única tarea de verdad no vista),
    más los criterios de latencia y AUROC de la revisión. `--pred` es la carpeta con `<conjunto>.jsonl` de L2x."""
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    bat = cargar_bateria(a.bateria)
    filas = {c: ev for c, (_, ev) in bat.items()}
    filas["jevbench"] = leer_jev(a.jevbench)
    P = {c: leer(Path(a.pred) / (c + ".jsonl")) for c in filas if (Path(a.pred) / (c + ".jsonl")).exists()}
    modelos = {"l2x": None, "l2": base["l2"], "cruzado": base["ce-fp32-onnx"]}
    if "knn" in base:
        modelos["knn-e5"] = base["knn"]

    def top(m, r):
        if m == "l2x":
            pr = {p["id"]: p["probs"] for p in P[c]}.get(r["id"])
            return (top1(r, pr), max(pr.values())) if pr else (None, 0.0)
        v = modelos[m].get(r["id"])
        return (v[0], v[1]) if v else (None, 0.0)

    A, F, C = {}, {}, {}
    for c in P:
        for m in modelos:
            if m == "knn-e5" and c == "jevbench":
                continue
            t = [top(m, r) for r in filas[c]]
            A[(m, c)] = [1.0 if x[0] == r["etiqueta"] else 0.0 for x, r in zip(t, filas[c])]
            C[(m, c)] = [x[1] for x in t]
            F[(m, c)] = macro_f1(filas[c], [{"id": r["id"], "probs": {x[0]: 1.0}} for x, r in zip(t, filas[c]) if x[0] is not None])
    tareas = [c for c in P if c != "jevbench"]
    res = {"tareas": {}, "agrupada": {}, "media_tareas": {}, "hatecheck-es": {}, "mcnemar": {}, "criterios": {}}
    for c in P:
        res["tareas"][c] = {m: dict(n=len(A[(m, c)]), **ic(A[(m, c)]), macro_f1=round(F[(m, c)], 4)) for m in modelos if (m, c) in A}
    for m in modelos:
        todo = [v for c in tareas for v in A[(m, c)]]
        res["agrupada"][m] = dict(n=len(todo), **ic(todo))
        res["media_tareas"][m] = {"acc": round(sum(res["tareas"][c][m]["acc"] for c in tareas) / len(tareas), 4),
                                  "macro_f1": round(sum(res["tareas"][c][m]["macro_f1"] for c in tareas) / len(tareas), 4)}
    if "hatecheck-es" in P:
        res["hatecheck-es"] = res["tareas"]["hatecheck-es"]
    for m in ("l2", "cruzado"):
        d = {}
        for c in tareas + ["jevbench"]:
            if c in P:
                n01, n10, p = mcnemar(A[("l2x", c)], A[(m, c)])
                d[c] = dict(solo_l2x=n01, **{"solo_" + m: n10}, p_mcnemar=round(p, 4), **ic_dif(A[("l2x", c)], A[(m, c)]))
        ax, am = [v for c in tareas for v in A[("l2x", c)]], [v for c in tareas for v in A[(m, c)]]
        n01, n10, p = mcnemar(ax, am)
        d["AGRUPADA"] = dict(solo_l2x=n01, **{"solo_" + m: n10}, p_mcnemar=round(p, 6), **ic_dif(ax, am))
        res["mcnemar"]["l2x - " + m] = d
    conf = [v for c in tareas for v in C[("l2x", c)]]
    oks = [v for c in tareas for v in A[("l2x", c)]]
    res["auroc_confianza"] = {"agrupada": auroc(conf, oks), "por_tarea": {c: auroc(C[("l2x", c)], A[("l2x", c)]) for c in tareas},
                              "nota": "AUROC de la probabilidad máxima para separar aciertos de fallos; no hay cabeza de escalada aprendida"}
    mt = res["media_tareas"]
    por_debajo = [c for c in tareas if res["tareas"][c]["l2x"]["acc"] < res["tareas"][c]["cruzado"]["acc"] - 0.01]
    cr = {
        "C1 macro-F1 medio (6 tareas) >= L2 + 12 pts": [mt["l2x"]["macro_f1"], round(mt["l2"]["macro_f1"] + 0.12, 4)],
        "C2 macro-F1 medio >= cruzado (hito 4) - 1 pt": [mt["l2x"]["macro_f1"], round(mt["cruzado"]["macro_f1"] - 0.01, 4)],
        "C3 exactitud agrupada >= cruzado - 1 pt": [res["agrupada"]["l2x"]["acc"], round(res["agrupada"]["cruzado"]["acc"] - 0.01, 4)],
        "C4 AUROC de la confianza >= 0,78": [res["auroc_confianza"]["agrupada"], 0.78],
    }
    res["criterios"] = {k: {"valor": v[0], "umbral": v[1], "pasa": bool(v[0] is not None and v[0] >= v[1])} for k, v in cr.items()}
    res["criterios"]["C5 latencia p50 <= 60 ms en el i7 (int8, 4 hilos, <= 8 opciones)"] = {"valor": None, "umbral": 60, "pasa": None, "nota": "se mide aparte con bench_l2x_i7.py sobre l2x_int8.onnx"}
    res["refuta"] = {"tareas_por_debajo_del_cruzado": por_debajo, "refutado": len(por_debajo) >= 3,
                     "regla": "por debajo del cruzado en exactitud (-1 pt) en >= 3 de las 6 tareas"}
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "informe-l2x.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md = ["# L2x frente a la batería no vista", "", "| conjunto | n | " + " | ".join(modelos) + " |", "|---|---|" + "---|" * len(modelos)]
    for c in P:
        md.append("| %s%s | %d | %s |" % (c, " (única tarea de verdad no vista)" if c == "hatecheck-es" else "", res["tareas"][c]["l2x"]["n"],
                                           " | ".join(("%.3f [%.3f–%.3f]" % (res["tareas"][c][m]["acc"], res["tareas"][c][m]["lo"], res["tareas"][c][m]["hi"])) if m in res["tareas"][c] else "–" for m in modelos)))
    md.append("| **agrupada (6 tareas)** | %d | %s |" % (res["agrupada"]["l2x"]["n"], " | ".join("%.3f [%.3f–%.3f]" % (res["agrupada"][m]["acc"], res["agrupada"][m]["lo"], res["agrupada"][m]["hi"]) for m in modelos)))
    md.append("| **macro-F1 medio** | | %s |" % " | ".join("%.3f" % mt[m]["macro_f1"] for m in modelos))
    md += ["", "## McNemar pareado (L2x frente a cada uno)", ""]
    for k, d in res["mcnemar"].items():
        for c, v in d.items():
            md.append("- %s, %s: diferencia %+.1f pts [%+.1f; %+.1f], p = %s" % (k, c, 100 * v["dif"], 100 * v["lo"], 100 * v["hi"], v["p_mcnemar"]))
    md += ["", "## Criterios", ""]
    for k, v in res["criterios"].items():
        md.append("- %s: %s (valor %s, umbral %s)" % (k, {True: "PASA", False: "NO PASA", None: "PENDIENTE"}[v["pasa"]], v["valor"], v["umbral"]))
    md.append("- Refutación: %s (%s)" % ("SÍ" if res["refuta"]["refutado"] else "no", ", ".join(res["refuta"]["tareas_por_debajo_del_cruzado"]) or "ninguna tarea por debajo"))
    (sal / "informe-l2x.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


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
    b = sub.add_parser("base-top1")
    b.add_argument("--pred", required=True)
    b.add_argument("--bateria", required=True)
    b.add_argument("--jevbench", required=True)
    b.add_argument("--salida", required=True)
    x = sub.add_parser("informe-l2x")
    x.add_argument("--pred", required=True, help="carpeta con <conjunto>.jsonl de L2x")
    x.add_argument("--bateria", required=True)
    x.add_argument("--jevbench", required=True)
    x.add_argument("--base", required=True, help="baseline-top1.json")
    x.add_argument("--salida", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "base-top1":
        return base_top1(a) or 0
    if a.cmd == "informe-l2x":
        return informe_l2x(a) or 0
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
