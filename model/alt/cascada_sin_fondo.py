"""R10 E: cascada L2 (kNN-e5 con SOLO 8 de oro por clase, sin fondo) → L3 (4B por logprobs) en las 6 tareas de desarrollo de v2.

    python -m model.alt.cascada_sin_fondo medir --dev DIR_DEV --vec DIR_VEC --l3 DIR_PRED_ZS [--l3-extra tarea=fichero] --salida informe.json
    python -m model.alt.cascada_sin_fondo criterio

L2 = `escalado.l2` con banco = el soporte de oro de cada tarea (split train del .jsonl de dev, 8 por clase) y fondo vacío.
L3 = el 4B zero-shot por logprobs de T3 (pred/4b-zs/<tarea>.jsonl); en finanzas-documentos, el de T3b con preselección top-k
(pred/zs-topk), que cubre el 100 %. Señal de escalado: el margen (p1 − p2) de L2. Partición fija por hash del id (`auditoria.mitad`):
en A se elige el umbral u (cuantil global del margen) que da la mejor macro de la cascada con un escalado medio por tarea ≤ 50 %; en B
se mide. Métricas: macro de las 6 tareas (media de exactitudes), escalado medio por tarea, AUROC del margen (media por tarea).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CRIT_E = {"l2": "kNN-e5 solo oro (8 por clase), 32 vecinos, T 0.05, sin fondo", "l3": "4B zs logprobs (T3); finanzas: zs-topk (T3b)",
          "senal": "margen", "particion": "auditoria.mitad (hash del id): A selecciona u, B mide",
          "seleccion": "u = cuantil global del margen en A con escalado medio por tarea <= 0.50 y mejor macro de la cascada",
          "pasa_perdida_max": 0.01, "pasa_escalado_max": 0.50, "pasa_auroc_min": 0.70, "macro": "media simple de las 6 tareas dev"}
CRIT_E_SHA = hashlib.sha256(json.dumps(CRIT_E, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
FRACS = [round(0.05 * i, 2) for i in range(0, 11)]


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cargar(dev, vec, l3dir, extra):
    from model.alt import escalado as es
    from model.alt.llm_techo import top1
    from model.alt.p0_medir import cargar_bateria
    out = {}
    for c, (sop, ev) in sorted(cargar_bateria(dev).items()):
        z = np.load(Path(vec) / (c + ".npz"))
        if list(z["idte"]) != [r["id"] for r in ev]:
            raise ValueError("%s: vectores y filas no coinciden" % c)
        ruta = extra.get(c) or str(Path(l3dir) / (c + ".jsonl"))
        res = {r["id"]: r for r in leer(ruta)}
        ops = [r["opciones"] for r in ev]
        P, sen = es.l2(z["Xte"], ops, z["Xtr"], [r["etiqueta"] for r in sop], np.zeros((0, z["Xte"].shape[1])), [])
        ok2 = np.array([1.0 if ops[i][int(np.argmax(P[i, :len(ops[i])]))] == ev[i]["etiqueta"] else 0.0 for i in range(len(ev))])
        ok3, cub = [], 0
        for r in ev:
            x = res.get(r["id"])
            t = top1(x["probs"], r["opciones"], r.get("tipo", "choice")) if x and x.get("valida") and x.get("probs") else None
            cub += x is not None
            ok3.append(1.0 if t == r["etiqueta"] else 0.0)
        out[c] = {"ids": [r["id"] for r in ev], "ok2": ok2, "ok3": np.array(ok3), "margen": sen["margen"], "cobertura_l3": cub / len(ev)}
    return out


def medir(t):
    from model.alt import escalado as es
    from model.alt.auditoria import mitad
    A = {c: np.array([k for k, i in enumerate(d["ids"]) if mitad(i) == 0], int) for c, d in t.items()}
    B = {c: np.array([k for k, i in enumerate(d["ids"]) if mitad(i) == 1], int) for c, d in t.items()}

    def casc(P, u):
        acc, esc = {}, {}
        for c, d in t.items():
            i = P[c]
            e = d["margen"][i] < u
            acc[c] = float(np.where(e, d["ok3"][i], d["ok2"][i]).mean())
            esc[c] = float(e.mean())
        return acc, esc

    todos = np.concatenate([t[c]["margen"][A[c]] for c in t])
    cand = []
    for f in FRACS:
        u = float(np.quantile(todos, f)) if f > 0 else -np.inf
        acc, esc = casc(A, u)
        if np.mean(list(esc.values())) <= CRIT_E["pasa_escalado_max"] + 1e-9:
            cand.append((float(np.mean(list(acc.values()))), -f, f, u))
    _, _, f_el, u = max(cand)
    accB, escB = casc(B, u)
    l3B = {c: float(t[c]["ok3"][B[c]].mean()) for c in t}
    l2B = {c: float(t[c]["ok2"][B[c]].mean()) for c in t}
    au = {c: es.auroc(t[c]["margen"][B[c]], t[c]["ok2"][B[c]]) for c in t}
    m = lambda d: round(float(np.mean(list(d.values()))), 4)  # noqa: E731
    v = {"criterio_sha256": CRIT_E_SHA, "f": f_el, "u": round(u, 6), "cascada": m(accB), "l3": m(l3B), "l2": m(l2B),
         "cascada_menos_l3": round(m(accB) - m(l3B), 4), "escalado_medio": m(escB),
         "auroc_margen_medio": round(float(np.mean([x for x in au.values() if x is not None])), 4)}
    v["estado"] = "PASA" if (v["cascada_menos_l3"] >= -CRIT_E["pasa_perdida_max"] - 1e-9 and v["escalado_medio"] <= CRIT_E["pasa_escalado_max"] + 1e-9
                             and v["auroc_margen_medio"] >= CRIT_E["pasa_auroc_min"]) else "NO PASA"
    curva = []
    for f in FRACS + [0.6, 0.7, 0.8, 0.9, 1.0]:
        uu = float(np.quantile(todos, f)) if 0 < f < 1 else (-np.inf if f == 0 else np.inf)
        a, e = casc(B, uu)
        curva.append({"f": f, "cascada_B": m(a), "escalado_B": m(e)})
    return {"veredicto": v, "por_tarea_B": {c: {"cascada": round(accB[c], 4), "l2": round(l2B[c], 4), "l3": round(l3B[c], 4), "escalado": round(escB[c], 4),
                                               "auroc": (None if au[c] is None else round(au[c], 4)), "cobertura_l3": round(t[c]["cobertura_l3"], 4)} for c in t},
            "curva_B": curva, "n_A": {c: int(len(A[c])) for c in A}, "n_B": {c: int(len(B[c])) for c in B}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("medir")
    for k in ("--dev", "--vec", "--l3", "--salida"):
        m.add_argument(k, required=True)
    m.add_argument("--l3-extra", action="append", default=[], help="tarea=fichero (p. ej. finanzas con preselección)")
    sp.add_parser("criterio")
    a = ap.parse_args(argv)
    if a.cmd == "criterio":
        print(CRIT_E_SHA, json.dumps(CRIT_E, sort_keys=True, ensure_ascii=False))
        return
    t = cargar(a.dev, a.vec, a.l3, dict(x.split("=", 1) for x in a.l3_extra))
    out = {"criterio": CRIT_E, "criterio_sha256": CRIT_E_SHA, **medir(t)}
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["veredicto"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
