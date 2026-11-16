"""T5: referencia de Python para la cascada L2(kNN)→L3 del servidor Go. Exporta, por tarea, el banco (oro ∪ fondo con etiqueta del LLM,
logprobs donde la haya) y, para las filas de la mitad B de la partición por hash de §17.1 b, el vector de consulta y la decisión de Python
fila a fila. No mide nada nuevo: es el fixture de paridad.

    python -m model.alt.exportar_banco --bateria DIR[,DIR2] --vec DIR_VEC --fondo f1,f2 --l3-fondo a,b,c --l3 x,y --salida DIR

Regla (la de §17.1 b, congelada allí): L2 = kNN de `escalado.l2` (32 vecinos, T 0,05, umbral de oro 0,90); se escala si el margen < u,
con u = cuantil 0,25 del margen en la mitad A sin hatecheck-es; y SIEMPRE si la tarea no tiene oro o si el AUROC del margen en la mitad A
es < 0,55. Salida en DIR: `banco-<tarea>.npy` (X float32), `banco-<tarea>.jsonl` (etiqueta, es_oro), `consultas.npy`, `consultas.jsonl`
(id, tarea, opciones, etiqueta, l3, l2_probs, margen, escala, motivo, final) y `referencia.json` (u, tareas «siempre», exactitudes).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from model.alt import escalado as es
from model.alt.auditoria import leer_varios, macro, mitad

AUROC_SIEMPRE = 0.55
F = 0.25


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--bateria", "--vec", "--fondo", "--l3-fondo", "--l3", "--salida"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    from model.alt.p0_medir import cargar_bateria
    z = np.load(Path(a.vec) / "fondo.npz")
    fondo = leer_varios(a.fondo)[:len(z["ids"])]
    l3f = leer_varios(a.l3_fondo)
    l3 = leer_varios(a.l3)
    bats = [d for d in a.bateria.split(",") if d]
    tareas = es.cargar(bats, a.vec, fondo, l3f, l3, z)
    A = {c: np.array([k for k, i in enumerate(t["ids"]) if mitad(i) == 0], int) for c, t in tareas.items()}
    B = {c: np.array([k for k, i in enumerate(t["ids"]) if mitad(i) == 1], int) for c, t in tareas.items()}
    au = {c: es.auroc(t["sen"]["margen"][A[c]], t["ok2"][A[c]]) for c, t in tareas.items()}
    siempre = sorted(c for c, t in tareas.items() if t["oro"] == 0 or au[c] is None or au[c] < AUROC_SIEMPRE)
    resto = [c for c in tareas if c not in siempre]
    u = float(np.quantile(np.concatenate([tareas[c]["sen"]["margen"][A[c]] for c in resto]), F))
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    # bancos: reconstruidos igual que escalado.cargar
    sop_ev = {}
    for d in bats:
        sop_ev.update(cargar_bateria(d))
    fo_et = es.etiquetas(fondo, l3f)
    pos = {i: k for k, i in enumerate(z["ids"])}
    l3_et = {}
    for c, (sop, ev) in sop_ev.items():
        l3_et.update(es.etiquetas(ev, l3))
    Q, filas, acc = [], [], {}
    for c, t in sorted(tareas.items()):
        sop, ev = sop_ev[c]
        zz = np.load(Path(a.vec) / (c + ".npz"))
        fs = [f for f in fondo if f["conjunto"] == c and fo_et.get(f["id"]) is not None and f["id"] in pos]
        Xg = zz["Xtr"] if len(sop) else np.zeros((0, zz["Xte"].shape[1]))
        Xf = z["X"][[pos[f["id"]] for f in fs]] if fs else np.zeros((0, zz["Xte"].shape[1]))
        np.save(sal / ("banco-%s.npy" % c), np.vstack([Xg, Xf]).astype(np.float32))
        with open(sal / ("banco-%s.jsonl" % c), "w", encoding="utf-8") as fh:
            for r in sop:
                fh.write(json.dumps({"etiqueta": r["etiqueta"], "es_oro": True}, ensure_ascii=False) + "\n")
            for f in fs:
                fh.write(json.dumps({"etiqueta": fo_et[f["id"]], "es_oro": False}, ensure_ascii=False) + "\n")
        idte = {r["id"]: k for k, r in enumerate(ev)}
        okc = []
        for j in B[c]:
            i = t["ids"][j]
            r = ev[idte[i]]
            P, sen = es.l2(zz["Xte"][[idte[i]]], [r["opciones"]], Xg, [x["etiqueta"] for x in sop], Xf, [fo_et[f["id"]] for f in fs])
            p = P[0, :len(r["opciones"])]
            l2 = r["opciones"][int(np.argmax(p))]
            if c in siempre:
                esc, mot = True, ("sin_oro" if t["oro"] == 0 else "auroc")
            else:
                esc, mot = bool(sen["margen"][0] < u), ("margen" if sen["margen"][0] < u else "")
            fin = l3_et.get(i) if esc else l2
            okc.append(1.0 if fin == r["etiqueta"] else 0.0)
            Q.append(zz["Xte"][idte[i]].astype(np.float32))
            filas.append({"id": i, "tarea": c, "opciones": r["opciones"], "etiqueta": r["etiqueta"], "l3": l3_et.get(i),
                          "l2": l2, "l2_probs": [round(float(x), 6) for x in p], "margen": round(float(sen["margen"][0]), 6),
                          "sim_oro": round(float(sen["sim_oro"][0]), 6), "escala": esc, "motivo": mot, "final": fin})
        acc[c] = {"cascada": float(np.mean(okc)), "l2": float(t["ok2"][B[c]].mean()), "l3": float(t["ok3"][B[c]].mean()),
                  "escalado": float(np.mean([f["escala"] for f in filas if f["tarea"] == c]))}
    np.save(sal / "consultas.npy", np.vstack(Q))
    with open(sal / "consultas.jsonl", "w", encoding="utf-8") as fh:
        for f in filas:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    ref = {"regla": {"senal": "margen", "f": F, "u": round(u, 6), "auroc_siempre_lt": AUROC_SIEMPRE, "s_oro": es.S_ORO, "vecinos": es.VECINOS, "temp": es.TEMP},
           "auroc_margen_A": {c: (None if v is None else round(v, 4)) for c, v in au.items()}, "siempre": siempre,
           "por_tarea_B": {c: {k: round(v, 4) for k, v in d.items()} for c, d in acc.items()},
           "macro_sin_hc_B": {k: macro({c: d[k] for c, d in acc.items()}) for k in ("cascada", "l2", "l3")},
           "macro_todas_B": {k: macro({c: d[k] for c, d in acc.items()}, fuera=()) for k in ("cascada", "l2", "l3")},
           "n_consultas": len(filas), "dim": int(Q[0].shape[0]) if Q else 0,
           "transformers_vectores": str(z["transformers"]) if "transformers" in z.files else None}
    (sal / "referencia.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: ref[k] for k in ("regla", "siempre", "macro_sin_hc_B", "macro_todas_B", "n_consultas")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
