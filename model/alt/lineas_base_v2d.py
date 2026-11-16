"""R10-D: el oro bien usado con el 4B en la batería v2 (dev): demostraciones elegidas por kNN y mezcla de posteriores con el kNN-e5 de oro. Solo biblioteca estándar y numpy.

    python -m model.alt.lineas_base_v2d medir --dev DIR_DEV --pred DIR_PRED --t3 DIR_T3 --salida cifras.json [--crit crit-r10-d.json]

Sistemas (el criterio dice de dónde sale cada uno; en finanzas todos son «4B sobre las 10 candidatas del kNN»):
  zs         el 4B zero-shot (T3 en 5 tareas; zs-topk en finanzas)          fs8-azar  8 ejemplos al azar (T3; fs8-topk en finanzas)  = REFERENCIA
  fs8-knn    8 demostraciones por fila, las más parecidas por coseno de e5   knn       kNN-e5 k = 8 de oro, 32 vecinos, T = 0,05 (el posterior del oro)
  <base>+knn p = w·p_kNN + (1−w)·p_4B con UN w global del criterio (0; 0,1; …; 1), para base en {zs, fs8-azar, fs8-knn}
Partición: la mitad A (`auditoria.mitad(id) == 0`) elige w por base y la mejor variante de demostraciones (la de mayor macro en A, sin mezcla; empate: fs8-knn); la mitad B
mide, la referencia también. El sistema final candidato es la mezcla sobre esa variante. Macro = media de las 6 exactitudes por tarea; IC95 de la macro (diferencia
pareada) por bootstrap sobre tareas. Score aparte: exactitud, QWK, MAE y ±1; macro también sin las sintéticas.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb
from model.alt.auditoria import mitad


def partir(dev):
    """({tarea: (soporte, test de A)}, {tarea: (soporte, test de B)}) con la partición por hash del id de auditoria.mitad."""
    A = {t: (s, [r for r in ev if mitad(r["id"]) == 0]) for t, (s, ev) in dev.items()}
    B = {t: (s, [r for r in ev if mitad(r["id"]) == 1]) for t, (s, ev) in dev.items()}
    return A, B


def probs_de(path):
    if not Path(path).exists():
        return {}
    return {r["id"]: (r["probs"] if r.get("valida", True) and r.get("probs") else None) for r in lb.leer(path)}


def mezclar(pk, p4, r, w):
    """p = w·p_kNN + (1−w)·p_4B sobre las opciones de la fila; None si falta el 4B (salvo con w = 1, que es el kNN solo)."""
    if w >= 1.0:
        return dict(pk) if pk else None
    if not pk or not p4:
        return None
    return {o: w * float(pk.get(o, 0.0)) + (1 - w) * float(p4.get(o, 0.0)) for o in r["opciones"]}


def acc_macro(dev, fuentes, w, pk_por_tarea):
    """Exactitud macro sobre las filas de `dev` (A o B) de la base `fuentes` ({tarea: ruta}) mezclada con peso w (None: la base sola)."""
    accs = []
    for t, (_, ev) in dev.items():
        p4 = probs_de(fuentes[t])
        ok = []
        for r in ev:
            p = p4.get(r["id"]) if w is None else mezclar(pk_por_tarea[t].get(r["id"]), p4.get(r["id"]), r, w)
            y = lb.top1(p, r) if p else None
            ok.append(1.0 if y is not None and y == r["etiqueta"] else 0.0)
        accs.append(float(np.mean(ok)) if ok else 0.0)
    return float(np.mean(accs))


def fuente_de(spec, t, pred, t3):
    s = spec.get(t, spec["*"])
    base, nombre = (t3, s[3:]) if s.startswith("t3:") else (pred, s)
    return Path(base) / nombre / (t + ".jsonl")


def escribir_mezcla(dev, fuentes, w, pk_por_tarea, dest):
    for t, (_, ev) in dev.items():
        p4 = probs_de(fuentes[t])
        Path(dest).mkdir(parents=True, exist_ok=True)
        with open(Path(dest) / (t + ".jsonl"), "w", encoding="utf-8") as fh:
            for r in ev:
                p = mezclar(pk_por_tarea[t].get(r["id"]), p4.get(r["id"]), r, w)
                if p is not None or (r["id"] in p4 and r["id"] in pk_por_tarea[t]):   # sin kNN la fila falta (cobertura); con kNN pero 4B inválido, la fila es inválida
                    fh.write(json.dumps({"id": r["id"], "probs": p, "valida": p is not None, "ms": 0.0}) + "\n")


def cmd_medir(a):
    crit_bytes = Path(a.crit).read_bytes()
    crit = json.loads(crit_bytes)
    dev = lb.cargar_dev(a.dev)
    if sorted(dev) != sorted(crit["tareas_dev"]):
        raise SystemExit("las tareas de %s no son las de dev del criterio" % a.dev)
    sint, ref, pred, t3 = set(crit["sinteticas"]), crit["referencia"], Path(a.pred), a.t3
    A, B = partir(dev)
    fu = {s: {t: fuente_de(spec, t, pred, t3) for t in dev} for s, spec in crit["bases"].items()}
    kn = {t: probs_de(pred / crit["knn"] / (t + ".jsonl")) for t in dev}
    grid = crit["mezcla"]["w_grid"]
    out = {"criterio_sha256": hashlib.sha256(crit_bytes).hexdigest(), "referencia": ref, "sinteticas": sorted(sint), "n_A": sum(len(e) for _, e in A.values()),
           "n_B": sum(len(e) for _, e in B.values()), "A": {}, "sistemas_B": {}, "pareadas_B": {}}
    # --- elección en A: w por base y la mejor variante de demostraciones
    solo_A = {s: acc_macro(A, fu[s], None, kn) for s in crit["bases"]}
    out["A"]["macro_sin_mezcla"] = {s: round(v, 4) for s, v in solo_A.items()}
    w_por = {}
    for s in crit["bases"]:
        curva = {str(w): round(acc_macro(A, fu[s], w, kn), 4) for w in grid}
        w_por[s] = min(grid, key=lambda w: (-curva[str(w)], w))   # empate: el w menor
        out["A"]["curva_w"] = out["A"].get("curva_w", {})
        out["A"]["curva_w"][s] = curva
    out["A"]["w"] = w_por
    demos = crit["variantes_demos"]   # ["fs8-azar", "fs8-knn"]
    mejor = max(demos, key=lambda s: (solo_A[s], s == demos[-1]))   # empate: la última (fs8-knn)
    final = mejor + "+knn"
    out["A"]["mejor_variante_demos"], out["sistema_final"] = mejor, final
    # --- medida en B
    sistemas, oks = {}, {}
    mixdir = Path(a.salida).with_suffix(".mezclas")
    for s in crit["bases"]:
        sistemas[s] = fu[s]
        escribir_mezcla(dev, fu[s], w_por[s], kn, mixdir / (s + "+knn"))
        sistemas[s + "+knn"] = {t: mixdir / (s + "+knn") / (t + ".jsonl") for t in dev}
    sistemas["knn"] = {t: pred / crit["knn"] / (t + ".jsonl") for t in dev}
    for s, fuentes in sistemas.items():
        out["sistemas_B"][s], oks[s] = lb.cifras_sistema(B, lambda t, f=fuentes: f[t], sint)
        out["sistemas_B"][s]["macro_score"] = macro_score(out["sistemas_B"][s])
        out["sistemas_B"][s]["w"] = w_por[s[:-4]] if s.endswith("+knn") else None
    for s in oks:
        if s != ref:
            out["pareadas_B"][s] = lb.par_sistema(B, oks[s], oks[ref], sint)
    out["veredicto"] = veredicto(out, crit, final)
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    v = out["veredicto"]
    print("final %s (w=%s, demos=%s): %s | macro B %+.4f IC %s | peor tarea %+.4f | cobertura %.3f" % (final, w_por[mejor], mejor, v["veredicto"], v["macro_dif"], v["ic95_tareas"], v["peor_tarea"], v["cobertura"]))
    for s, c in out["sistemas_B"].items():
        print("%-16s macro B %.4f (sin sint. %.4f) cobertura %.3f invalidas %d | score: acc %s qwk %s mae %s" % (
            s, c["macro"]["valor"], c["macro_sin_sinteticas"]["valor"], c["cobertura_total"], c["invalidas_total"], c["macro_score"]["exactitud"], c["macro_score"]["qwk"], c["macro_score"]["mae"]))


def macro_score(c):
    sc = [d for d in c["tareas"].values() if d["tipo"] == "score"]
    f = lambda k, xs=None: round(float(np.mean([d[k] for d in (xs or sc) if d[k] is not None])), 4) if any(d[k] is not None for d in (xs or sc)) else None  # noqa: E731
    sin = [d for d in sc if not d["sintetica"]]
    return {"n_tareas": len(sc), "exactitud": f("exactitud"), "qwk": f("qwk"), "mae": f("mae"), "exactitud_pm1": f("exactitud_pm1"),
            "exactitud_sin_sinteticas": f("exactitud", sin), "qwk_sin_sinteticas": f("qwk", sin)}


def veredicto(out, crit, final):
    """PASA si el final gana ≥ macro_min a la referencia en B, ninguna tarea pierde más de tarea_max_perdida y la cobertura (final y referencia) es del 100 %."""
    pm, ref = crit["pasa"], crit["referencia"]
    s, r = out["sistemas_B"][final], out["sistemas_B"][ref]
    p = out["pareadas_B"][final]
    cob = min(s["cobertura_total"], r["cobertura_total"])
    peor = min(d["dif"] for d in p["tareas"].values())
    v = {"final": final, "macro_dif": p["macro"]["dif"], "ic95_tareas": p["macro"]["ic95_tareas"], "macro_dif_sin_sinteticas": p["macro_sin_sinteticas"]["dif"],
         "ic95_tareas_sin_sinteticas": p["macro_sin_sinteticas"]["ic95_tareas"], "peor_tarea": round(peor, 4), "cobertura": cob,
         "score_aparte": {"final": s["macro_score"], "referencia": r["macro_score"]}}
    v["veredicto"] = "no medido" if cob < 1.0 else ("PASA" if p["macro"]["dif"] >= pm["macro_min"] and peor >= -pm["tarea_max_perdida"] else "NO PASA")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("medir")
    m.add_argument("--dev", required=True)
    m.add_argument("--pred", required=True)
    m.add_argument("--t3", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--crit", default=str(Path(__file__).with_name("crit-r10-d.json")))
    a = ap.parse_args(argv)
    cmd_medir(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
