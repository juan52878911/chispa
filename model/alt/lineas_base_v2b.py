"""R10-T3b: score ordinal calibrado (y choice informativo) con el 4B en la batería v2 (dev) y lectura congelada (`crit-r10-t3b.json`). Solo biblioteca estándar y numpy.

    python -m model.alt.lineas_base_v2b rubrica --dev DIR_DEV --rubricas rubricas-r10-t3b.json --salida DIR     (DIR/rub/{test,soporte}/<tarea>.jsonl de las score)
    python -m model.alt.lineas_base_v2b medir --dev DIR_DEV --pred DIR_PRED --t3 DIR_T3 --salida cifras.json [--crit crit-r10-t3b.json] [--presel presel.json]

`rubrica` pone la descripción explícita de cada nivel (con su ejemplo corto) en `descripciones` de las filas de las tareas score; el prompt es el de
`llm_logprobs.py`, que ya manda las descripciones. `medir` compone cada sistema por tarea (DIR_T3/<sistema>/<tarea>.jsonl lo de T3; DIR_PRED/<sistema>/<tarea>.jsonl
lo nuevo), aplica la lectura de score que pida el criterio (mediana CORAL, argmax, esperanza ordinal o esperanza con umbrales ajustados en el soporte de oro) y mide
con las MISMAS funciones de `lineas_base_v2.py`: exactitud, QWK, MAE y ±1 en score; exactitud en choice. Los logprobs de TODOS los niveles salen de las `probs`
de `llm_logprobs.py` (una probabilidad por opción; con ≤ 62 opciones cada etiqueta sale del top 20 o se puntúa por prefijo: `raw.con_piso` cuenta las que no).

Umbrales (variante calibrada): con las predicciones del 4B sobre el soporte de oro (8 por nivel), se toma la mediana de la esperanza de cada nivel, se fuerza que no
decrezca con la regresión isotónica (PAV) y los K−1 umbrales son los puntos medios entre niveles consecutivos; nivel predicho = 1 + nº de umbrales por debajo de la esperanza.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb


def cmd_rubrica(a):
    rub = json.loads(Path(a.rubricas).read_text(encoding="utf-8"))["tareas"]
    dev = lb.cargar_dev(a.dev)
    for t, r in rub.items():
        sop, ev = dev[t]
        niv = r["niveles"]
        for sub, filas in (("test", ev), ("soporte", sop)):
            d = Path(a.salida) / "rub" / sub
            d.mkdir(parents=True, exist_ok=True)
            with open(d / (t + ".jsonl"), "w", encoding="utf-8") as fh:
                for f in filas:
                    if sorted(f["opciones"], key=int) != sorted(niv, key=int):
                        raise SystemExit("%s: los niveles de la rúbrica no son las opciones de la tarea" % t)
                    g = dict(f)
                    g["descripciones"] = {o: "%s Ejemplo: «%s»" % (niv[o]["descripcion"], niv[o]["ejemplo"]) for o in f["opciones"]}
                    fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        print("rubrica", t, len(ev), len(sop))


def fuente_de(spec, t, pred, t3):
    """spec[t] o spec['*']: 'sistema' (DIR_PRED) o 't3:sistema' (DIR_T3); devuelve la ruta."""
    s = spec.get(t, spec["*"])
    base, nombre = (t3, s[3:]) if s.startswith("t3:") else (pred, s)
    return Path(base) / nombre / (t + ".jsonl")


def pav(y):
    """Regresión isotónica no decreciente (mínimos cuadrados, pesos iguales)."""
    bloques = []
    for v in y:
        bloques.append([float(v), 1])
        while len(bloques) > 1 and bloques[-2][0] > bloques[-1][0]:
            (m2, n2), (m1, n1) = bloques[-2], bloques[-1]
            bloques[-2:] = [[(m2 * n2 + m1 * n1) / (n1 + n2), n1 + n2]]
    return [m for m, n in bloques for _ in range(n)]


def umbrales(sop, preds, excluir=()):
    """K−1 umbrales de la esperanza, ajustados con las predicciones del soporte de oro (ver el docstring del módulo)."""
    ops = lb.orden(sop[0])
    por = {o: [] for o in ops}
    for r in sop:
        p = preds.get(r["id"])
        if r["id"] in excluir or not p or not p.get("probs") or not p.get("valida", True):
            continue
        pr = [float(p["probs"].get(o, 0.0)) for o in ops]
        if sum(pr) > 0:
            por[r["etiqueta"]].append(lb.esperanza(ops, pr))
    if any(not v for v in por.values()):
        raise ValueError("faltan predicciones del soporte de algún nivel")
    med = pav([float(np.median(por[o])) for o in ops])
    return [round((a + b) / 2, 6) for a, b in zip(med[:-1], med[1:])]


def macro_score(c):
    sc = [d for d in c["tareas"].values() if d["tipo"] == "score"]
    f = lambda k, xs=None: round(float(np.mean([d[k] for d in (xs or sc) if d[k] is not None])), 4)  # noqa: E731
    sin = [d for d in sc if not d["sintetica"]]
    return {"n_tareas": len(sc), "exactitud": f("exactitud"), "qwk": f("qwk"), "mae": f("mae"), "exactitud_pm1": f("exactitud_pm1"),
            "exactitud_sin_sinteticas": f("exactitud", sin), "qwk_sin_sinteticas": f("qwk", sin)}


def ids_ejemplos(rub):
    return {n["fuente_ejemplo"] for v in rub["tareas"].values() for n in v["niveles"].values()}


def cmd_medir(a):
    crit_bytes = Path(a.crit).read_bytes()
    crit = json.loads(crit_bytes)
    dev = lb.cargar_dev(a.dev)
    if sorted(dev) != sorted(crit["tareas_dev"]):
        raise SystemExit("las tareas de %s no son las de dev del criterio" % a.dev)
    sint, ref = set(crit["sinteticas"]), crit["referencia"]
    rub = json.loads(Path(a.rubricas).read_text(encoding="utf-8"))
    pred = Path(a.pred)
    if "mayoritaria" in crit["sistemas"]:   # sin modelo: la clase mayoritaria del soporte de oro
        for t, (sop, ev) in dev.items():
            m = lb.mayoritaria(sop, ev)["clase"]
            (pred / "mayoritaria").mkdir(parents=True, exist_ok=True)
            (pred / "mayoritaria" / (t + ".jsonl")).write_text("".join(json.dumps({"id": r["id"], "probs": {o: float(o == m) for o in r["opciones"]}, "valida": True, "ms": 0.0}) + "\n" for r in ev), encoding="utf-8")
    out = {"criterio_sha256": hashlib.sha256(crit_bytes).hexdigest(), "referencia": ref, "sinteticas": sorted(sint), "sistemas": {}, "pareadas": {}, "umbrales": {}}
    oks = {}
    for s, spec in crit["sistemas"].items():
        spec = dict(spec)
        modo, cal = spec.pop("esperanza", False), spec.pop("calibrar", None)
        excl = ids_ejemplos(rub) if spec.pop("excluir_ejemplos_rubrica", False) else ()
        esp, sin_umbral = modo, set()
        if cal:
            esp = {}
            for t, (sop, ev) in dev.items():
                f = pred / cal / (t + ".jsonl")
                if ev[0]["tipo"] == "score" and f.exists():
                    try:
                        esp[t] = umbrales(sop, {r["id"]: r for r in lb.leer(f)}, excl)
                        out["umbrales"].setdefault(s, {})[t] = esp[t]
                    except ValueError as e:   # soporte sin predicciones de algún nivel: esa tarea queda sin cubrir («no medido»), no se mide con otra lectura
                        sin_umbral.add(t)
                        out["umbrales"].setdefault(s, {})[t] = str(e)
                elif ev[0]["tipo"] == "score":
                    sin_umbral.add(t)
        out["sistemas"][s], oks[s] = lb.cifras_sistema(dev, lambda t, spec=spec, su=sin_umbral: Path("/nonexistent") if t in su else fuente_de(spec, t, a.pred, a.t3), sint, esp)
        out["sistemas"][s]["macro_score"] = macro_score(out["sistemas"][s])
    for s in out["sistemas"]:
        if s != ref:
            out["pareadas"][s] = lb.par_sistema(dev, oks[s], oks[ref], sint)
    for par in crit.get("pares_extra", []):
        out.setdefault("pareadas_extra", {})["%s_menos_%s" % tuple(par)] = lb.par_sistema(dev, oks[par[0]], oks[par[1]], sint)
    out["veredictos"] = veredictos(out, crit)
    out["v4_topk"] = v4(dev, crit, a, out)
    if a.presel and Path(a.presel).exists():
        out["preseleccion"] = json.loads(Path(a.presel).read_text(encoding="utf-8"))
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for c, v in out["veredictos"].items():
        print("%-12s %-9s acc %+.4f qwk %+.4f cob %.2f bajo mayoritaria: %s" % (c, v["veredicto"], v["macro_acc_dif"], v["macro_qwk_dif"], v["cobertura_score"], v["bajo_mayoritaria"]))
    print("V4 topk:", json.dumps(out["v4_topk"]))


def veredictos(out, crit):
    """Score: PASA si la exactitud macro sube ≥ acc_min, el QWK macro sube ≥ qwk_min y ninguna tarea score queda por debajo de la clase mayoritaria (soporte de oro)."""
    pm, ref, ver = crit["pasa"], crit["referencia"], {}
    base = out["sistemas"][ref]
    for c in crit["candidatos"]:
        s = out["sistemas"][c]
        sc = {t: d for t, d in s["tareas"].items() if d["tipo"] == "score"}
        cob = min(d["cobertura"] for d in sc.values())
        bajo = sorted(t for t, d in sc.items() if d["exactitud"] < d["mayoritaria"]["exactitud_test"])
        da, dq = round(s["macro_score"]["exactitud"] - base["macro_score"]["exactitud"], 4), round(s["macro_score"]["qwk"] - base["macro_score"]["qwk"], 4)
        v = {"macro_acc_dif": da, "macro_qwk_dif": dq, "macro_mae_dif": round(s["macro_score"]["mae"] - base["macro_score"]["mae"], 4), "cobertura_score": cob,
             "bajo_mayoritaria": bajo, "macro_score": s["macro_score"], "base": base["macro_score"],
             "informativo_macro_6_tareas": out["pareadas"][c]["macro"]["dif"]}
        v["veredicto"] = "no medido" if cob < 1.0 else ("PASA" if da >= pm["acc_min"] and dq >= pm["qwk_min"] and not bajo else "NO PASA")
        ver[c] = v
    return ver


def v4(dev, crit, a, out):
    """Informativo: preselección top-k en finanzas, zs-topk frente al zero-shot completo de T3 en las filas que T3 llegó a puntuar (pareado)."""
    t = crit["v4"]["tarea"]
    _, ev = dev[t]
    f_t3, f_nu = Path(a.t3) / crit["v4"]["t3"] / (t + ".jsonl"), Path(a.pred) / crit["v4"]["nuevo"] / (t + ".jsonl")
    if not (f_t3.exists() and f_nu.exists()):
        return {"veredicto": "no medido", "motivo": "falta %s" % (f_t3 if not f_t3.exists() else f_nu)}
    p3 = {r["id"]: r for r in lb.leer(f_t3)}
    comun = [r for r in ev if r["id"] in p3]
    o3, _, _, _ = lb.evaluar_tarea(comun, p3)
    on, _, _, _ = lb.evaluar_tarea(comun, {r["id"]: r for r in lb.leer(f_nu)})
    na, nb, p = lb.mcnemar(on, o3)
    cob = out["sistemas"][crit["v4"]["sistema"]]["tareas"][t]["cobertura"]
    d = float(on.mean() - o3.mean())
    return {"filas_comunes": len(comun), "exactitud_topk": round(float(on.mean()), 4), "exactitud_t3": round(float(o3.mean()), 4), "dif": round(d, 4),
            "ic95": lb.ic_dif_filas(on, o3), "mcnemar_p": p, "cobertura_topk": cob, "cobertura_t3": round(len(comun) / len(ev), 4),
            "veredicto": "informativo" if cob >= 1.0 else "no medido"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rubrica")
    r.add_argument("--dev", required=True)
    r.add_argument("--rubricas", required=True)
    r.add_argument("--salida", required=True)
    m = sub.add_parser("medir")
    m.add_argument("--dev", required=True)
    m.add_argument("--pred", required=True)
    m.add_argument("--t3", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--presel")
    m.add_argument("--crit", default=str(Path(__file__).with_name("crit-r10-t3b.json")))
    m.add_argument("--rubricas", default=str(Path(__file__).with_name("rubricas-r10-t3b.json")))
    a = ap.parse_args(argv)
    {"rubrica": cmd_rubrica, "medir": cmd_medir}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
