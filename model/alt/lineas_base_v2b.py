"""R10-T3b: variantes de especificación del 4B en la batería v2 (dev) y lectura congelada (`crit-r10-t3b.json`). Solo biblioteca estándar y numpy.

    python -m model.alt.lineas_base_v2b rubrica --dev DIR_DEV --rubricas rubricas-r10-t3b.json --salida DIR     (DIR/rub/test/<tarea>.jsonl de las score)
    python -m model.alt.lineas_base_v2b medir --dev DIR_DEV --pred DIR_PRED --t3 DIR_T3 --salida cifras.json [--crit crit-r10-t3b.json] [--presel presel.json]

`rubrica` pone la descripción explícita de cada nivel (con su ejemplo corto) en `descripciones` de las filas de test de las tareas score; el prompt
es el mismo de `llm_logprobs.py`, que ya manda las descripciones. `medir` compone cada sistema por tarea (DIR_T3/<sistema>/<tarea>.jsonl para lo que
viene de T3, DIR_PRED/<sistema>/<tarea>.jsonl para lo nuevo; lo que dice el criterio), aplica la esperanza ordinal donde el criterio lo pide y mide con
las MISMAS funciones de `lineas_base_v2.py`. Veredicto por candidato: PASA si la macro de las 6 tareas gana ≥ macro_min frente a la base y ninguna
tarea `choice` pierde más de choice_max_perdida, con cobertura completa. Las score van aparte (macro, ±1 y MAE).
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
    sal = Path(a.salida) / "rub" / "test"
    sal.mkdir(parents=True, exist_ok=True)
    for t, r in rub.items():
        _, ev = dev[t]
        niv = r["niveles"]
        for f in ev:
            if sorted(f["opciones"], key=int) != sorted(niv, key=int):
                raise SystemExit("%s: los niveles de la rúbrica no son las opciones de la tarea" % t)
        with open(sal / (t + ".jsonl"), "w", encoding="utf-8") as fh:
            for f in ev:
                g = dict(f)
                g["descripciones"] = {o: "%s Ejemplo: «%s»" % (niv[o]["descripcion"], niv[o]["ejemplo"]) for o in f["opciones"]}
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        print("rubrica", t, len(ev))


def fuente_de(spec, t, pred, t3):
    """spec[t] o spec['*']: 'sistema' (DIR_PRED) o 't3:sistema' (DIR_T3); devuelve la ruta."""
    s = spec.get(t, spec["*"])
    base, nombre = (t3, s[3:]) if s.startswith("t3:") else (pred, s)
    return Path(base) / nombre / (t + ".jsonl")


def macro_score(c):
    sc = [d for d in c["tareas"].values() if d["tipo"] == "score"]
    return {"n_tareas": len(sc), "exactitud": round(float(np.mean([d["exactitud"] for d in sc])), 4),
            "exactitud_sin_sinteticas": round(float(np.mean([d["exactitud"] for d in sc if not d["sintetica"]])), 4),
            "exactitud_pm1": round(float(np.mean([d["exactitud_pm1"] for d in sc])), 4), "mae": round(float(np.mean([d["mae"] for d in sc if d["mae"] is not None])), 4) if any(d["mae"] is not None for d in sc) else None}


def cmd_medir(a):
    crit_bytes = Path(a.crit).read_bytes()
    crit = json.loads(crit_bytes)
    dev = lb.cargar_dev(a.dev)
    if sorted(dev) != sorted(crit["tareas_dev"]):
        raise SystemExit("las tareas de %s no son las de dev del criterio" % a.dev)
    sint, ref = set(crit["sinteticas"]), crit["referencia"]
    out = {"criterio_sha256": hashlib.sha256(crit_bytes).hexdigest(), "referencia": ref, "sinteticas": sorted(sint), "sistemas": {}, "pareadas": {}}
    oks = {}
    for s, spec in crit["sistemas"].items():
        spec = dict(spec)
        esp = bool(spec.pop("esperanza", False))
        out["sistemas"][s], oks[s] = lb.cifras_sistema(dev, lambda t, spec=spec: fuente_de(spec, t, a.pred, a.t3), sint, esp)
        out["sistemas"][s]["macro_score"] = macro_score(out["sistemas"][s])
    for s in out["sistemas"]:
        if s != ref:
            out["pareadas"][s] = lb.par_sistema(dev, oks[s], oks[ref], sint)
    for par in crit.get("pares_extra", []):
        out.setdefault("pareadas_extra", {})["%s_menos_%s" % tuple(par)] = lb.par_sistema(dev, oks[par[0]], oks[par[1]], sint)
    pm = crit["pasa"]
    ver = {}
    for c in crit["candidatos"]:
        cob = min(out["sistemas"][c]["cobertura_total"], out["sistemas"][ref]["cobertura_total"])
        p = out["pareadas"][c]
        choice = {t: d["dif"] for t, d in p["tareas"].items() if out["sistemas"][c]["tareas"][t]["tipo"] == "choice"}
        peor = min(choice.values())
        v = {"macro_dif": p["macro"]["dif"], "ic95_tareas": p["macro"]["ic95_tareas"], "macro_dif_sin_sinteticas": p["macro_sin_sinteticas"]["dif"],
             "peor_choice": round(peor, 4), "cobertura": cob}
        if cob < 1.0:
            v["veredicto"] = "no medido"
        else:
            v["veredicto"] = "PASA" if p["macro"]["dif"] >= pm["macro_min"] and peor >= -pm["choice_max_perdida"] else "NO PASA"
        v["score_aparte"] = {"macro_score": out["sistemas"][c]["macro_score"], "base": out["sistemas"][ref]["macro_score"]}
        ver[c] = v
    out["veredictos"] = ver
    out["v4_topk"] = v4(dev, crit, a, out)
    if a.presel and Path(a.presel).exists():
        out["preseleccion"] = json.loads(Path(a.presel).read_text(encoding="utf-8"))
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for c, v in ver.items():
        print("%-14s %-9s macro %+.4f IC %s peor choice %+.4f cob %.2f" % (c, v["veredicto"], v["macro_dif"], v["ic95_tareas"], v["peor_choice"], v["cobertura"]))
    print("V4 topk:", json.dumps(out["v4_topk"]))


def v4(dev, crit, a, out):
    """Preselección top-k en finanzas: zs-topk frente al zero-shot completo de T3 en las filas que T3 llegó a puntuar (pareado)."""
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
    ok = cob >= 1.0 and d >= -crit["v4"]["perdida_max"]
    return {"filas_comunes": len(comun), "exactitud_topk": round(float(on.mean()), 4), "exactitud_t3": round(float(o3.mean()), 4), "dif": round(d, 4),
            "ic95": lb.ic_dif_filas(on, o3), "mcnemar_p": p, "cobertura_topk": cob, "cobertura_t3": round(len(comun) / len(ev), 4),
            "veredicto": "PASA" if ok else ("NO PASA" if cob >= 1.0 else "no medido")}


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
    a = ap.parse_args(argv)
    {"rubrica": cmd_rubrica, "medir": cmd_medir}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
