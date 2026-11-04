"""Desglose de JevBench para comparar variantes y semillas: exactitud con IC95 bootstrap, por categoría, por split y MAE de score.

    python -m model.evaluacion.jevbench_desglose --jevbench RUTA --salida DIR \\
        --grupo base=pred/s0.jsonl,pred/s1.jsonl --grupo acum=pred/a0.jsonl,pred/a1.jsonl [--base base]

Cada `--grupo nombre=ficheros` son las predicciones (`{id, probs}` de `jevbench_l2.py`) de una variante, una por semilla. Para cada grupo:
  · por semilla: exactitud global con IC95 (bootstrap sobre casos, 2000 remuestreos, semilla 0, percentil), exactitud por categoría,
    por split y MAE de los casos score; una predicción sin respuesta (probs vacías) cuenta como fallo;
  · entre semillas: media ± desviación (ddof = 1) de cada cifra, y si `media − desv > base` (la media del grupo `--base`).
Categoría = campo `tarea` de la fila de `jevbench.py` (la `family` del caso: intent, trap, adversarial…). Se añade «trap+adversarial»
(las familias trap y adversarial juntas). Split = easy/hard/original, del nombre del fichero de origen (`fuente` = «JevBench <split>»).
MAE = |nivel predicho − nivel oro| en niveles de score; el nivel predicho es la mediana CORAL (`jevbench_informe.top1`), como la exactitud.
Escribe `DIR/desglose.json` y `DIR/desglose.md`. Solo biblioteca estándar y numpy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .jevbench_informe import leer, top1

REMUESTREOS = 2000
COMBINADA = "trap+adversarial"
FAMILIAS_COMBINADA = ("trap", "adversarial")


def split_de(fila):
    """easy/hard/original: `fuente` = «JevBench <split>» (el nombre del fichero de origen)."""
    return str(fila.get("fuente", "")).split()[-1] or "?"


def ic95(ok, remuestreos=REMUESTREOS, semilla=0):
    """(media, bajo, alto): percentiles 2,5 y 97,5 de la media de `ok` (0/1) en `remuestreos` remuestreos con reposición."""
    v = np.asarray(ok, dtype=float)
    if not len(v):
        return (None, None, None)
    ix = np.random.default_rng(semilla).integers(0, len(v), size=(remuestreos, len(v)))
    m = v[ix].mean(1)
    return (float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def _acc(v):
    return {"n": len(v), "acc": float(np.mean(v)) if v else None}


def desglose(filas, preds):
    """Cifras de una predicción (`preds`: {id: {probs}}) sobre las `filas` que la tienen."""
    fs = [r for r in filas if r["id"] in preds]
    ok, err, cat, spl = {}, [], {}, {}
    for r in fs:
        pr = top1(preds[r["id"]]["probs"], r["opciones"], r["tipo"])
        ok[r["id"]] = float(pr == r["etiqueta"])
        if r["tipo"] == "score":
            niv = sorted(int(o) for o in r["opciones"])
            oro = int(r["etiqueta"])
            # sin respuesta: el peor error posible (el extremo opuesto al oro)
            err.append(abs(int(pr) - oro) if pr is not None else max(oro - niv[0], niv[-1] - oro))
        cat.setdefault(r["tarea"], []).append(ok[r["id"]])
        spl.setdefault(split_de(r), []).append(ok[r["id"]])
    m, lo, hi = ic95(list(ok.values()))
    comb = [x for c in FAMILIAS_COMBINADA for x in cat.get(c, [])]
    return {
        "global": {"n": len(ok), "acc": m, "ic95": [lo, hi]},
        "por_categoria": {c: _acc(v) for c, v in sorted(cat.items())},
        COMBINADA: _acc(comb),
        "por_split": {s: _acc(v) for s, v in sorted(spl.items())},
        "mae_score": {"n": len(err), "mae": float(np.mean(err)) if err else None},
    }


def _cifras(d):
    """Aplana un desglose a {nombre: valor} para promediar entre semillas."""
    out = {"global": d["global"]["acc"], COMBINADA: d[COMBINADA]["acc"], "mae_score": d["mae_score"]["mae"]}
    out.update({"cat:" + c: v["acc"] for c, v in d["por_categoria"].items()})
    out.update({"split:" + s: v["acc"] for s, v in d["por_split"].items()})
    return out


def agrupar(filas, grupos, base=None):
    """grupos: {nombre: [pred, ...]} (pred = {id: {...}}). → {nombre: {n_semillas, semillas: [desglose], media_desv, vs_base}}."""
    res = {}
    for nombre, ps in grupos.items():
        ds = [desglose(filas, p) for p in ps]
        cs = [_cifras(d) for d in ds]
        md = {}
        for k in sorted({k for c in cs for k in c}):
            v = [c[k] for c in cs if c.get(k) is not None]
            md[k] = {"media": float(np.mean(v)), "desv": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n_semillas": len(v)} if v else None
        res[nombre] = {"n_semillas": len(ds), "semillas": ds, "media_desv": md}
    if base is not None and base in res:
        ref = res[base]["media_desv"]
        for nombre, r in res.items():
            if nombre == base:
                continue
            # MAE: menos es mejor, así que la condición es media + desv < base
            r["vs_base"] = {k: (v["media"] + v["desv"] < ref[k]["media"]) if k == "mae_score" else (v["media"] - v["desv"] > ref[k]["media"])
                            for k, v in r["media_desv"].items() if v and ref.get(k)}
    return res


def _f(x):
    return "-" if x is None else "%.3f" % x


def informe_md(res, base, n_filas):
    L = ["# Desglose JevBench", "", "%d casos. IC95 por bootstrap sobre casos (2000 remuestreos, semilla 0). Media ± desv. entre semillas (ddof 1)." % n_filas, ""]
    L += ["## Resumen por variante", "",
          "| variante | semillas | exactitud | trap+adversarial | MAE score | media−desv > base (global) | media−desv > base (trap+adv) |", "|---|---|---|---|---|---|---|"]
    for nombre, r in res.items():
        md, vb = r["media_desv"], r.get("vs_base", {})
        celda = lambda k: "%s ± %s" % (_f(md[k]["media"]), _f(md[k]["desv"])) if md.get(k) else "-"
        L.append("| %s | %d | %s | %s | %s | %s | %s |" % (nombre, r["n_semillas"], celda("global"), celda(COMBINADA), celda("mae_score"),
                 "(base)" if nombre == base else vb.get("global", "-"), "(base)" if nombre == base else vb.get(COMBINADA, "-")))
    L += ["", "Para MAE la marca va en `vs_base.mae_score` del json: exige media + desv < base (menos es mejor).", ""]
    for nombre, r in res.items():
        splits = sorted(r["semillas"][0]["por_split"])
        L += ["## %s" % nombre, "",
              "| semilla | n | exactitud | IC95 | trap+adversarial | MAE score (n) | " + " | ".join("split " + s for s in splits) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(splits)]
        for i, d in enumerate(r["semillas"]):
            L.append("| %d | %d | %s | %s–%s | %s | %s (%d) | %s |" % (
                i, d["global"]["n"], _f(d["global"]["acc"]), _f(d["global"]["ic95"][0]), _f(d["global"]["ic95"][1]), _f(d[COMBINADA]["acc"]),
                _f(d["mae_score"]["mae"]), d["mae_score"]["n"], " | ".join(_f(d["por_split"][s]["acc"]) for s in splits)))
        L += ["", "| categoría | n | media ± desv |", "|---|---|---|"]
        for k in sorted(k for k in r["media_desv"] if k.startswith("cat:")):
            c = k[4:]
            md = r["media_desv"][k]
            L.append("| %s | %d | %s ± %s |" % (c, r["semillas"][0]["por_categoria"][c]["n"], _f(md["media"]), _f(md["desv"])))
        L.append("")
    return "\n".join(L)


def leer_pred(ruta):
    return {p["id"]: p for p in leer(ruta)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jevbench", required=True, help="clon del repo, datasets/public o un .jsonl")
    ap.add_argument("--grupo", action="append", required=True, metavar="NOMBRE=F1,F2", help="variante y sus predicciones (una por semilla); repetible")
    ap.add_argument("--base", help="nombre del grupo de referencia (por defecto el primero)")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args(argv)
    from .jevbench import cargar_jevbench
    filas = cargar_jevbench(a.jevbench)
    grupos = {}
    for g in a.grupo:
        nombre, _, fs = g.partition("=")
        if not fs:
            ap.error("--grupo espera NOMBRE=F1,F2: %r" % g)
        grupos[nombre] = [leer_pred(f) for f in fs.split(",") if f]
    base = a.base or next(iter(grupos))
    if base not in grupos:
        ap.error("--base %r no es un grupo" % base)
    res = agrupar(filas, grupos, base)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "desglose.json").write_text(json.dumps({"base": base, "n": len(filas), "grupos": res}, indent=1, ensure_ascii=False))
    md = informe_md(res, base, len(filas))
    (sal / "desglose.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
