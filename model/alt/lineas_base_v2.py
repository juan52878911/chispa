"""R10-T3: líneas base de la batería v2 (6 tareas de DESARROLLO) y lectura congelada (`crit-r10-t3.json`). Solo biblioteca estándar y numpy.

    python -m model.alt.lineas_base_v2 filas  --dev DIR_DEV --salida DIR           (test/<tarea>.jsonl y soporte/<tarea>.jsonl; imprime los recuentos)
    python -m model.alt.lineas_base_v2 medir  --dev DIR_DEV --pred DIR_PRED --salida cifras.json [--crit crit-r10-t3.json]

DIR_PRED/<sistema>/<tarea>.jsonl: una fila por caso con {id, probs, valida, ms} (la salida de `llm_logprobs.py` sirve tal cual; `lineas_base_v2_pred.py`
escribe lo mismo para NLI, L2 y kNN). El JSON de cifras no lleva textos, solo ids y números.

Exactitud: top-1 (en `score`, la mediana CORAL: nº de umbrales con P(y>j) > 0,5, como `model.evaluacion.metricas.etiqueta_coral`); una fila inválida
(sin probs, `valida` falso, suma 0) o sin predicción cuenta como fallo, y `cobertura` dice cuántas filas del test llegaron a puntuarse.
Macro = media de las exactitudes por tarea. IC95 de la macro: bootstrap sobre TAREAS (se remuestrean las tareas con reemplazo y se promedian sus
exactitudes); IC95 por tarea: bootstrap sobre filas. Diferencias pareadas frente al sistema de referencia (4B zero-shot): bootstrap pareado por filas,
McNemar exacto, y la macro con bootstrap sobre tareas. Tareas `score`: exactitud ±1 nivel y MAE (sobre las filas válidas), en niveles.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REMUESTREOS = 2000


def leer(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def cargar_dev(directorio):
    """{tarea: (soporte, test)}; ignora los ._* de macOS."""
    out = {}
    for f in sorted(Path(directorio).glob("*.jsonl")):
        if f.name.startswith("."):
            continue
        rs = leer(f)
        out[f.stem] = ([r for r in rs if r["split"] == "train"], [r for r in rs if r["split"] == "test"])
    return out


def cmd_filas(a):
    dev = cargar_dev(a.dev)
    sal = Path(a.salida)
    (sal / "test").mkdir(parents=True, exist_ok=True)
    (sal / "soporte").mkdir(parents=True, exist_ok=True)
    nt = ns = 0
    for t, (sop, ev) in dev.items():
        for sub, rs in (("test", ev), ("soporte", sop)):
            with open(sal / sub / (t + ".jsonl"), "w", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
        nt, ns = nt + len(ev), ns + len(sop)
    print("tareas=%d test=%d soporte=%d" % (len(dev), nt, ns))


# ---------------------------------------------------------------- predicción -> acierto
def orden(r):
    return sorted(r["opciones"], key=int) if r["tipo"] == "score" else list(r["opciones"])


def top1(probs, r, esp=False):
    """Opción predicha o None (inválida). `esp` (solo score): esperanza ordinal, la suma de nivel × p(nivel) redondeada al nivel, en vez de la mediana CORAL."""
    ops = orden(r)
    p = [float(probs.get(o, 0.0)) for o in ops] if probs else [0.0]
    if sum(p) <= 0:
        return None
    if r["tipo"] == "score" and esp:
        e = sum(int(o) * x for o, x in zip(ops, p)) / sum(p)
        return min(ops, key=lambda o: (abs(int(o) - e), int(o)))   # nivel más cercano; en el empate exacto, el menor
    if r["tipo"] == "score":
        q = np.asarray(p) / sum(p)
        cola = np.cumsum(q[::-1])[::-1][1:]
        return ops[int((cola > 0.5).sum())]
    return ops[int(np.argmax(p))]   # empate: la primera opción


def evaluar_tarea(ev, preds, esp=False):
    """Vectores por fila: acierto, predicción (o None), ms (o nan)."""
    ok, pred, ms = [], [], []
    for r in ev:
        p = preds.get(r["id"])
        y = top1(p["probs"], r, esp) if p and p.get("valida", True) and p.get("probs") else None
        pred.append(y)
        ok.append(1.0 if y is not None and y == r["etiqueta"] else 0.0)
        ms.append(float(p["ms"]) if p and p.get("ms") is not None else float("nan"))
    return np.array(ok), pred, np.array(ms), sum(1 for r in ev if r["id"] in preds)


def ic_filas(ok, semilla=0):
    if len(ok) == 0:
        return [None, None]
    rs = np.random.RandomState(semilla)
    m = ok[rs.randint(0, len(ok), (REMUESTREOS, len(ok)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def ic_tareas(v, semilla=0):
    """Bootstrap sobre tareas: remuestrea los valores por tarea con reemplazo y promedia."""
    v = np.asarray(v, dtype=float)
    rs = np.random.RandomState(semilla)
    m = v[rs.randint(0, len(v), (REMUESTREOS, len(v)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def mcnemar(a, b):
    """(solo a acierta, solo b acierta, p exacto bilateral)."""
    na, nb = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = na + nb
    if n == 0:
        return na, nb, 1.0
    k = min(na, nb)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return na, nb, round(min(1.0, 2 * p), 6)


def ic_dif_filas(a, b, semilla=0):
    d = a - b
    return ic_filas(d, semilla)


def pct(x, q):
    x = x[~np.isnan(x)]
    return round(float(np.percentile(x, q)), 2) if len(x) else None


def cifras_sistema(dev, fuente, sinteticas, esp=False):
    """`fuente`: carpeta con <tarea>.jsonl o función tarea -> ruta (None si no hay)."""
    por, oks = {}, {}
    for t, (_, ev) in dev.items():
        f = fuente(t) if callable(fuente) else Path(fuente) / (t + ".jsonl")
        preds = {r["id"]: r for r in leer(f)} if f and Path(f).exists() else {}
        ok, pred, ms, presentes = evaluar_tarea(ev, preds, esp)
        oks[t] = ok
        cub = np.array([r["id"] in preds for r in ev])
        d = {"n_test": len(ev), "cobertura": round(presentes / len(ev), 4), "sintetica": t in sinteticas, "tipo": ev[0]["tipo"],
             "exactitud": round(float(ok.mean()), 4), "exactitud_cubiertas": round(float(ok[cub].mean()), 4) if cub.any() else None, "ic95_filas": ic_filas(ok), "invalidas": sum(1 for y in pred if y is None),
             "lat_ms_p50": pct(ms, 50), "lat_ms_p95": pct(ms, 95)}
        if ev[0]["tipo"] == "score":
            gold = np.array([int(r["etiqueta"]) for r in ev])
            val = np.array([y is not None for y in pred])
            d["esperanza_ordinal"] = bool(esp)
            py = np.array([int(y) if y is not None else 0 for y in pred])
            err = np.abs(py - gold)
            d["exactitud_pm1"] = round(float(((err <= 1) & val).mean()), 4)
            d["mae"] = round(float(err[val].mean()), 4) if val.any() else None
        por[t] = d
    acc = [por[t]["exactitud"] for t in dev]
    sin = [por[t]["exactitud"] for t in dev if t not in sinteticas]
    out = {"tareas": por,
           "macro": {"valor": round(float(np.mean(acc)), 4), "ic95_tareas": ic_tareas(acc), "n_tareas": len(acc)},
           "macro_sin_sinteticas": {"valor": round(float(np.mean(sin)), 4), "ic95_tareas": ic_tareas(sin), "n_tareas": len(sin)},
           "cobertura_total": round(sum(d["cobertura"] * d["n_test"] for d in por.values()) / sum(d["n_test"] for d in por.values()), 4),
           "invalidas_total": sum(d["invalidas"] for d in por.values())}
    return out, oks


def cmd_medir(a):
    crit_bytes = Path(a.crit).read_bytes()
    crit = json.loads(crit_bytes)
    dev = cargar_dev(a.dev)
    if sorted(dev) != sorted(crit["tareas_dev"]):
        raise SystemExit("las tareas de %s no son las de dev del criterio: %s" % (a.dev, sorted(dev)))
    sint = set(crit["sinteticas"])
    ref = "4b-zs"
    pred = Path(a.pred)
    sistemas = [s for s in crit["sistemas"] if (pred / s).is_dir()]
    out = {"criterio_sha256": hashlib.sha256(crit_bytes).hexdigest(), "referencia": ref, "sinteticas": sorted(sint), "sistemas": {}, "pareadas": {}}
    oks = {}
    for s in sistemas:
        out["sistemas"][s], oks[s] = cifras_sistema(dev, pred / s, sint)
        l1 = pred / ("lat1-" + s)
        if l1.is_dir():   # latencia en serie (1 petición a la vez) de una submuestra
            ms = np.array([float(r["ms"]) for t in dev if (l1 / (t + ".jsonl")).exists() for r in leer(l1 / (t + ".jsonl")) if r.get("ms") is not None])
            out["sistemas"][s]["latencia_serie_ms"] = {"n": int(len(ms)), "p50": pct(ms, 50), "p95": pct(ms, 95)}
    if ref in oks:
        for s in sistemas:
            if s == ref:
                continue
            out["pareadas"][s] = par_sistema(dev, oks[s], oks[ref], sint)
    out["veredicto"] = veredicto(out, crit)
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    v = out["veredicto"]
    print("veredicto:", v["lectura"], "| delta", v.get("delta"), "| sin sinteticas", v.get("delta_sin_sinteticas"))
    for s, c in out["sistemas"].items():
        print("%-14s macro %.4f (sin sint. %.4f) cobertura %.3f invalidas %d" % (s, c["macro"]["valor"], c["macro_sin_sinteticas"]["valor"], c["cobertura_total"], c["invalidas_total"]))


def par_sistema(dev, oks_s, oks_ref, sint):
    """Diferencias pareadas sistema − referencia por tarea (bootstrap pareado por filas, McNemar) y la macro (bootstrap sobre tareas)."""
    tabla, difs, difs_sin = {}, [], []
    for t in dev:
        a1, b1 = oks_s[t], oks_ref[t]
        na, nb, p = mcnemar(a1, b1)
        d = float(a1.mean() - b1.mean())
        tabla[t] = {"dif": round(d, 4), "ic95": ic_dif_filas(a1, b1), "solo_sistema": na, "solo_referencia": nb, "mcnemar_p": p, "sintetica": t in sint}
        difs.append(d)
        if t not in sint:
            difs_sin.append(d)
    return {"tareas": tabla, "macro": {"dif": round(float(np.mean(difs)), 4), "ic95_tareas": ic_tareas(difs)},
            "macro_sin_sinteticas": {"dif": round(float(np.mean(difs_sin)), 4), "ic95_tareas": ic_tareas(difs_sin)}}


def veredicto(out, crit):
    """Δ = macro(14b-awq-zs) − macro(4b-zs); la lectura congelada decide con el valor puntual y solo si los dos cubren todo el test."""
    lec, ref, big = crit["lectura"], out["referencia"], "14b-awq-zs"
    s = out["sistemas"]
    v = {"delta_definicion": crit["delta"], "lectura": "no medido", "motivo": ""}
    if ref not in s or big not in s:
        v["motivo"] = "falta %s" % (big if ref in s else ref)
        return v
    v["cobertura"] = {ref: s[ref]["cobertura_total"], big: s[big]["cobertura_total"]}
    p = out["pareadas"][big]
    v["delta"], v["delta_ic95_tareas"] = p["macro"]["dif"], p["macro"]["ic95_tareas"]
    v["delta_sin_sinteticas"], v["delta_sin_sinteticas_ic95_tareas"] = p["macro_sin_sinteticas"]["dif"], p["macro_sin_sinteticas"]["ic95_tareas"]
    if min(v["cobertura"].values()) < 1.0:
        v["motivo"] = "cobertura incompleta: no se aplica la lectura"
        return v
    d = v["delta"]
    v["lectura"] = lec["si_ge"] if d >= lec["delta_ge"] else (lec["si_lt"] if d < lec["delta_lt"] else lec["entre"])
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filas")
    f.add_argument("--dev", required=True)
    f.add_argument("--salida", required=True)
    m = sub.add_parser("medir")
    m.add_argument("--dev", required=True)
    m.add_argument("--pred", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--crit", default=str(Path(__file__).with_name("crit-r10-t3.json")))
    a = ap.parse_args(argv)
    {"filas": cmd_filas, "medir": cmd_medir}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
