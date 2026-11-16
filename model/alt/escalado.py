"""R8, punto 1: regla de escalado L2→L3. ¿Las señales del kNN predicen cuándo falla L2, y una cascada que escala ≤ 30 % queda cerca de L3?

    python -m model.alt.escalado medir --bateria DIR[,DIR2] --vec DIR_VEC --fondo f1.jsonl[,f2.jsonl] --l3-fondo a.jsonl[,b.jsonl,...] \
        --l3 l3_test.jsonl[,...] --salida informe.json [--semillas 10]
    python -m model.alt.escalado criterio

L2 (fijado de antemano): kNN-e5 (32 vecinos, temperatura 0,05, solo opciones de la fila) sobre el banco oro (split train de la batería)
∪ fondo con etiqueta del LLM (en --l3-fondo el último registro de un id manda: JSON de R1/R4 debajo, logprobs de R7/R7b/R8 encima), con
la regla «umbral»: si el ejemplo de oro más cercano tiene similitud ≥ S_ORO = 0,90 se vota solo con el oro.
L3: el 4B por logprobs (--l3: R5 en la muestra del paso 2 y R7 fuera de ella). Solo cuentan las filas de test que tienen L3.
Señales de confianza de L2 (más alto = más seguro): pmax, margen (p1 − p2), −entropía normalizada, similitud al oro más cercano y
similitud al vecino más cercano. Por semilla s se parten las filas de cada tarea al 50 % (calibración / validación).
Selección (semillas 0–4, mitades de calibración): la señal con mejor AUROC medio y la fracción de escalado f ∈ {0,05 … 0,30} que da la
mejor exactitud de la cascada; el umbral es el cuantil f de esa señal en la calibración (global, el mismo para todas las tareas).
Validación (semillas 5–9, mitades de validación, con señal y f congeladas; el umbral se recalcula como cuantil f en la calibración de esa
semilla): AUROC de la señal para el acierto de L2, fracción escalada real, exactitud de L2, L3 y la cascada.
Exactitud agrupada = media ponderada por el tamaño de cada tarea en su batería (como el paso 2), sobre las filas con L3.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

VECINOS, TEMP, S_ORO = 32, 0.05, 0.90
FRACS = [round(0.05 * i, 2) for i in range(1, 7)]
SENALES = ("pmax", "margen", "neg_entropia", "sim_oro", "sim_vecino")
CRIT8E = {"l2": "kNN oro ∪ fondo-LLM (logprobs donde haya), regla umbral s_oro 0.90, 32 vecinos, T 0.05", "senales": list(SENALES),
          "seleccion": "semillas 0-4, mitad de calibración: mejor AUROC medio y f en 0.05..0.30 con mejor cascada",
          "validacion": "semillas 5-9, mitad de validación", "escalado_max": 0.30, "pasa_auroc_min": 0.70, "pasa_dist_l3_max": 0.02,
          "refutado_si_auroc_lt": 0.60, "agrupada": "ponderada por tamaño de tarea, filas con L3"}
CRIT8E_SHA = hashlib.sha256(json.dumps(CRIT8E, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
# R9: regla por tarea. Selección en las semillas 0–9 (las de R8, ya vistas): tareas cuyo margen tiene AUROC < 0,55 (media de las mitades de
# calibración) se escalan siempre; en el resto, margen con la f ∈ 0,05..0,30 de mejor cascada. Validación SOLO en las semillas 10–19,
# nunca usadas. Mismos umbrales que CRIT8E; `tope` dice cómo se cuenta el ≤ 30 %.
CRIT9E = dict(CRIT8E, regla="por tarea: AUROC del margen < 0.55 en la selección -> escalar siempre; resto por margen con f global",
              auroc_siempre_lt=0.55, seleccion="semillas 0-9, mitad de calibración", validacion="semillas 10-19 (nuevas), mitad de validación",
              tope="__TOPE__", escalado_total_max=0.50)
def crit9(tope):
    c = dict(CRIT9E, tope=tope)
    return c, hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def varios(rutas):
    return [r for p in str(rutas).split(",") if p for r in leer(p)]


def etiquetas(filas, registros):
    """{id: top-1 o None (inválida)} para las filas con registro; el último registro de un id manda."""
    from model.alt.llm_techo import top1
    res = {}
    for r in registros:
        res[r["id"]] = r
    out = {}
    for f in filas:
        r = res.get(f["id"])
        if r is not None:
            out[f["id"]] = top1(r["probs"], f["opciones"], f.get("tipo", "noul")) if r.get("valida") and r.get("probs") else None
    return out


def l2(Xte, ops, Xg, yg, Xf, yf):
    """(P (n,m), señales {nombre: (n,)}) del kNN con la regla umbral."""
    n, m = len(Xte), max(len(o) for o in ops)
    Xb = np.vstack([Xg, Xf]) if len(Xf) else Xg
    yb = list(yg) + list(yf)
    G = len(yg)
    S = Xte @ Xb.T if len(yb) else np.zeros((n, 0))
    P = np.zeros((n, m))
    sen = {k: np.zeros(n) for k in SENALES}
    for i in range(n):
        s = S[i]
        so = float(s[:G].max()) if G else -1.0
        cand = np.arange(G) if (G and so >= S_ORO) else np.arange(len(yb))
        p = dict.fromkeys(ops[i], 0.0)
        if len(cand):
            sc = s[cand]
            k = cand[np.argsort(-sc, kind="stable")[:VECINOS]]
            w = np.exp((s[k] - s[k].max()) / TEMP)
            for j, wj in zip(k, w):
                if yb[j] in p:
                    p[yb[j]] += float(wj)
        tot = sum(p.values())
        v = np.array([p[o] / tot for o in ops[i]]) if tot > 0 else np.full(len(ops[i]), 1.0 / len(ops[i]))
        P[i, :len(v)] = v
        q = np.sort(v)[::-1]
        sen["pmax"][i] = q[0]
        sen["margen"][i] = q[0] - (q[1] if len(q) > 1 else 0.0)
        ent = -(v[v > 0] * np.log(v[v > 0])).sum() / max(np.log(len(v)), 1e-12)
        sen["neg_entropia"][i] = -ent
        sen["sim_oro"][i] = so
        sen["sim_vecino"][i] = float(s.max()) if len(s) else -1.0
    return P, sen


def cargar(baterias, vec, fondo, l3_fondo, l3, z):
    from model.alt.p0_medir import cargar_bateria
    sop_ev = {}
    for d in baterias:
        sop_ev.update(cargar_bateria(d))
    fo_et = etiquetas(fondo, l3_fondo)
    pos = {i: k for k, i in enumerate(z["ids"])}
    tareas = {}
    for c in sorted(sop_ev):
        sop, ev = sop_ev[c]
        zz = np.load(Path(vec) / (c + ".npz"))
        if list(zz["idte"]) != [r["id"] for r in ev]:
            raise ValueError("%s: los vectores no corresponden a las filas" % c)
        et3 = etiquetas(ev, l3)
        sel = [k for k, r in enumerate(ev) if r["id"] in et3]
        if not sel:
            continue
        fs = [f for f in fondo if f["conjunto"] == c and fo_et.get(f["id"]) is not None and f["id"] in pos]
        Xf = z["X"][[pos[f["id"]] for f in fs]] if fs else np.zeros((0, zz["Xte"].shape[1]))
        Xg = zz["Xtr"] if len(sop) else np.zeros((0, zz["Xte"].shape[1]))
        ops = [ev[k]["opciones"] for k in sel]
        P, sen = l2(zz["Xte"][sel], ops, Xg, [r["etiqueta"] for r in sop], Xf, [fo_et[f["id"]] for f in fs])
        y = [ev[k]["etiqueta"] for k in sel]
        ok2 = np.array([1.0 if ops[i][int(np.argmax(P[i, :len(ops[i])]))] == y[i] else 0.0 for i in range(len(sel))])
        ok3 = np.array([1.0 if et3[ev[k]["id"]] == ev[k]["etiqueta"] else 0.0 for k in sel])
        tareas[c] = {"ok2": ok2, "ok3": ok3, "sen": sen, "peso": len(ev), "n": len(sel), "oro": len(sop), "fondo": len(fs),
                     "ids": [ev[k]["id"] for k in sel], "plantilla": [ev[k].get("plantilla") for k in sel]}
    return tareas


def auroc(conf, ok):
    from model.alt.p0_medir import auroc as a
    r = a(list(map(float, conf)), list(map(float, ok)))
    return None if r is None else float(r)


def ponderada(vals, pesos):
    return float(sum(pesos[c] * v for c, v in vals.items()) / sum(pesos[c] for c in vals))


def mitades(tareas, semilla):
    out = {}
    for c, t in tareas.items():
        idx = list(range(t["n"]))
        random.Random("escalado-%s-%d" % (c, semilla)).shuffle(idx)
        h = len(idx) // 2
        out[c] = (np.array(sorted(idx[:h]), dtype=int), np.array(sorted(idx[h:]), dtype=int))
    return out


def cascada(tareas, partes, senal, umbral):
    """{tarea: (exactitud cascada, L2, L3, fracción escalada)} en las filas `partes`."""
    out = {}
    for c, t in tareas.items():
        i = partes[c]
        if not len(i):
            continue
        esc = t["sen"][senal][i] < umbral
        okc = np.where(esc, t["ok3"][i], t["ok2"][i])
        out[c] = (float(okc.mean()), float(t["ok2"][i].mean()), float(t["ok3"][i].mean()), float(esc.mean()))
    return out


def aplicar(tareas, senal, f, semillas=10):
    """La regla ya congelada (señal y fracción) en todas las semillas: umbral = cuantil f en la mitad de calibración, medida en la otra."""
    pesos = {c: t["peso"] for c, t in tareas.items()}
    res = []
    for s in range(semillas):
        cal, val = (dict((c, v[j]) for c, v in mitades(tareas, s).items()) for j in (0, 1))
        u = float(np.quantile(np.concatenate([tareas[c]["sen"][senal][cal[c]] for c in tareas]), f))
        cv = cascada(tareas, val, senal, u)
        res.append({"semilla": s, "cascada": ponderada({c: v[0] for c, v in cv.items()}, pesos), "l2": ponderada({c: v[1] for c, v in cv.items()}, pesos),
                    "l3": ponderada({c: v[2] for c, v in cv.items()}, pesos),
                    "escalado": float(np.average([v[3] for v in cv.values()], weights=[len(val[c]) for c in cv])),
                    "auroc": auroc(np.concatenate([tareas[c]["sen"][senal][val[c]] for c in tareas]), np.concatenate([tareas[c]["ok2"][val[c]] for c in tareas])),
                    "por_tarea": {c: {"cascada": round(v[0], 4), "l2": round(v[1], 4), "l3": round(v[2], 4), "escalado": round(v[3], 4)} for c, v in cv.items()}})
    m = {k: round(float(np.mean([r[k] for r in res if r[k] is not None])), 4) for k in ("cascada", "l2", "l3", "escalado", "auroc")}
    m["cascada_menos_l3"] = round(m["cascada"] - m["l3"], 4)
    pt = {c: {k: round(float(np.mean([r["por_tarea"][c][k] for r in res])), 4) for k in ("cascada", "l2", "l3", "escalado")} for c in tareas}
    return {"regla": {"senal": senal, "fraccion": f}, "resumen": m, "por_tarea": pt, "semillas": res}


def cascada_t(tareas, partes, umbral, siempre):
    out = {}
    for c, t in tareas.items():
        i = partes[c]
        if not len(i):
            continue
        esc = np.ones(len(i), bool) if c in siempre else t["sen"]["margen"][i] < umbral
        okc = np.where(esc, t["ok3"][i], t["ok2"][i])
        out[c] = (float(okc.mean()), float(t["ok2"][i].mean()), float(t["ok3"][i].mean()), float(esc.mean()))
    return out


def medir_por_tarea(tareas, tope):
    """R9 (CRIT9E): elige tareas «siempre» y f en las semillas 0–9 y valida en las 10–19."""
    crit, sha = crit9(tope)
    pesos = {c: t["peso"] for c, t in tareas.items()}
    sel_s, val_s = range(0, 10), range(10, 20)
    au = {c: [] for c in tareas}
    for s in sel_s:
        for c, (cal, _) in mitades(tareas, s).items():
            au[c].append(auroc(tareas[c]["sen"]["margen"][cal], tareas[c]["ok2"][cal]))
    au_m = {c: (float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else 0.5) for c, v in au.items()}
    siempre = sorted(c for c, a in au_m.items() if a < crit["auroc_siempre_lt"])
    resto = [c for c in tareas if c not in siempre]
    def umbral(cal, f):
        v = np.concatenate([tareas[c]["sen"]["margen"][cal[c]] for c in resto]) if resto else np.zeros(1)
        return float(np.quantile(v, f))
    casc = {f: [] for f in FRACS}
    for s in sel_s:
        cal = {c: v[0] for c, v in mitades(tareas, s).items()}
        for f in FRACS:
            casc[f].append(ponderada({c: v[0] for c, v in cascada_t(tareas, cal, umbral(cal, f), siempre).items()}, pesos))
    f_el = max(FRACS, key=lambda f: (np.mean(casc[f]), -f))
    res = []
    for s in val_s:
        cal, val = (dict((c, v[j]) for c, v in mitades(tareas, s).items()) for j in (0, 1))
        cv = cascada_t(tareas, val, umbral(cal, f_el), siempre)
        n_val = {c: len(val[c]) for c in cv}
        esc_todas = float(np.average([v[3] for v in cv.values()], weights=[n_val[c] for c in cv]))
        esc_resto = float(np.average([cv[c][3] for c in resto], weights=[n_val[c] for c in resto])) if resto else 0.0
        okr = np.concatenate([tareas[c]["ok2"][val[c]] for c in resto]) if resto else np.zeros(0)
        res.append({"semilla": s, "cascada": ponderada({c: v[0] for c, v in cv.items()}, pesos), "l2": ponderada({c: v[1] for c, v in cv.items()}, pesos),
                    "l3": ponderada({c: v[2] for c, v in cv.items()}, pesos), "escalado_todas": esc_todas, "escalado_resto": esc_resto,
                    "auroc_resto": auroc(np.concatenate([tareas[c]["sen"]["margen"][val[c]] for c in resto]), okr) if resto else None,
                    "por_tarea": {c: {"cascada": round(v[0], 4), "l2": round(v[1], 4), "l3": round(v[2], 4), "escalado": round(v[3], 4)} for c, v in cv.items()}})
    m = {k: round(float(np.mean([r[k] for r in res if r[k] is not None])), 4) for k in ("cascada", "l2", "l3", "escalado_todas", "escalado_resto", "auroc_resto")}
    m["cascada_menos_l3"] = round(m["cascada"] - m["l3"], 4)
    esc = m["escalado_resto"] if tope == "sin_siempre" else m["escalado_todas"]
    v = {"criterio_sha256": sha, "tope": tope, "siempre": siempre, "fraccion": f_el, **m}
    v["c_auroc"] = bool(m["auroc_resto"] >= crit["pasa_auroc_min"])
    v["c_cascada"] = bool(esc <= crit["escalado_max"] + 1e-9 and m["escalado_todas"] <= crit["escalado_total_max"] + 1e-9
                          and m["cascada_menos_l3"] >= -crit["pasa_dist_l3_max"])
    v["estado"] = "REFUTADO" if m["auroc_resto"] < crit["refutado_si_auroc_lt"] else ("PASA" if v["c_auroc"] and v["c_cascada"] else "NO PASA")
    return {"criterio": crit, "seleccion": {"auroc_margen_por_tarea": {c: round(a, 4) for c, a in au_m.items()},
                                            "cascada_media": {"%.2f" % f: round(float(np.mean(x)), 4) for f, x in casc.items()}},
            "validacion": res, "veredicto": v}


def medir(tareas, semillas=10):
    pesos = {c: t["peso"] for c, t in tareas.items()}
    todos = lambda i_por, k: np.concatenate([tareas[c]["sen"][k][i_por[c]] for c in tareas])  # noqa: E731
    sel_s = [s for s in range(semillas) if s < semillas // 2]
    val_s = [s for s in range(semillas) if s >= semillas // 2]
    # selección
    au = {k: [] for k in SENALES}
    casc = {(k, f): [] for k in SENALES for f in FRACS}
    for s in sel_s:
        cal = {c: v[0] for c, v in mitades(tareas, s).items()}
        ok2 = np.concatenate([tareas[c]["ok2"][cal[c]] for c in tareas])
        for k in SENALES:
            au[k].append(auroc(todos(cal, k), ok2))
            for f in FRACS:
                u = float(np.quantile(todos(cal, k), f))
                casc[(k, f)].append(ponderada({c: v[0] for c, v in cascada(tareas, cal, k, u).items()}, pesos))
    au_m = {k: (float(np.mean([x for x in au[k] if x is not None])) if any(x is not None for x in au[k]) else 0.5) for k in SENALES}
    senal = max(SENALES, key=lambda k: au_m[k])
    f_el = max(FRACS, key=lambda f: (np.mean(casc[(senal, f)]), -f))
    # validación
    res = []
    for s in val_s:
        cal, val = (dict((c, v[j]) for c, v in mitades(tareas, s).items()) for j in (0, 1))
        u = float(np.quantile(todos(cal, senal), f_el))
        cv = cascada(tareas, val, senal, u)
        ok2v = np.concatenate([tareas[c]["ok2"][val[c]] for c in tareas])
        res.append({"semilla": s, "umbral": round(u, 5), "auroc": auroc(todos(val, senal), ok2v),
                    "auroc_por_tarea": {c: auroc(tareas[c]["sen"][senal][val[c]], tareas[c]["ok2"][val[c]]) for c in tareas},
                    "cascada": ponderada({c: v[0] for c, v in cv.items()}, pesos), "l2": ponderada({c: v[1] for c, v in cv.items()}, pesos),
                    "l3": ponderada({c: v[2] for c, v in cv.items()}, pesos),
                    "escalado": float(np.average([v[3] for v in cv.values()], weights=[len(val[c]) for c in cv])),
                    "por_tarea": {c: {"cascada": round(v[0], 4), "l2": round(v[1], 4), "l3": round(v[2], 4), "escalado": round(v[3], 4)} for c, v in cv.items()}})
    def m(k):
        v = [r[k] for r in res if r[k] is not None]
        return float(np.mean(v)) if v else float("nan")
    resumen = {k: round(m(k), 4) for k in ("auroc", "cascada", "l2", "l3", "escalado")}
    resumen["cascada_menos_l3"] = round(resumen["cascada"] - resumen["l3"], 4)
    v = {"criterio_sha256": CRIT8E_SHA, "senal": senal, "fraccion": f_el, **resumen}
    v["c_auroc"] = bool(resumen["auroc"] >= CRIT8E["pasa_auroc_min"])   # nan (L2 nunca falla) no pasa
    v["c_cascada"] = bool(resumen["escalado"] <= CRIT8E["escalado_max"] + 1e-9 and resumen["cascada_menos_l3"] >= -CRIT8E["pasa_dist_l3_max"])
    v["estado"] = "REFUTADO" if resumen["auroc"] < CRIT8E["refutado_si_auroc_lt"] else ("PASA" if v["c_auroc"] and v["c_cascada"] else "NO PASA")
    return {"seleccion": {"auroc_medio": {k: round(x, 4) for k, x in au_m.items()},
                          "cascada_media": {"%s@%.2f" % kf: round(float(np.mean(x)), 4) for kf, x in casc.items()}},
            "validacion": res, "veredicto": v,
            "tareas": {c: {"n": t["n"], "oro": t["oro"], "fondo": t["fondo"], "l2": round(float(t["ok2"].mean()), 4), "l3": round(float(t["ok3"].mean()), 4)} for c, t in tareas.items()}}


def cmd_medir(a):
    z = np.load(Path(a.vec) / "fondo.npz")
    fondo = varios(a.fondo)[:len(z["ids"])]
    tareas = cargar([d for d in a.bateria.split(",") if d], a.vec, fondo, varios(a.l3_fondo), varios(a.l3), z)
    out = {"criterio": CRIT8E, "criterio_sha256": CRIT8E_SHA, "semillas": a.semillas,
           "transformers_vectores": str(z["transformers"]) if "transformers" in z.files else None}
    if a.por_tarea:
        out.update(medir_por_tarea(tareas, a.por_tarea))
        Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(out["veredicto"], ensure_ascii=False, indent=1))
        return
    if a.regla:
        senal, f = a.regla.split(":")
        out["aplicada"] = aplicar(tareas, senal, float(f), a.semillas)
        out["tareas"] = {c: {"n": t["n"], "oro": t["oro"], "fondo": t["fondo"], "l2": round(float(t["ok2"].mean()), 4), "l3": round(float(t["ok3"].mean()), 4)} for c, t in tareas.items()}
        Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(out["aplicada"]["resumen"], ensure_ascii=False))
        return
    out.update(medir(tareas, a.semillas))
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["veredicto"], ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("medir")
    for k in ("--bateria", "--vec", "--fondo", "--l3-fondo", "--l3", "--salida"):
        m.add_argument(k, required=True)
    m.add_argument("--semillas", type=int, default=10)
    m.add_argument("--regla", help="señal:fracción ya congelada (R8b): se aplica sin seleccionar")
    m.add_argument("--por-tarea", choices=("todas", "sin_siempre"), help="R9 (CRIT9E): regla por tarea; cómo se cuenta el tope del 30 %")
    c = sp.add_parser("criterio")
    c.add_argument("--r9", choices=("todas", "sin_siempre"))
    a = ap.parse_args(argv)
    if a.cmd == "criterio":
        if a.r9:
            c9, s9 = crit9(a.r9)
            print(s9, json.dumps(c9, sort_keys=True, ensure_ascii=False))
            return
        print(CRIT8E_SHA, json.dumps(CRIT8E, sort_keys=True, ensure_ascii=False))
        return
    cmd_medir(a)


if __name__ == "__main__":
    sys.exit(main())
