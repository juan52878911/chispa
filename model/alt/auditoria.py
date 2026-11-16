"""R10 T1: auditoría de lo ya medido (sin criterio de pasa / no pasa; correcciones y reservas). Solo numpy.

    python -m model.alt.auditoria --bateria DIR --nuevas DIR --vec DIR_VEC --fondo f1,f2 --json-fondo l3_fondo,l3_fondo2 \
        --lp-fondo lp_todo,lp_r8 --l3 lp_muestra,lp_confirma,lp_nuevas --salida auditoria.json

a) casi-duplicados test→fondo: similitud máxima de cada fila de test con el fondo de su tarea; R1 (fondo de R1, JSON) y R6b (fondo +
   fondo2, JSON frente a logprobs) repetidos sin las filas de test que tengan un vecino del fondo ≥ 0,95.
b) cascada L2→L3 con una partición FIJA por hash del id (no depende de semilla): selección de la señal y de f en la mitad A; en la
   mitad B, cascada − L3 con IC95 por bootstrap pareado (por tarea) y la curva exactitud/escalado.
c) hatecheck-es: kNN y L3 frente a la clase mayoritaria, con IC95 por bootstrap de PLANTILLAS (los casos de una plantilla son casi
   el mismo texto).
d) fakenews-es: distribución de predicciones del kNN y del LLM (¿predictor constante?).
Métricas: siempre macro por tarea y agrupada sin hatecheck-es (media de tareas, no por filas).
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

UMBRAL_DUP = 0.95


def leer_varios(r):
    return [json.loads(l) for p in str(r).split(",") if p for l in open(p, encoding="utf-8") if l.strip()]


def macro(acc, fuera=("hatecheck-es",)):
    v = [a for c, a in acc.items() if c not in fuera]
    return round(float(np.mean(v)), 4) if v else None


def mitad(i):
    return int(hashlib.sha256(("auditoria-" + i).encode()).hexdigest(), 16) % 2


def boot_par(d_por, B=2000, sem=0):
    """IC95 de la media macro (sin hatecheck) de una diferencia por fila, remuestreando dentro de cada tarea."""
    rng = np.random.RandomState(sem)
    tareas = [c for c in d_por if c != "hatecheck-es"]
    est = float(np.mean([d_por[c].mean() for c in tareas]))
    vals = sorted(float(np.mean([d_por[c][rng.randint(0, len(d_por[c]), len(d_por[c]))].mean() for c in tareas])) for _ in range(B))
    return [round(est, 4), round(vals[int(0.025 * B)], 4), round(vals[int(0.975 * B) - 1], 4)]


# ---------------------------------------------------------------- a) casi-duplicados y R1 / R6b sin ellos
def parte_a(bateria, vec, fondo, json_l3, lp_l3):
    from model.alt import fusion_knn_l2 as fk
    from model.alt import knn_soporte_llm as ks
    z = np.load(Path(vec) / "fondo.npz")
    datos = ks.cargar(bateria, vec)
    pos = {i: k for k, i in enumerate(z["ids"])}
    fo = [f for f in fondo if f["id"] in pos]
    Xf = z["X"][[pos[f["id"]] for f in fo]]
    r1_ids = {f["id"] for f in fo if f["id"].startswith("fondo:")}
    out = {"umbral": UMBRAL_DUP, "por_tarea": {}}
    variantes = {"R1": (lambda f: f["id"] in r1_ids, json_l3, None), "R6b": (lambda f: True, json_l3, json_l3 + lp_l3)}
    acc = {v: {"todas": {}, "limpias": {}} for v in ("oro", "R1_llm", "R6b_json", "R6b_lp")}
    for c, d in sorted(datos.items()):
        idx_c = [k for k, f in enumerate(fo) if f["conjunto"] == c]
        S = d["Xte"] @ Xf[idx_c].T if idx_c else np.zeros((len(d["y"]), 0))
        smax = S.max(1) if idx_c else np.zeros(len(d["y"]))
        limpia = smax < UMBRAL_DUP
        out["por_tarea"][c] = {"n_test": int(len(limpia)), "con_vecino_ge_095": int((~limpia).sum()), "con_vecino_ge_099": int((smax >= 0.99).sum()),
                               "sim_max_p50": round(float(np.median(smax)), 4), "sim_max_p95": round(float(np.quantile(smax, 0.95)), 4)}
        okoro = fk.acierto(fk.knn(d["Xte"], d["Xtr"], list(d["ytr"]), d["ops"])[0], d)
        res = {"oro": okoro}
        for nombre, (filtro, l3a, l3b) in variantes.items():
            for etiq, l3 in (("json", l3a), ("lp", l3b)):
                if l3 is None:
                    continue
                sub = [f for f in fondo if filtro(f)]
                sop = ks.soportes(sub, l3)
                par = sop.get(c, {}).get("llm", [])
                if not par:
                    continue
                ids_sub = [f["id"] for f in sub]
                Xs = z["X"][[pos[ids_sub[i]] for i, _ in par]]
                ok = fk.acierto(fk.knn(d["Xte"], Xs, [t for _, t in par], d["ops"])[0], d)
                res[{"R1json": "R1_llm", "R6bjson": "R6b_json", "R6blp": "R6b_lp"}[nombre + etiq]] = ok
        for k, ok in res.items():
            acc[k]["todas"][c] = round(float(ok.mean()), 4)
            acc[k]["limpias"][c] = round(float(ok[limpia].mean()), 4) if limpia.any() else None
    out["exactitud"] = {k: {g: {"por_tarea": v[g], "macro_sin_hatecheck": macro({c: a for c, a in v[g].items() if a is not None})} for g in v} for k, v in acc.items()}
    for g in ("todas", "limpias"):
        e = out["exactitud"]
        out.setdefault("diferencias_macro_sin_hatecheck", {})[g] = {
            "R1_llm_menos_oro": round(e["R1_llm"][g]["macro_sin_hatecheck"] - e["oro"][g]["macro_sin_hatecheck"], 4),
            "R6b_lp_menos_json": round(e["R6b_lp"][g]["macro_sin_hatecheck"] - e["R6b_json"][g]["macro_sin_hatecheck"], 4)}
    return out


# ---------------------------------------------------------------- b) cascada con partición fija
def parte_b(baterias, vec, fondo, lp_fondo, l3):
    from model.alt import escalado as es
    z = np.load(Path(vec) / "fondo.npz")
    tareas = es.cargar(baterias, vec, fondo[:len(z["ids"])], lp_fondo, l3, z)
    A = {c: np.array([k for k, i in enumerate(t["ids"]) if mitad(i) == 0], int) for c, t in tareas.items()}
    B = {c: np.array([k for k, i in enumerate(t["ids"]) if mitad(i) == 1], int) for c, t in tareas.items()}
    au = {k: float(np.mean([x for x in [es.auroc(tareas[c]["sen"][k][A[c]], tareas[c]["ok2"][A[c]]) for c in tareas if c != "hatecheck-es"] if x is not None]))
          for k in es.SENALES}
    senal = max(au, key=au.get)
    curva = []
    for f in [round(0.05 * i, 2) for i in range(0, 21)]:
        res = {}
        for nombre, P in (("A", A), ("B", B)):
            todos = np.concatenate([tareas[c]["sen"][senal][A[c]] for c in tareas if c != "hatecheck-es"])
            u = float(np.quantile(todos, f)) if f > 0 else -np.inf
            if f >= 1:
                u = np.inf
            acc, esc = {}, {}
            for c, t in tareas.items():
                i = P[c]
                e = t["sen"][senal][i] < u
                acc[c] = float(np.where(e, t["ok3"][i], t["ok2"][i]).mean())
                esc[c] = float(e.mean())
            res[nombre] = {"macro_sin_hc": macro(acc), "escalado_medio_sin_hc": round(float(np.mean([v for c, v in esc.items() if c != "hatecheck-es"])), 4)}
        curva.append({"f": f, **res})
    l3_macro = {nombre: macro({c: float(t["ok3"][P[c]].mean()) for c, t in tareas.items()}) for nombre, P in (("A", A), ("B", B))}
    l2_macro = {nombre: macro({c: float(t["ok2"][P[c]].mean()) for c, t in tareas.items()}) for nombre, P in (("A", A), ("B", B))}
    f_el = max((r for r in curva if r["f"] <= 0.30), key=lambda r: (r["A"]["macro_sin_hc"], -r["f"]))["f"]
    todos = np.concatenate([tareas[c]["sen"][senal][A[c]] for c in tareas if c != "hatecheck-es"])
    u = float(np.quantile(todos, f_el))
    d = {}
    for c, t in tareas.items():
        i = B[c]
        e = t["sen"][senal][i] < u
        d[c] = np.where(e, t["ok3"][i], t["ok2"][i]) - t["ok3"][i]
    return {"particion": "sha256('auditoria-' + id) % 2; A = selección, B = medida", "senal": senal, "auroc_A_sin_hc": {k: round(v, 4) for k, v in au.items()},
            "f": f_el, "umbral": round(u, 5), "cascada_menos_l3_B_macro_sin_hc": boot_par(d), "l3_macro_sin_hc": l3_macro, "l2_macro_sin_hc": l2_macro,
            "auroc_B_por_tarea": {c: (lambda x: None if x is None else round(x, 4))(es.auroc(t["sen"][senal][B[c]], t["ok2"][B[c]])) for c, t in tareas.items()},
            "n_A": {c: int(len(A[c])) for c in A}, "n_B": {c: int(len(B[c])) for c in B}, "curva": curva}


# ---------------------------------------------------------------- c) hatecheck por plantilla y d) fakenews
def parte_cd(baterias, vec, fondo, lp_fondo, l3, bateria_dir):
    from model.alt import escalado as es
    from model.alt.p0_medir import cargar_bateria
    z = np.load(Path(vec) / "fondo.npz")
    tareas = es.cargar(baterias, vec, fondo[:len(z["ids"])], lp_fondo, l3, z)
    out = {}
    t = tareas.get("hatecheck-es")
    if t:
        ev = {r["id"]: r for r in cargar_bateria(bateria_dir)["hatecheck-es"][1]}
        y = [ev[i]["etiqueta"] for i in t["ids"]]
        may = Counter(y).most_common(1)[0]
        okm = np.array([1.0 if v == may[0] else 0.0 for v in y])
        pl = [p or i for p, i in zip(t["plantilla"], t["ids"])]
        grupos = {}
        for k, p in enumerate(pl):
            grupos.setdefault(p, []).append(k)
        G = list(grupos.values())
        rng = np.random.RandomState(0)
        def ic(ok):
            vals = sorted(float(np.concatenate([ok[G[j]] for j in rng.randint(0, len(G), len(G))]).mean()) for _ in range(2000))
            return [round(float(ok.mean()), 4), round(vals[50], 4), round(vals[1949], 4)]
        out["hatecheck"] = {"n": len(y), "plantillas": len(G), "mayoritaria": may[0], "frac_mayoritaria": round(may[1] / len(y), 4),
                            "mayoritaria_ic_plantilla": ic(okm), "l2_knn_ic_plantilla": ic(t["ok2"]), "l3_ic_plantilla": ic(t["ok3"]),
                            "l2_menos_mayoritaria_ic_plantilla": ic(t["ok2"] - okm)}
    return out, tareas


def parte_d(tareas_nuevas, l3_nuevas, nuevas_dir):
    from model.alt import fusion_knn_l2 as fk
    from model.alt.llm_techo import top1
    from model.alt.p0_medir import cargar_bateria
    out = {}
    bat = cargar_bateria(nuevas_dir)
    if "fakenews-es" not in bat:
        return out
    _, ev = bat["fakenews-es"]
    res = {r["id"]: r for r in l3_nuevas}
    pred = [top1(res[r["id"]]["probs"], r["opciones"], r["tipo"]) if r["id"] in res and res[r["id"]].get("valida") else None for r in ev]
    y = [r["etiqueta"] for r in ev]
    out["fakenews"] = {"n": len(y), "clases": dict(Counter(y)), "llm_predicciones": dict(Counter(map(str, pred))),
                       "llm_acierto": round(sum(1 for p, v in zip(pred, y) if p == v) / len(y), 4)}
    t = tareas_nuevas.get("fakenews-es")
    if t:
        out["fakenews"]["oro_en_soporte"] = t["oro"]
        out["fakenews"]["fondo_en_soporte"] = t["fondo"]
        out["fakenews"]["l2_acierto"] = round(float(t["ok2"].mean()), 4)
        out["fakenews"]["l2_pmax_valores_distintos"] = int(len(np.unique(np.round(t["sen"]["pmax"], 6))))
        out["fakenews"]["l2_pmax_mediana"] = round(float(np.median(t["sen"]["pmax"])), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--bateria", "--nuevas", "--vec", "--fondo", "--json-fondo", "--lp-fondo", "--l3", "--salida"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    fondo = leer_varios(a.fondo)
    json_l3, lp_l3 = leer_varios(a.json_fondo), leer_varios(a.lp_fondo)
    l3 = leer_varios(a.l3)
    out = {"nota": "auditoría sin criterio; macro por tarea y agrupada sin hatecheck-es"}
    out["a_casi_duplicados"] = parte_a(a.bateria, a.vec, fondo, json_l3, lp_l3)
    print("a OK", json.dumps(out["a_casi_duplicados"]["diferencias_macro_sin_hatecheck"]), flush=True)
    out["b_cascada_particion_fija"] = parte_b([a.bateria], a.vec, fondo, json_l3 + lp_l3, l3)
    print("b OK", out["b_cascada_particion_fija"]["cascada_menos_l3_B_macro_sin_hc"], flush=True)
    c, _ = parte_cd([a.bateria], a.vec, fondo, json_l3 + lp_l3, l3, a.bateria)
    out["c_hatecheck"] = c
    print("c OK", json.dumps(c)[:300], flush=True)
    _, tn = parte_cd([a.nuevas], a.vec, fondo, json_l3 + lp_l3, l3, a.nuevas)
    out["d_fakenews"] = parte_d(tn, l3, a.nuevas)
    print("d OK", json.dumps(out["d_fakenews"]), flush=True)
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
