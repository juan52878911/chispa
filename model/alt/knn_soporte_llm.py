"""Paso 3 (R1): kNN-e5 con soporte etiquetado por el LLM (Qwen3-4B bf16) sobre la batería no vista, frente al kNN con soporte de oro.

    python -m model.alt.knn_soporte_llm vectores --bateria DIR --fondo fondo.jsonl --salida DIR_VEC [--hilos 8]     (torch, e5 congelado)
    python -m model.alt.knn_soporte_llm medir --bateria DIR --vec DIR_VEC --fondo fondo.jsonl --l3 l3_fondo.jsonl \
        [--fuente fondo-fuente.jsonl] [--vec-oro DIR_VEC_PASO1] --salida informe.json [--semillas 10]

`vectores` escribe DIR_VEC/<tarea>.npz (Xtr, Xte, idtr, idte; igual que `fusion_knn_l2 vectores`) y DIR_VEC/fondo.npz (X, ids).
`medir` (solo numpy) reconstruye el kNN del paso 1 (32 vecinos, temperatura 0,05, solo opciones de la fila) con varios soportes:
  oro       el split train de la batería (≤ 8 por clase), k = 1, 2, 4, 8 y todo;
  llm       los textos del fondo con respuesta válida del LLM, etiqueta = top-1 del LLM; k = 1 … 128 por clase y todo;
  llm_p09   lo mismo, solo con probabilidad máxima ≥ 0,9;
  oro+llm   la unión de oro y llm (todo);
  fuente    los mismos textos del fondo con la etiqueta de su fuente (lo que daría el mismo soporte sin ruido del LLM).
El criterio (CRIT, congelado; su sha256 va en el LEEME antes de medir) compara llm/todo con oro/todo, los dos con los vectores de
esta misma máquina. Exactitud agrupada = aciertos / casos sobre todo el test de las 6 tareas (como el 0,580 del paso 1).
Las etiquetas del LLM son salidas de Qwen3-4B (Apache-2.0); nada de Jev.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

CRIT = {"variante": "llm/todo", "referencia": "oro/todo", "perdida_max": 0.03, "agrupada_min": 0.55, "hatecheck_min": 0.60,
        "refutado_si_agrupada_le": 0.516}
CRIT_SHA = hashlib.sha256(json.dumps(CRIT, sort_keys=True).encode()).hexdigest()
KS_ORO = (1, 2, 4, 8)
KS_LLM = (1, 2, 4, 8, 16, 32, 64, 128)
KS_FUENTE = (8, 32, 128)


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------- vectores (torch)
def cmd_vectores(a):
    import torch
    torch.set_num_threads(a.hilos)
    from model.alt.p0_medir import cargar_bateria
    from model.l2.linea_base_knn_e5 import vectores
    from model.l2.modelo import Codificador
    enc = Codificador(device="cpu", max_tok=256)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    for c, (sop, ev) in ({} if a.sin_bateria else cargar_bateria(a.bateria)).items():
        Xtr = vectores(enc, [r["texto"] for r in sop]).numpy()
        Xte = vectores(enc, [r["texto"] for r in ev]).numpy()
        np.savez(sal / (c + ".npz"), Xtr=Xtr, Xte=Xte, idtr=np.array([r["id"] for r in sop]), idte=np.array([r["id"] for r in ev]))
        print(c, Xtr.shape, Xte.shape, flush=True)
    fo = leer(a.fondo)[:a.max_fondo]
    X = vectores(enc, [r["texto"] for r in fo]).numpy()
    np.savez(sal / "fondo.npz", X=X, ids=np.array([r["id"] for r in fo]))
    print("fondo", X.shape, flush=True)


# ---------------------------------------------------------------- soportes
def soportes(fondo, l3, fuente=None):
    """{conjunto: {"llm": [(i, etiqueta)], "llm_p09": [...], "fuente": [...]}} con i = posición en fondo.jsonl."""
    from model.alt.llm_techo import top1
    res = {r["id"]: r for r in l3}
    fu = {r["id"]: r["etiqueta"] for r in (fuente or [])}
    out = {}
    for i, f in enumerate(fondo):
        d = out.setdefault(f["conjunto"], {"llm": [], "llm_p09": [], "fuente": [], "en_fondo": 0})
        d["en_fondo"] += 1
        if f["id"] in fu:
            d["fuente"].append((i, fu[f["id"]]))
        r = res.get(f["id"])
        if not r or not r.get("valida") or not r.get("probs"):
            continue
        t = top1(r["probs"], f["opciones"], f["tipo"])
        if t is None:
            continue
        d["llm"].append((i, t))
        if max(r["probs"].values()) >= 0.9:
            d["llm_p09"].append((i, t))
    return out


def por_clase(ys):
    v = {}
    for y in ys:
        v[y] = v.get(y, 0) + 1
    return v


def media_ic(v):
    from model.alt.fusion_knn_l2 import media_ic as m
    return m(v)


def boot_dif(a, b, por, remuestreos=2000, semilla=0):
    """Diferencia agrupada (a − b) con IC95 pareado, remuestreando casos dentro de cada tarea."""
    rng = random.Random(semilla)
    d = {c: [x - y for x, y in zip(a[c], b[c])] for c in por}
    n = sum(len(v) for v in d.values())
    tot = lambda dd: sum(sum(v) for v in dd.values()) / n  # noqa: E731
    vals = sorted(tot({c: [v[rng.randrange(len(v))] for _ in v] for c, v in d.items()}) for _ in range(remuestreos))
    return [round(tot(d), 4), round(vals[int(0.025 * remuestreos)], 4), round(vals[int(0.975 * remuestreos) - 1], 4)]


# ---------------------------------------------------------------- medición
def medir(datos, Xf, sop, semillas=10):
    from model.alt import fusion_knn_l2 as fk
    from model.alt.p0_medir import mcnemar
    tareas = sorted(datos)
    n_tot = sum(len(datos[c]["y"]) for c in tareas)

    def correr(nombre, k, s):
        """{tarea: vector de aciertos}, {tarea: tamaño del soporte por clase}"""
        oks, tam = {}, {}
        for c in tareas:
            d = datos[c]
            if nombre == "oro":
                X, ys = d["Xtr"], list(d["ytr"])
            elif nombre == "oro+llm":
                idx = [i for i, _ in sop[c]["llm"]]
                X, ys = np.vstack([d["Xtr"], Xf[idx]]), list(d["ytr"]) + [t for _, t in sop[c]["llm"]]
            else:
                par = sop[c][nombre]
                X, ys = Xf[[i for i, _ in par]], [t for _, t in par]
            sel = fk.muestrear(ys, k, s)
            X, ys = X[sel] if len(sel) else X[:0], [ys[i] for i in sel]
            P, _, _ = fk.knn(d["Xte"], X, ys, d["ops"])
            oks[c] = fk.acierto(P, d)
            tam[c] = por_clase(ys)
        return oks, tam

    curvas, ok_todo = {}, {}
    plan = [("oro", KS_ORO), ("llm", KS_LLM), ("llm_p09", ()), ("oro+llm", ()), ("fuente", KS_FUENTE)]
    for nombre, ks in plan:
        if nombre == "fuente" and not all(sop[c]["fuente"] for c in tareas):
            continue
        filas = {}
        for k in list(ks) + [None]:
            agr, pt, tams = [], {c: [] for c in tareas}, {c: [] for c in tareas}
            for s in (range(semillas) if k is not None else [0]):
                oks, tam = correr(nombre, k, s)
                agr.append(float(sum(v.sum() for v in oks.values()) / n_tot))
                for c in tareas:
                    pt[c].append(float(oks[c].mean()))
                    tams[c].append(sum(tam[c].values()) / max(len(tam[c]), 1))
                if k is None:
                    ok_todo[nombre] = {c: [float(x) for x in oks[c]] for c in tareas}
            m, ic = media_ic(agr)
            filas["todo" if k is None else str(k)] = {
                "agrupada": [round(m, 4), round(ic, 4)],
                "por_tarea": {c: round(float(np.mean(pt[c])), 4) for c in tareas},
                "soporte_medio_por_clase": {c: round(float(np.mean(tams[c])), 1) for c in tareas}}
            print(nombre, k, filas["todo" if k is None else str(k)]["agrupada"], flush=True)
        curvas[nombre] = filas

    a, b = ok_todo["llm"], ok_todo["oro"]
    comp = {"llm_menos_oro": boot_dif(a, b, tareas),
            "mcnemar": mcnemar([x for c in tareas for x in a[c]], [x for c in tareas for x in b[c]]),
            "por_tarea": {c: {"llm": round(float(np.mean(a[c])), 4), "oro": round(float(np.mean(b[c])), 4),
                              "llm_menos_oro": boot_dif({c: a[c]}, {c: b[c]}, [c]), "mcnemar": mcnemar(a[c], b[c])} for c in tareas}}
    if "fuente" in ok_todo:
        comp["llm_menos_fuente"] = boot_dif(a, ok_todo["fuente"], tareas)
    return {"curvas": curvas, "comparacion": comp, "n_test": {c: int(len(datos[c]["y"])) for c in tareas},
            "soporte": {c: {"en_fondo": sop[c]["en_fondo"], "llm": len(sop[c]["llm"]), "llm_p09": len(sop[c]["llm_p09"]),
                            "fuente": len(sop[c]["fuente"]), "clases_llm": len(por_clase([t for _, t in sop[c]["llm"]])),
                            "acuerdo_llm_fuente": acuerdo(sop[c])} for c in tareas}}


def acuerdo(s):
    fu = dict(s["fuente"])
    v = [1.0 if fu[i] == t else 0.0 for i, t in s["llm"] if i in fu]
    return round(sum(v) / len(v), 4) if v else None


def veredicto(out):
    llm = out["curvas"]["llm"]["todo"]
    agr, hc = llm["agrupada"][0], llm["por_tarea"].get("hatecheck-es")
    dif = out["comparacion"]["llm_menos_oro"][0]
    v = {"criterio_sha256": CRIT_SHA, "agrupada_llm": agr, "agrupada_oro": out["curvas"]["oro"]["todo"]["agrupada"][0],
         "llm_menos_oro": out["comparacion"]["llm_menos_oro"], "hatecheck_llm": hc,
         "c_perdida": dif >= -CRIT["perdida_max"], "c_agrupada": agr >= CRIT["agrupada_min"], "c_hatecheck": hc is not None and hc >= CRIT["hatecheck_min"]}
    v["refutado"] = agr <= CRIT["refutado_si_agrupada_le"]
    v["estado"] = "REFUTADO" if v["refutado"] else ("PASA" if v["c_perdida"] and v["c_agrupada"] and v["c_hatecheck"] else "NO PASA")
    return v


def cargar(bateria, vec):
    from model.alt import fusion_knn_l2 as fk
    from model.alt.p0_medir import cargar_bateria
    out = {}
    for c, (sop, ev) in cargar_bateria(bateria).items():
        z = np.load(Path(vec) / (c + ".npz"))
        if list(z["idte"]) != [r["id"] for r in ev] or list(z["idtr"]) != [r["id"] for r in sop]:
            raise ValueError("%s: los vectores no corresponden a las filas" % c)
        ops = [r["opciones"] for r in ev]
        out[c] = {"ids": [r["id"] for r in ev], "ops": ops, "M": fk.mascara(ops), "Xtr": z["Xtr"], "Xte": z["Xte"],
                  "ytr": [r["etiqueta"] for r in sop], "y": np.array([o.index(r["etiqueta"]) for o, r in zip(ops, ev)])}
    return out


def cmd_medir(a):
    z = np.load(Path(a.vec) / "fondo.npz")
    fondo = leer(a.fondo)[:len(z["ids"])]   # una prueba en seco vectoriza solo el principio del fondo
    if list(z["ids"]) != [r["id"] for r in fondo]:
        raise ValueError("fondo.npz no corresponde a fondo.jsonl")
    datos = cargar(a.bateria, a.vec)
    sop = soportes(fondo, leer(a.l3), leer(a.fuente) if a.fuente else None)
    faltan = [c for c in datos if c not in sop or not sop[c]["llm"]]
    if faltan:
        raise ValueError("sin soporte del LLM en: %s" % faltan)
    out = {"criterios": CRIT, "criterio_sha256": CRIT_SHA, "semillas": a.semillas}
    if a.vec_oro:   # misma batería con los vectores del paso 1 (otra máquina): cuánto cambian
        d1 = cargar(a.bateria, a.vec_oro)
        out["vectores_frente_a_paso1"] = {c: round(float(max(np.abs(datos[c]["Xte"] - d1[c]["Xte"]).max(), np.abs(datos[c]["Xtr"] - d1[c]["Xtr"]).max())), 6) for c in datos}
    out.update(medir(datos, z["X"], sop, a.semillas))
    out["veredicto"] = veredicto(out)
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["veredicto"], ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("vectores")
    v.add_argument("--bateria", required=True)
    v.add_argument("--fondo", required=True)
    v.add_argument("--salida", required=True)
    v.add_argument("--hilos", type=int, default=8)
    v.add_argument("--max-fondo", type=int, help="solo los primeros N textos del fondo (prueba en seco)")
    v.add_argument("--sin-bateria", action="store_true", help="solo el fondo (prueba en seco con los vectores del paso 1)")
    m = sp.add_parser("medir")
    m.add_argument("--bateria", required=True)
    m.add_argument("--vec", required=True)
    m.add_argument("--fondo", required=True)
    m.add_argument("--l3", required=True)
    m.add_argument("--fuente")
    m.add_argument("--vec-oro")
    m.add_argument("--salida", required=True)
    m.add_argument("--semillas", type=int, default=10)
    sp.add_parser("criterio")
    a = ap.parse_args(argv)
    if a.cmd == "criterio":
        print(CRIT_SHA, json.dumps(CRIT, sort_keys=True))
        return
    {"vectores": cmd_vectores, "medir": cmd_medir}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
