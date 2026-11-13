"""Paso 1: fusión kNN-e5 + L2 y escalado por acuerdo, sobre la batería no vista (6 tareas), a coste 0.

    python -m model.alt.fusion_knn_l2 vectores --bateria DIR --salida DIR_VEC          (torch, e5 congelado; una sola vez)
    python -m model.alt.fusion_knn_l2 medir --bateria DIR --pred DIR_PRED --vec DIR_VEC --salida informe.json [--semillas 10] [--fusion loglineal|lineal]

`vectores` guarda DIR_VEC/<tarea>.npz (Xtr, Xte, idtr, idte): e5 multilingual-e5-small congelado, media de tokens, «query: », 256 tokens.
`medir` solo usa numpy: reconstruye el kNN (k ejemplos por clase del split train, 32 vecinos, temperatura 0,05, como
`model.l2.linea_base_knn_e5.probs_knn`) con distancias, fusiona con L2 y ajusta TODO (peso de la fusión, k, escalado) dejando una tarea fuera.

Fusión (diseño 1, log-lineal): p ∝ exp(w·log max(p_knn, EPS) + (1−w)·log max(p_l2, EPS)). `lineal` (variante): p = w·p_knn + (1−w)·p_l2.
w se elige en la rejilla 0..1 (paso 0,05) por la exactitud media por tarea en las otras 5 tareas.
k se elige, para cada tarea fuera, por la exactitud media de la fusión en las otras 5 tareas.
Escalado: regresión logística (ridge, rasgos estandarizados, tareas con el mismo peso) que predice el acierto de la fusión con
[acuerdo top-1 kNN/L2, prob. máxima, margen, entropía normalizada, similitud al vecino más cercano, similitud al centroide de la clase predicha].
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EPS = 1e-3
VECINOS = 32
TEMP = 0.05
REJILLA_W = [round(0.05 * i, 2) for i in range(21)]
KS = (1, 2, 4, 8, 16)
RASGOS = ("acuerdo", "pmax", "margen", "entropia", "sim_vecino", "sim_centroide")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        14: 2.145, 19: 2.093, 29: 2.045}


# ---------------------------------------------------------------- vectores (torch)
def cmd_vectores(a):
    import torch
    torch.set_num_threads(a.hilos)
    from model.alt.p0_medir import cargar_bateria
    from model.l2.linea_base_knn_e5 import vectores
    from model.l2.modelo import Codificador
    enc = Codificador(device="cpu", max_tok=256)
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    for c, (sop, ev) in cargar_bateria(a.bateria).items():
        Xtr = vectores(enc, [r["texto"] for r in sop]).numpy()
        Xte = vectores(enc, [r["texto"] for r in ev]).numpy()
        np.savez(Path(a.salida) / (c + ".npz"), Xtr=Xtr, Xte=Xte, idtr=np.array([r["id"] for r in sop]), idte=np.array([r["id"] for r in ev]))
        print(c, Xtr.shape, Xte.shape, flush=True)


# ---------------------------------------------------------------- kNN con distancias (numpy)
def muestrear(ys, k, semilla):
    """Índices del soporte: k por clase al azar (todos los de la clase si tiene menos). k=None: todo el soporte."""
    if k is None:
        return np.arange(len(ys))
    rng = np.random.RandomState(semilla)
    por = {}
    for i, y in enumerate(ys):
        por.setdefault(y, []).append(i)
    idx = []
    for y in sorted(por):
        v = por[y]
        idx += [int(i) for i in rng.permutation(v)[:k]] if len(v) > k else v
    return np.array(sorted(idx), dtype=int)


def knn(Xte, Xsop, ysop, opciones, vecinos=VECINOS, temp=TEMP):
    """(P (n,m) con relleno 0, sim_vecino (n,), sim_centroide por opción (n,m)). Misma regla que `probs_knn`: los `vecinos` más
    cercanos por coseno, pesos exp((s−smax)/temp), votos a la etiqueta, solo opciones de la fila; sin votos: uniforme."""
    n, m = len(Xte), max(len(o) for o in opciones)
    S = Xte @ Xsop.T if len(ysop) else np.zeros((n, 0))
    P = np.zeros((n, m))
    simv = np.zeros(n)
    etiquetas = sorted(set(ysop))
    cent = np.zeros((len(etiquetas), Xte.shape[1]))
    for j, y in enumerate(etiquetas):
        v = Xsop[[i for i, t in enumerate(ysop) if t == y]].mean(0)
        cent[j] = v / max(np.linalg.norm(v), 1e-12)
    SC = Xte @ cent.T if len(etiquetas) else np.zeros((n, 0))
    col = {y: j for j, y in enumerate(etiquetas)}
    C = np.zeros((n, m))
    for i, ops in enumerate(opciones):
        if len(ysop):
            s = S[i]
            idx = np.argsort(-s, kind="stable")[:vecinos]
            w = np.exp((s[idx] - s[idx].max()) / temp)
            simv[i] = float(s[idx[0]])
            p = dict.fromkeys(ops, 0.0)
            for j, wj in zip(idx, w):
                if ysop[j] in p:
                    p[ysop[j]] += float(wj)
            tot = sum(p.values())
            P[i, :len(ops)] = [p[o] / tot for o in ops] if tot > 0 else 1.0 / len(ops)
            C[i, :len(ops)] = [SC[i, col[o]] if o in col else 0.0 for o in ops]
        else:
            P[i, :len(ops)] = 1.0 / len(ops)
    return P, simv, C


# ---------------------------------------------------------------- fusión y rasgos
def rellenar(preds, ids, opciones, nombre):
    """Matriz (n,m) de probabilidades en el orden de `opciones`; cada id tiene que estar en `preds`."""
    falta = [i for i in ids if i not in preds]
    if falta:
        raise ValueError("%s no cubre %d de %d ids" % (nombre, len(falta), len(ids)))
    m = max(len(o) for o in opciones)
    P = np.zeros((len(ids), m))
    for i, (id_, ops) in enumerate(zip(ids, opciones)):
        P[i, :len(ops)] = [float(preds[id_].get(o, 0.0)) for o in ops]
        if P[i].sum() <= 0:
            P[i, :len(ops)] = 1.0 / len(ops)
    return P


def mascara(opciones):
    m = max(len(o) for o in opciones)
    return np.array([[j < len(o) for j in range(m)] for o in opciones])


def fusionar(Pk, Pl, M, w, modo="loglineal"):
    if modo == "lineal":
        P = np.where(M, w * Pk + (1 - w) * Pl, 0.0)
    else:
        s = w * np.log(np.maximum(Pk, EPS)) + (1 - w) * np.log(np.maximum(Pl, EPS))
        s = np.where(M, s, -np.inf)
        s = s - s.max(1, keepdims=True)
        P = np.exp(s)
    return P / P.sum(1, keepdims=True)


def top1(P, M):
    return np.argmax(np.where(M, P, -np.inf), axis=1)


def rasgos(Pf, Pk, Pl, M, simv, C):
    """(n,6): ver RASGOS. La entropía va normalizada por log(nº de opciones de la fila)."""
    n = len(Pf)
    t = top1(Pf, M)
    ordenado = -np.sort(-np.where(M, Pf, -1.0), axis=1)
    pmax = ordenado[:, 0]
    margen = pmax - (ordenado[:, 1] if Pf.shape[1] > 1 else 0.0)
    k = M.sum(1)
    ent = -(np.where(Pf > 0, Pf * np.log(np.where(Pf > 0, Pf, 1.0)), 0.0)).sum(1) / np.log(np.maximum(k, 2))
    acuerdo = (top1(Pk, M) == top1(Pl, M)).astype(float)
    return np.column_stack([acuerdo, pmax, margen, ent, simv, C[np.arange(n), t]])


# ---------------------------------------------------------------- regresión logística (ridge, Newton)
def ajustar_lr(X, y, pesos=None, lam=1.0, iters=50):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    pesos = np.ones(len(X)) if pesos is None else pesos * len(X) / pesos.sum()
    b = np.zeros(Z.shape[1])
    reg = lam * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ b, -30, 30)))
        g = Z.T @ (pesos * (p - y)) + reg @ b
        H = (Z * (pesos * p * (1 - p))[:, None]).T @ Z + reg + 1e-9 * np.eye(len(b))
        paso = np.linalg.solve(H, g)
        b -= paso
        if np.abs(paso).max() < 1e-8:
            break
    return mu, sd, b


def predecir_lr(modelo, X):
    mu, sd, b = modelo
    return 1.0 / (1.0 + np.exp(-np.clip(np.column_stack([(X - mu) / sd, np.ones(len(X))]) @ b, -30, 30)))


# ---------------------------------------------------------------- estadística
def media_ic(v):
    """(media, mitad del IC95 t entre semillas)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), 0.0
    df = n - 1
    t = T975.get(df) or T975[min((d for d in T975 if d >= df), default=29)]
    return float(v.mean()), float(t * v.std(ddof=1) / math.sqrt(n))


def auc(conf, ok):
    from model.alt.p0_medir import auroc
    a = auroc(conf, ok)
    return float("nan") if a is None else a


# ---------------------------------------------------------------- carga
def cargar(bateria, pred, vec, modelo_l2="l2", modelo_ce="ce-fp32-onnx"):
    """{tarea: dict}; falla si las predicciones no cubren todo el test."""
    from model.alt.p0_medir import cargar_bateria, leer
    out = {}
    for c, (sop, ev) in cargar_bateria(bateria).items():
        z = np.load(Path(vec) / (c + ".npz"))
        if list(z["idte"]) != [r["id"] for r in ev] or list(z["idtr"]) != [r["id"] for r in sop]:
            raise ValueError("%s: los vectores no corresponden a las filas" % c)
        ops = [r["opciones"] for r in ev]
        ids = [r["id"] for r in ev]
        d = {"ids": ids, "ops": ops, "M": mascara(ops), "Xtr": z["Xtr"], "Xte": z["Xte"], "ytr": [r["etiqueta"] for r in sop]}
        d["y"] = np.array([o.index(r["etiqueta"]) for o, r in zip(ops, ev)])
        for clave, mod in (("Pl", modelo_l2), ("Pc", modelo_ce), ("Pk_guardado", "knn")):
            ps = {p["id"]: p["probs"] for p in leer(Path(pred) / mod / (c + ".jsonl"))}
            d[clave] = rellenar(ps, ids, ops, "%s/%s" % (mod, c))
        out[c] = d
    return out


def acierto(P, d):
    return (top1(P, d["M"]) == d["y"]).astype(float)


# ---------------------------------------------------------------- medición
def configuracion(datos, k, semilla):
    out = {}
    for c, d in datos.items():
        idx = muestrear(d["ytr"], k, semilla)
        out[c] = knn(d["Xte"], d["Xtr"][idx], [d["ytr"][i] for i in idx], d["ops"])
    return out


def medir(datos, semillas=10, modo="loglineal", ks=KS, verbose=True):
    tareas = sorted(datos)
    ver = {}
    cfg_todo = configuracion(datos, None, 0)
    tot = n = 0
    for c in tareas:
        a_new, a_old = acierto(cfg_todo[c][0], datos[c]), acierto(datos[c]["Pk_guardado"], datos[c])
        ver[c] = {"acc_recalculado": round(float(a_new.mean()), 4), "acc_guardado": round(float(a_old.mean()), 4),
                  "difieren_top1": int((top1(cfg_todo[c][0], datos[c]["M"]) != top1(datos[c]["Pk_guardado"], datos[c]["M"])).sum()),
                  "max_dif_prob": round(float(np.abs(cfg_todo[c][0] - datos[c]["Pk_guardado"]).max()), 5)}
        tot += a_new.sum()
        n += len(a_new)
    ver["agrupada_recalculada"] = round(float(tot / n), 4)
    if verbose:
        print("verificación kNN todo el soporte:", json.dumps(ver), flush=True)

    claves = list(ks) + ["todo"]
    ex = {}
    for k in claves:
        for s in ([0] if k == "todo" else range(semillas)):
            cfg = cfg_todo if k == "todo" else configuracion(datos, k, s)
            e = {}
            for c in tareas:
                d = datos[c]
                Pk, simv, C = cfg[c]
                e[c] = {"Pk": Pk, "simv": simv, "C": C,
                        "acc_w": np.array([acierto(fusionar(Pk, d["Pl"], d["M"], w, modo), d).mean() for w in REJILLA_W])}
            ex[(k, s)] = e
        if verbose:
            print("k", k, flush=True)

    def mejor_w(e, fuera):
        return int(np.argmax(np.mean([e[c]["acc_w"] for c in tareas if c != fuera], axis=0)))

    por_k = {}
    for k in claves:
        por_k[k] = {c: {"knn": [], "l2": [], "ce": [], "fus": [], "w": [], "ok_knn": [], "ok_fus": [], "auc_fus": [], "auc_pmax": [],
                        "conf_fus": [], "conf_pmax": []} for c in tareas}
        for s in ([0] if k == "todo" else range(semillas)):
            e = ex[(k, s)]
            F = {}
            for c in tareas:
                d = datos[c]
                w = REJILLA_W[mejor_w(e, c)]
                Pf = fusionar(e[c]["Pk"], d["Pl"], d["M"], w, modo)
                F[c] = (w, Pf, rasgos(Pf, e[c]["Pk"], d["Pl"], d["M"], e[c]["simv"], e[c]["C"]))
            for c in tareas:
                d = datos[c]
                w, Pf, Xr = F[c]
                okf, okk = acierto(Pf, d), acierto(e[c]["Pk"], d)
                r = por_k[k][c]
                r["knn"].append(float(okk.mean()))
                r["l2"].append(float(acierto(d["Pl"], d).mean()))
                r["ce"].append(float(acierto(d["Pc"], d).mean()))
                r["fus"].append(float(okf.mean()))
                r["w"].append(w)
                r["ok_knn"].append(okk)
                r["ok_fus"].append(okf)
                otras = [o for o in tareas if o != c]
                lr = ajustar_lr(np.vstack([F[o][2] for o in otras]), np.concatenate([acierto(F[o][1], datos[o]) for o in otras]),
                                np.concatenate([np.full(len(datos[o]["y"]), 1.0 / len(datos[o]["y"])) for o in otras]))
                conf = predecir_lr(lr, Xr)
                r["auc_fus"].append(auc(conf, okf))
                r["auc_pmax"].append(auc(Xr[:, 1], okf))
                r["conf_fus"].append(conf)
                r["conf_pmax"].append(Xr[:, 1])

    # k elegido por tarea fuera: exactitud media de la fusión (w ajustado en las otras 5) en las otras 5 tareas
    elegido = {}
    for c in tareas:
        otras = [o for o in tareas if o != c]
        puntaje = {}
        for k in claves:
            v = []
            for s in ([0] if k == "todo" else range(semillas)):
                e = ex[(k, s)]
                j = mejor_w(e, c)
                v.append(np.mean([e[o]["acc_w"][j] for o in otras]))
            puntaje[k] = float(np.mean(v))
        mejor = max(puntaje.values())
        elegido[c] = [k for k in claves if puntaje[k] >= mejor - 1e-12][-1]   # empate: el de más soporte
    return {"verificacion_knn": ver, "tareas": tareas, "modo": modo, "semillas": semillas, "k_elegido": elegido, "_por_k": por_k}


# ---------------------------------------------------------------- informe (JSON de cifras)
def _r(par):
    return {"media": round(par[0], 4), "ic95": round(par[1], 4)}


def resumen(res, datos, ic, ic_dif, mcnemar):
    por_k, tareas = res["_por_k"], res["tareas"]
    n_t = {c: len(datos[c]["y"]) for c in tareas}
    N = sum(n_t.values())
    out = {"verificacion_knn": res["verificacion_knn"], "modo": res["modo"], "semillas": res["semillas"], "n_test": n_t, "k_elegido": res["k_elegido"]}
    curva = {}
    for k, pt in por_k.items():
        fila = {c: {"knn": _r(media_ic(pt[c]["knn"])), "fusion": _r(media_ic(pt[c]["fus"])), "w": round(float(np.mean(pt[c]["w"])), 3)} for c in tareas}
        ns = len(pt[tareas[0]]["knn"])
        for m, nom in (("knn", "agrupada_knn"), ("fus", "agrupada_fusion")):
            fila[nom] = _r(media_ic([sum(pt[c][m][i] * n_t[c] for c in tareas) / N for i in range(ns)]))
        curva[str(k)] = fila
    out["curva_k"] = curva

    tabla, pareado, auroc_t, sel = {}, {"fusion_vs_knn": {}, "fusion_vs_ce": {}}, {}, {}
    for c in tareas:
        d = datos[c]
        pt = por_k[res["k_elegido"][c]][c]
        okk, okf = np.mean(pt["ok_knn"], axis=0), np.mean(pt["ok_fus"], axis=0)
        a_l2, a_ce = acierto(d["Pl"], d), acierto(d["Pc"], d)
        sel[c] = (okk, okf, a_l2, a_ce, pt["ok_knn"][0], pt["ok_fus"][0])
        mejor = max(float(okk.mean()), float(a_l2.mean()))
        tabla[c] = {"k": res["k_elegido"][c], "knn": ic(okk), "l2": ic(a_l2), "ce": ic(a_ce), "fusion": ic(okf), "w": round(float(np.mean(pt["w"])), 3),
                    "mejor_knn_l2": round(mejor, 4), "fusion_menos_mejor": round(float(okf.mean()) - mejor, 4)}
        for nombre, ref in (("fusion_vs_knn", sel[c][4]), ("fusion_vs_ce", a_ce)):
            n01, n10, p = mcnemar(list(sel[c][5]), list(ref))
            pareado[nombre][c] = {"solo_fusion": n01, "solo_otro": n10, "p": round(p, 6), **ic_dif(list(sel[c][5]), list(ref))}
        auroc_t[c] = {"escalado": round(float(np.nanmean(pt["auc_fus"])), 4), "pmax": round(float(np.nanmean(pt["auc_pmax"])), 4)}
    junto = lambda i: np.concatenate([sel[c][i] for c in tareas])
    tabla["agrupada"] = {"knn": ic(junto(0)), "l2": ic(junto(2)), "ce": ic(junto(3)), "fusion": ic(junto(1))}
    for nombre, ref in (("fusion_vs_knn", junto(4)), ("fusion_vs_ce", junto(3))):
        n01, n10, p = mcnemar(list(junto(5)), list(ref))
        pareado[nombre]["agrupada"] = {"solo_fusion": n01, "solo_otro": n10, "p": round(p, 6), **ic_dif(list(junto(5)), list(ref))}
    out["tabla"], out["mcnemar"] = tabla, pareado

    aucs, aucp, aucs_h, aucp_h = [], [], [], []
    for s in range(res["semillas"]):
        cf, cp, ok = [], [], []
        for c in tareas:
            k = res["k_elegido"][c]
            i = 0 if k == "todo" else s
            cf.append(por_k[k][c]["conf_fus"][i])
            cp.append(por_k[k][c]["conf_pmax"][i])
            ok.append(por_k[k][c]["ok_fus"][i])
            if c == "hatecheck-es":
                aucs_h.append(por_k[k][c]["auc_fus"][i])
                aucp_h.append(por_k[k][c]["auc_pmax"][i])
        aucs.append(auc(np.concatenate(cf), np.concatenate(ok)))
        aucp.append(auc(np.concatenate(cp), np.concatenate(ok)))
    out["auroc"] = {"agrupado": {"escalado": _r(media_ic(aucs)), "pmax": _r(media_ic(aucp))},
                    "hatecheck-es": {"escalado": _r(media_ic(aucs_h)), "pmax": _r(media_ic(aucp_h))}, "por_tarea": auroc_t}
    return out


def veredicto(out):
    ag = out["tabla"]["agrupada"]["fusion"]["acc"]
    peor = min(out["tabla"][c]["fusion_menos_mejor"] for c in out["tabla"] if c != "agrupada")
    a_ag = out["auroc"]["agrupado"]["escalado"]["media"]
    a_h = out["auroc"]["hatecheck-es"]["escalado"]["media"]
    pasa = ag >= 0.60 and peor >= -0.01 and a_ag >= 0.78 and a_h >= 0.70
    refutado = ag <= 0.578 or a_h < 0.65   # «no supera a kNN solo (0,578)»
    return {"fusion_agrupada": ag, "peor_tarea_frente_al_mejor": peor, "auroc_agrupado": a_ag, "auroc_hatecheck": a_h,
            "veredicto": "PASA" if pasa else ("REFUTADO" if refutado else "no concluyente")}


def cmd_medir(a):
    from model.alt.p0_medir import ic, ic_dif, mcnemar
    datos = cargar(a.bateria, a.pred, a.vec)
    out = resumen(medir(datos, a.semillas, a.fusion), datos, ic, ic_dif, mcnemar)
    out["veredicto"] = veredicto(out)
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out["veredicto"]), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("vectores")
    v.add_argument("--bateria", required=True)
    v.add_argument("--salida", required=True)
    v.add_argument("--hilos", type=int, default=6)
    m = sp.add_parser("medir")
    m.add_argument("--bateria", required=True)
    m.add_argument("--pred", required=True)
    m.add_argument("--vec", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--semillas", type=int, default=10)
    m.add_argument("--fusion", choices=("loglineal", "lineal"), default="loglineal")
    a = ap.parse_args(argv)
    {"vectores": cmd_vectores, "medir": cmd_medir}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
