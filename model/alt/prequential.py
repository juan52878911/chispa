"""Paso 4 (R6): simulación prequential de «L3 asíncrono con banco de prototipos».

    python -m model.alt.prequential medir --bateria DIR[,DIR2] --vec DIR_VEC --l3 a.jsonl[,b.jsonl] --fondo fondo.jsonl \
        --l3-fondo l3_fondo.jsonl --vec-fondo DIR_VEC/fondo.npz --salida informe.json [--semillas 10] [--retraso 50]
    python -m model.alt.prequential criterio

Solo numpy. Por tarea y semilla, el flujo son las filas de test que tienen registro del LLM (ficheros --l3), en orden aleatorio.
En el paso t se predice la fila t con kNN sobre el banco actual (32 vecinos, temperatura 0,05, coseno sobre e5 normalizado, solo las
opciones de la fila; igual que `fusion_knn_l2.knn`) y se anota el acierto contra la etiqueta de oro; después entra al banco la
etiqueta del LLM de la fila t−N (N = --retraso, 50 por defecto) si fue válida. La etiqueta de la fila r solo se usa desde el
paso r+N+1; nunca antes de r+N. Una tarea sin filas train (fakenews-es) empieza con el banco vacío: sin votos el kNN es uniforme y
elige la primera opción (predictor constante; hay que leer esa cifra como «sin nada»).

Variantes (mismo flujo y mismas semillas):
  oro_estatico          (1) solo oro (el split train; ≤ 8 por clase), nunca cambia.
  oro_llm_estatico      (2) oro ∪ fondo-LLM (todo el fondo de la tarea con etiqueta válida del LLM), pesos iguales, como el 0,656 de R1.
  preq_oro              (3) oro + etiquetas diferidas del flujo.
  preq_oro_fondo        (4) oro ∪ fondo-LLM + etiquetas diferidas del flujo.
  descriptivas (no entran en el criterio): oro_llm_estatico_peso (2 con la regla del oro), preq_oro_umbral y preq_oro_fondo_umbral
  (3 y 4 con la regla alternativa).

Regla «el oro manda» (elegida, fijada antes de ver ningún resultado): voto ponderado. Cada vecino aporta exp((s−smax)/T) multiplicado
por su peso de fuente: 1 si es oro, ALFA = 0,5 si viene del LLM (fondo o flujo). Motivo: es continua, no tiene umbral que ajustar
sobre un espacio de similitudes que cambia de una tarea a otra, y el oro sigue pesando más sin vetar al LLM cuando el oro está lejos
(donde R1 vio que el LLM ayuda: 0,638 frente a 0,578); un umbral s* duro dependería de la escala de similitud de cada tarea.
Con ALFA < 1 un vecino de oro y uno del LLM igual de cercanos votan 2:1 a favor del oro. La regla alternativa («umbral»): si el
oro más cercano de todo el banco tiene similitud ≥ S_ORO = 0,90 se vota solo con oro; si no, todos con peso 1. ALFA y S_ORO se fijaron
a priori y no se ajustan al test. En (2) los pesos son iguales a propósito, para que (2) sea la referencia R1 sin la regla.

Salida (solo cifras): por variante, por tarea y agrupado (aciertos/casos del flujo, media e IC95 t entre semillas), curva por
décimas del flujo, y diferencias pareadas (bootstrap de casos dentro de cada tarea, agrupado) frente a (1) y (2).
Las etiquetas del LLM son salidas de Qwen3-4B (Apache-2.0); nada de Jev.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

VECINOS = 32
TEMP = 0.05
ALFA = 0.5       # peso de un ejemplo del LLM frente a 1 del oro (regla «peso», la elegida)
S_ORO = 0.90     # similitud de oro a partir de la cual la regla «umbral» (descriptiva) usa solo oro
RETRASO = 50
TRAMOS = 10
NUEVAS = ("hatecheck-es", "lexboe", "fakenews-es")
SIN_ORO = ("fakenews-es",)

# variante: (usa_fondo, dinamica, regla, en_criterio)
VARIANTES = {
    "oro_estatico": (False, False, "igual", True),
    "oro_llm_estatico": (True, False, "igual", True),
    "preq_oro": (False, True, "peso", True),
    "preq_oro_fondo": (True, True, "peso", True),
    "oro_llm_estatico_peso": (True, False, "peso", False),
    "preq_oro_umbral": (False, True, "umbral", False),
    "preq_oro_fondo_umbral": (True, True, "umbral", False),
}

CRIT_R6 = {"variante": "preq_oro_fondo", "ref_nuevas": "oro_estatico", "ref_originales": "oro_llm_estatico",
           "nuevas": list(NUEVAS), "sube_min": 0.05, "perdida_max_original": 0.02, "refutado_si_dif_le": 0.0,
           "alfa": ALFA, "retraso": RETRASO, "vecinos": VECINOS, "temp": TEMP, "regla": "peso"}
CRIT_R6_SHA = hashlib.sha256(json.dumps(CRIT_R6, sort_keys=True).encode()).hexdigest()


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _lista(s):
    return [x for x in str(s).split(",") if x]


def media_ic(v):
    from model.alt.fusion_knn_l2 import media_ic as m
    return m(v)


# ---------------------------------------------------------------- etiquetas del LLM
def etiquetas_llm(filas, l3):
    """{id: etiqueta top-1 del LLM o None si no fue válida} solo para las filas con registro en l3."""
    from model.alt.llm_techo import top1
    res = {}
    for r in l3:
        res[r["id"]] = r   # el último registro de un id manda
    out = {}
    for f in filas:
        r = res.get(f["id"])
        if r is None:
            continue
        t = None
        if r.get("valida") and r.get("probs"):
            t = top1(r["probs"], f["opciones"], f.get("tipo", "noul"))
        out[f["id"]] = t
    return out


# ---------------------------------------------------------------- kNN incremental sobre un banco = índices de un pozo
def votar(s, y, w, n_oro, ops_lut, n_ops, regla):
    """Probabilidades (n_ops,) por voto ponderado de los `VECINOS` más cercanos del banco.
    s: similitudes de la fila con los n elementos del banco; y: códigos de etiqueta (n,); w: peso de fuente (n,);
    los primeros n_oro elementos del banco son de oro. Sin banco o sin votos: uniforme."""
    n = len(s)
    if n == 0:
        return np.full(n_ops, 1.0 / n_ops)
    ss, yy, ww = s, y, w
    if regla == "umbral" and n_oro and s[:n_oro].max() >= S_ORO:
        ss, yy, ww = s[:n_oro], y[:n_oro], w[:n_oro]
    k = min(VECINOS, len(ss))
    idx = np.argpartition(-ss, k - 1)[:k] if len(ss) > k else np.arange(len(ss))
    sv = ss[idx]
    pv = np.exp((sv - sv.max()) / TEMP) * ww[idx]
    c = ops_lut[yy[idx]]
    m = c >= 0
    v = np.bincount(c[m], weights=pv[m], minlength=n_ops) if m.any() else np.zeros(n_ops)
    tot = v.sum()
    return v / tot if tot > 0 else np.full(n_ops, 1.0 / n_ops)


class Tarea:
    """Pozo común de una tarea: [oro | fondo | flujo] con vectores, códigos de etiqueta y similitudes precalculadas."""

    def __init__(self, nombre, Xoro, yoro, Xfondo, yfondo, Xflujo, ops_flujo, y_flujo, llm_flujo, n_test):
        self.nombre, self.n_test = nombre, n_test
        self.G, self.F, self.n = len(yoro), len(yfondo), len(y_flujo)
        etiquetas = sorted({*yoro, *yfondo, *(t for t in llm_flujo if t is not None), *(o for ops in ops_flujo for o in ops)})
        self.cod = {e: i for i, e in enumerate(etiquetas)}
        self.ops = ops_flujo
        self.y = np.asarray(y_flujo, dtype=int)                       # índice de la opción correcta en cada fila
        self.ypozo = np.array([self.cod[e] for e in yoro] + [self.cod[e] for e in yfondo]
                              + [self.cod[t] if t is not None else -1 for t in llm_flujo], dtype=int)
        self.llm_ok = np.array([t is not None for t in llm_flujo])
        self.llm_acierta = np.array([1.0 if t is not None and ops.index(t) == yi else 0.0 for t, ops, yi in zip(llm_flujo, ops_flujo, self.y)])
        self.luts = []
        for ops in ops_flujo:
            lut = np.full(len(etiquetas), -1, dtype=int)
            for j, o in enumerate(ops):
                lut[self.cod[o]] = j
            self.luts.append(lut)
        pozo = np.vstack([a for a in (Xoro, Xfondo, Xflujo) if len(a)]).astype(np.float32)
        self.S = (Xflujo.astype(np.float32) @ pozo.T) if self.n else np.zeros((0, len(pozo)), dtype=np.float32)

    def correr(self, orden, usa_fondo, dinamica, regla, retraso):
        """Aciertos (0/1) por fila del flujo, indexados por posición original (no por posición en `orden`)."""
        G, F, n = self.G, self.F, self.n
        base = G + F if usa_fondo else G
        cap = base + (n if dinamica else 0)
        idx = np.empty(cap, dtype=int)
        idx[:G] = np.arange(G)
        if usa_fondo:
            idx[G:G + F] = np.arange(G, G + F)
        cnt = base
        wp = np.ones(G + F + n) if regla != "peso" else np.concatenate([np.ones(G), np.full(F + n, ALFA)])
        ok = np.zeros(n)
        for t, i in enumerate(orden):
            b = idx[:cnt]
            P = votar(self.S[i, b], self.ypozo[b], wp[b], G, self.luts[i], len(self.ops[i]), regla)
            ok[i] = 1.0 if int(np.argmax(P)) == self.y[i] else 0.0
            if dinamica and t - retraso >= 0:      # la etiqueta de la fila t−N entra tras predecir la fila t
                r = orden[t - retraso]
                if self.llm_ok[r]:
                    idx[cnt] = G + F + r
                    cnt += 1
        return ok


# ---------------------------------------------------------------- datos
def cargar_tareas(baterias, vec, l3, fondo, l3_fondo, z_fondo):
    from model.alt.p0_medir import cargar_bateria
    sop_ev = {}
    for d in baterias:
        sop_ev.update(cargar_bateria(d))
    llm = {}
    for f in l3:
        llm.update({r["id"]: r for r in leer(f)})
    llm = list(llm.values())
    fo_et = etiquetas_llm(fondo, l3_fondo)
    pos = {i: k for k, i in enumerate(z_fondo["ids"])}
    if any(f["id"] not in pos for f in fondo if f["id"] in fo_et):
        raise ValueError("fondo.npz no cubre el fondo con etiqueta del LLM")
    tareas = {}
    for c in sorted(sop_ev):
        sop, ev = sop_ev[c]
        z = np.load(Path(vec) / (c + ".npz"))
        if list(z["idte"]) != [r["id"] for r in ev] or list(z["idtr"]) != [r["id"] for r in sop]:
            raise ValueError("%s: los vectores no corresponden a las filas" % c)
        et = etiquetas_llm(ev, llm)
        sel = [k for k, r in enumerate(ev) if r["id"] in et]
        if not sel:
            raise ValueError("%s: ninguna fila de test tiene registro del LLM" % c)
        fs = [f for f in fondo if f["conjunto"] == c and fo_et.get(f["id"]) is not None]
        Xf = z_fondo["X"][[pos[f["id"]] for f in fs]] if fs else np.zeros((0, z["Xte"].shape[1]), dtype=np.float32)
        Xoro = z["Xtr"] if len(sop) else np.zeros((0, z["Xte"].shape[1]), dtype=np.float32)
        tareas[c] = Tarea(c, Xoro, [r["etiqueta"] for r in sop], Xf, [fo_et[f["id"]] for f in fs], z["Xte"][sel],
                          [ev[k]["opciones"] for k in sel], [ev[k]["opciones"].index(ev[k]["etiqueta"]) for k in sel],
                          [et[ev[k]["id"]] for k in sel], len(ev))
    return tareas


# ---------------------------------------------------------------- medición
def grupos(tareas):
    nombres = list(tareas)
    return {"todas": nombres, "nuevas": [c for c in nombres if c in CRIT_R6["nuevas"]],
            "originales": [c for c in nombres if c not in ("lexboe", "fakenews-es")]}


def boot_dif(a, b, nombres, remuestreos=2000, semilla=0):
    """Diferencia agrupada (a − b) por caso con IC95 percentil: remuestrea filas dentro de cada tarea. a, b: {tarea: vector por fila}."""
    nombres = [c for c in nombres if c in a]
    if not nombres:
        return None
    rng = np.random.RandomState(semilla)
    d = [np.asarray(a[c], dtype=float) - np.asarray(b[c], dtype=float) for c in nombres]
    n = sum(len(v) for v in d)
    tot = np.zeros(remuestreos)
    for v in d:
        tot += v[rng.randint(0, len(v), (remuestreos, len(v)))].sum(1)
    tot /= n
    return [round(float(sum(v.sum() for v in d) / n), 4), round(float(np.percentile(tot, 2.5)), 4), round(float(np.percentile(tot, 97.5)), 4)]


def medir(tareas, semillas=10, retraso=RETRASO, verbose=True):
    """Devuelve el informe. ok[v][c] = (semillas, n) aciertos por fila; ord[c] = (semillas, n) orden del flujo."""
    ordenes = {c: [np.random.RandomState(s).permutation(t.n) for s in range(semillas)] for c, t in tareas.items()}
    ok = {}
    for v, (fondo, dinamica, regla, _) in VARIANTES.items():
        ok[v] = {}
        for c, t in tareas.items():
            if not dinamica:   # estático: el acierto por fila no depende del orden
                fila = t.correr(np.arange(t.n), fondo, False, regla, retraso)
                ok[v][c] = np.tile(fila, (semillas, 1))
            else:
                ok[v][c] = np.array([t.correr(o, fondo, True, regla, retraso) for o in ordenes[c]])
        if verbose:
            print(v, {c: round(float(ok[v][c].mean()), 3) for c in tareas}, flush=True)
    gr = grupos(tareas)
    casos = {c: t.n for c, t in tareas.items()}

    def agr(v, nombres):   # exactitud agrupada por semilla
        return np.array([sum(ok[v][c][s].sum() for c in nombres) / max(sum(casos[c] for c in nombres), 1) for s in range(semillas)])

    def curva(v, nombres):
        out = []
        for k in range(TRAMOS):
            por_s = []
            for s in range(semillas):
                aci = cas = 0
                for c in nombres:
                    seg = np.array_split(ordenes[c][s], TRAMOS)[k]
                    aci += ok[v][c][s][seg].sum()
                    cas += len(seg)
                por_s.append(aci / max(cas, 1))
            out.append(round(float(np.mean(por_s)), 4))
        return out

    informe = {"variantes": {}, "diferencias": {}}
    for v in VARIANTES:
        info = {"en_criterio": VARIANTES[v][3], "por_tarea": {}, "agrupada": {}, "curva": {"por_tarea": {}}}
        for c in tareas:
            m, h = media_ic(ok[v][c].mean(1))
            info["por_tarea"][c] = {"exactitud": [round(m, 4), round(h, 4)], "aciertos_medios": round(float(ok[v][c].sum(1).mean()), 1), "casos": casos[c]}
            info["curva"]["por_tarea"][c] = curva(v, [c])
        for g, nombres in gr.items():
            if nombres:
                m, h = media_ic(agr(v, nombres))
                info["agrupada"][g] = {"exactitud": [round(m, 4), round(h, 4)], "casos": sum(casos[c] for c in nombres)}
                info["curva"][g] = curva(v, nombres)
        informe["variantes"][v] = info
    fila = {v: {c: ok[v][c].mean(0) for c in tareas} for v in VARIANTES}   # exactitud por fila, media entre semillas
    for v in VARIANTES:
        d = {}
        for ref in ("oro_estatico", "oro_llm_estatico"):
            if v == ref:
                continue
            e = {g: boot_dif(fila[v], fila[ref], nombres) for g, nombres in gr.items() if nombres}
            e["por_tarea"] = {c: boot_dif({c: fila[v][c]}, {c: fila[ref][c]}, [c]) for c in tareas}
            d["frente_a_" + ref] = e
        informe["diferencias"][v] = d
    informe["llm_directo"] = {c: {"exactitud": round(float(t.llm_acierta.mean()), 4), "validas": int(t.llm_ok.sum()), "casos": t.n} for c, t in tareas.items()}
    return informe


def veredicto(informe, retraso=RETRASO, semillas=None):
    """Aplica CRIT_R6 con las exactitudes medias entre semillas (estimaciones puntuales, como los criterios de R1)."""
    v, V = CRIT_R6, informe["variantes"]
    nuevas = [c for c in CRIT_R6["nuevas"] if c in V[v["variante"]]["por_tarea"]]
    if "nuevas" not in V[v["variante"]]["agrupada"]:
        return {"criterio_sha256": CRIT_R6_SHA, "estado": "NO EVALUABLE", "motivo": "ninguna tarea nueva en la batería"}
    a4 = V[v["variante"]]["agrupada"]["nuevas"]["exactitud"][0]
    a1 = V[v["ref_nuevas"]]["agrupada"]["nuevas"]["exactitud"][0]
    dif = round(a4 - a1, 4)
    perd = {}
    for c in V[v["variante"]]["por_tarea"]:
        if c in ("lexboe", "fakenews-es"):
            continue
        perd[c] = round(V[v["variante"]]["por_tarea"][c]["exactitud"][0] - V[v["ref_originales"]]["por_tarea"][c]["exactitud"][0], 4)
    c_sube = dif >= v["sube_min"] - 1e-9
    c_perdida = all(p >= -v["perdida_max_original"] - 1e-9 for p in perd.values())
    out = {"criterio_sha256": CRIT_R6_SHA, "nuevas_presentes": nuevas, "nuevas_variante4": a4, "nuevas_oro_estatico": a1,
           "dif_nuevas": dif, "dif_originales_vs_oro_llm_estatico": perd, "c_sube": c_sube, "c_perdida": c_perdida,
           "parametros_coinciden": retraso == v["retraso"], "refutado": dif <= v["refutado_si_dif_le"]}
    out["estado"] = "REFUTADO" if out["refutado"] else ("PASA" if c_sube and c_perdida else "NO PASA")
    return out


def cmd_medir(a):
    z = np.load(a.vec_fondo)
    varios = lambda r: [x for f in _lista(r) for x in leer(f)]  # noqa: E731  (fondo + fondo2, en el orden de fondo.npz)
    tareas = cargar_tareas(_lista(a.bateria), a.vec, _lista(a.l3), varios(a.fondo), varios(a.l3_fondo), z)
    out = {"criterio": CRIT_R6, "criterio_sha256": CRIT_R6_SHA, "parametros": {"alfa": ALFA, "s_oro": S_ORO, "retraso": a.retraso, "semillas": a.semillas,
           "vecinos": VECINOS, "temp": TEMP},
           "tareas": {c: {"flujo": t.n, "test": t.n_test, "oro": t.G, "fondo_llm": t.F, "llm_validas_en_flujo": int(t.llm_ok.sum())} for c, t in tareas.items()}}
    out.update(medir(tareas, a.semillas, a.retraso))
    out["veredicto"] = veredicto(out, a.retraso, a.semillas)
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["veredicto"], ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("medir")
    m.add_argument("--bateria", required=True, help="directorio o lista separada por comas")
    m.add_argument("--vec", required=True)
    m.add_argument("--l3", required=True, help="fichero o lista separada por comas con las etiquetas del LLM del test")
    m.add_argument("--fondo", required=True)
    m.add_argument("--l3-fondo", required=True)
    m.add_argument("--vec-fondo", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--semillas", type=int, default=10)
    m.add_argument("--retraso", type=int, default=RETRASO)
    sp.add_parser("criterio")
    a = ap.parse_args(argv)
    if a.cmd == "criterio":
        print(CRIT_R6_SHA, json.dumps(CRIT_R6, sort_keys=True))
        return
    cmd_medir(a)


if __name__ == "__main__":
    sys.exit(main())
