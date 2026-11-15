"""Paso 2: techo del LLM (Qwen3-4B-Instruct-2507) en la batería no vista y el LLM como etiquetador de un fondo sin etiqueta.

    python -m model.alt.llm_bateria muestra --bateria DIR --salida DIR [--por-tarea 250] [--q4 hatecheck-es=150,...]
    python -m model.alt.llm_bateria fondo --hub DIR_HF_HUB --bateria DIR --salida DIR [--por-tarea 2000] [--bajar]
    python -m model.alt.llm_bateria medir --muestra DIR --bf16 l3_bateria.jsonl [--q4 l3_q4.jsonl] --bateria DIR --pred DIR --vec DIR --salida informe.json
    python -m model.alt.llm_bateria fondo-resumen --fondo DIR --l3 l3_fondo.jsonl --salida resumen.json

`muestra` escribe `filas.jsonl` (formato de `llm_jevbench.py`: id, tipo, pregunta, opciones, descripciones, texto; más conjunto y
etiqueta, que el cliente no lee): una muestra estratificada por etiqueta de `--por-tarea` filas de test por tarea (hatecheck-es
completa) y `filas-q4.jsonl`, una submuestra ANIDADA (mismas filas) para comparar Q4_K_M con bf16 por pares.
`fondo` escribe `fondo.jsonl` (textos sin etiqueta, mismo formato, intercalados por tarea para que un corte deje todas las tareas
con parte) y `fondo-fuente.jsonl` (id → etiqueta de la fuente, que NO va a la instancia: solo sirve para medir cuánto acierta el
etiquetador). Un texto del fondo nunca está en el test de ninguna tarea de la batería (texto normalizado); en hatecheck-es además
se descarta toda plantilla que tenga un caso en el test (los casos de una plantilla son casi el mismo texto).
`medir` (numpy; necesita los vectores y predicciones del paso 1) da la exactitud del LLM por tarea y la agrupada ponderada por el
tamaño de cada tarea en la batería completa (comparable con 0,610 de la fusión), las diferencias pareadas con kNN, L2, el cruzado
y la fusión en las mismas filas, el AUROC del escalado (validez y probabilidad máxima) y Q4 frente a bf16, con los criterios.
Las etiquetas son de las fuentes; nada de Jev. Las salidas del LLM (Apache-2.0) pueden servir de etiquetas para el paso 3.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

CAMPOS = ("id", "conjunto", "tarea", "tipo", "pregunta", "opciones", "descripciones", "texto")
Q4_DEF = {"hatecheck-es": 150, "inyecciones": 60, "jailbreak": 60, "massive-es": 40, "massive-esc": 60, "tuits-es": 60}
CRIT5 = {"pasa_agrupada_min": 0.62, "pasa_hatecheck_min": 0.75, "refutado_agrupada_bajo": 0.58}   # paso 5 (R2), congelado
CRIT5L = {"metodo": "logprobs: etiqueta de un carácter, max_tokens=1, top_logprobs=20, relleno por prompt_logprobs (max_extra 4)",
          "pasa_agrupada_min": 0.62, "pasa_hatecheck_min": 0.75, "refutado_agrupada_bajo": 0.58}   # R5, congelado (sha en el LEEME)
CRIT = {"agrupada_min": 0.65, "hatecheck_sobre_knn": 0.05, "q4_perdida_max": 0.02, "refutado_bajo": 0.610}


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def escribir(p, filas):
    with open(p, "w", encoding="utf-8") as fh:
        for r in filas:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def estratificada(filas, n, sal):
    """n filas con la proporción de etiquetas de `filas` (resto mayor), orden y elección fijados por `sal`."""
    if n >= len(filas):
        return list(filas)
    rng = random.Random(sal)
    por = {}
    for r in filas:
        por.setdefault(r["etiqueta"], []).append(r)
    cuota = {e: n * len(v) / len(filas) for e, v in por.items()}
    base = {e: int(c) for e, c in cuota.items()}
    for e in sorted(cuota, key=lambda e: (-(cuota[e] - base[e]), e))[:n - sum(base.values())]:
        base[e] += 1
    out = []
    for e in sorted(por):
        v = list(por[e])
        rng.shuffle(v)
        out += v[:base[e]]
    rng.shuffle(out)
    return out


def a_fila(r, conjunto, etiqueta=True):
    f = {k: r[k] for k in CAMPOS if k in r}
    f["conjunto"] = conjunto
    if etiqueta:
        f["etiqueta"] = r["etiqueta"]
    return f


def intercalar(filas):
    por = {}
    for f in filas:
        por.setdefault(f["conjunto"], []).append(f)
    out = []
    for i in range(max((len(v) for v in por.values()), default=0)):
        out += [por[c][i] for c in sorted(por) if i < len(por[c])]
    return out


# ---------------------------------------------------------------- muestra
def cmd_muestra(a):
    from model.alt.p0_medir import cargar_bateria
    q4 = dict(Q4_DEF)
    for par in (a.q4 or "").split(","):
        if par:
            k, v = par.split("=")
            q4[k] = int(v)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    todas, sub, meta = [], [], {}
    for c, (_, ev) in sorted(cargar_bateria(a.bateria).items()):
        n = len(ev) if c in a.completa.split(",") else a.por_tarea
        m = estratificada(ev, n, "llm-bateria-%s-%d" % (c, a.semilla))
        s = estratificada(m, q4.get(c, 0), "llm-bateria-q4-%s-%d" % (c, a.semilla))
        todas += [a_fila(r, c) for r in m]
        sub += [a_fila(r, c) for r in s]
        meta[c] = {"n_bateria": len(ev), "n_muestra": len(m), "n_q4": len(s), "opciones": len(ev[0]["opciones"])}
    escribir(sal / "filas.jsonl", todas)
    # Q4: la submuestra anidada primero y luego el resto de la muestra, intercalados por tarea (un corte por tiempo deja todas con parte)
    sid = {f["id"] for f in sub}
    escribir(sal / "filas-q4.jsonl", intercalar(sub) + intercalar([f for f in todas if f["id"] not in sid]))
    meta["_total"] = {"muestra": len(todas), "q4_anidada": len(sub), "q4_fichero": len(todas), "semilla": a.semilla}
    (sal / "muestra.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


# ---------------------------------------------------------------- fondo sin etiqueta
def bajar(hub):
    from huggingface_hub import snapshot_download
    man = json.loads((Path(__file__).parent / "resultados-20261113-p0-p2" / "bateria-manifiesto.json").read_text(encoding="utf-8"))
    pat = {"Paul/hatecheck-spanish": ["test.csv"], "mteb/amazon_massive_intent": ["*/es.json.gz"],
           "mteb/amazon_massive_scenario": ["*/es.json.gz"], "mteb/tweet_sentiment_multilingual": ["*/spanish.jsonl.gz"],
           "jackhhao/jailbreak-classification": ["default/*.csv", "*.csv"], "deepset/prompt-injections": ["*.parquet", "data/*", "*.csv"]}
    for c, m in man.items():
        if not isinstance(m, dict) or "repo" not in m:
            continue
        snapshot_download(m["repo"], repo_type="dataset", revision=m["rev"], cache_dir=hub, allow_patterns=pat[m["repo"]] + ["README.md"])


def cmd_fondo(a):
    from model.alt.bateria_no_vista import convertir_todo
    from model.alt.p0_medir import cargar_bateria
    if a.bajar:
        bajar(a.hub)
    bat = cargar_bateria(a.bateria)
    test_txt = {norm(r["texto"]) for _, ev in bat.values() for r in ev}
    plant_test = {r.get("plantilla") for r in bat["hatecheck-es"][1]} - {None}
    por, meta = {}, {}
    for c, (filas, _) in sorted(convertir_todo(a.hub).items()):
        if c not in bat:
            continue
        cand = [r for r in filas if r["split"] in ("train", "dev") or c == "hatecheck-es"]
        n0 = len(cand)
        cand = [r for r in cand if norm(r["texto"]) not in test_txt]
        n1 = len(cand)
        if c == "hatecheck-es":
            cand = [r for r in cand if r.get("plantilla") not in plant_test]
        vistos, uni = set(), []
        for r in cand:
            if norm(r["texto"]) not in vistos:
                vistos.add(norm(r["texto"]))
                uni.append(r)
        rng = random.Random("llm-fondo-%s" % c)
        rng.shuffle(uni)
        por[c] = uni[:a.por_tarea]
        meta[c] = {"candidatas": n0, "fuera_por_texto_de_test": n0 - n1, "fuera_por_plantilla_de_test": n1 - len(cand),
                   "unicas": len(uni), "en_fondo": len(por[c])}
    fondo, fuente = [], []
    for i in range(max(len(v) for v in por.values())):   # intercalado: un corte deja todas las tareas con parte
        for c in sorted(por):
            if i < len(por[c]):
                r = por[c][i]
                f = a_fila(r, c, etiqueta=False)
                f["id"] = "fondo:" + r["id"]
                fondo.append(f)
                fuente.append({"id": f["id"], "conjunto": c, "etiqueta": r["etiqueta"]})
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    escribir(sal / "fondo.jsonl", fondo)
    escribir(sal / "fondo-fuente.jsonl", fuente)
    meta["_total"] = len(fondo)
    (sal / "fondo.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


def cmd_fondo2(a):
    """R4: fondo ampliado. Textos sin usar de los mismos train/dev (massive-es y massive-esc, las únicas tareas del fondo que no se
    agotaron) más train/dev de LexBOE, con las mismas exclusiones que `fondo` (ningún texto de test de la batería, de las tareas nuevas
    ni de su soporte de oro, ni ya usado en el fondo de esa tarea). Orden: bloque 1 = LexBOE y massive-esc intercalados; bloque 2 =
    massive-es y el resto de massive-esc intercalados (massive-es es lo más caro: 60 opciones). Un corte por tiempo deja el bloque 1."""
    from model.alt.bateria_no_vista import convertir_todo
    from model.alt.p0_medir import cargar_bateria
    from model.alt.tareas_nuevas import lexboe
    bat = cargar_bateria(a.bateria)
    nuevas = cargar_bateria(a.nuevas)
    prohibido = {norm(r["texto"]) for d in (bat, nuevas) for sop, ev in d.values() for r in sop + ev}
    previo = {}
    for r in leer(Path(a.fondo_previo) / "fondo.jsonl"):
        previo.setdefault(r["conjunto"], set()).add(norm(r["texto"]))
    def elegir(filas, c, n):
        vistos, uni = set(previo.get(c, ())), []
        for r in filas:
            t = norm(r["texto"])
            if r["split"] in ("train", "dev") and t not in prohibido and t not in vistos:
                vistos.add(t)
                uni.append(r)
        rng = random.Random("llm-fondo2-%s" % c)
        rng.shuffle(uni)
        return uni[:n], len(uni)
    todo = convertir_todo(a.hub)
    por, meta = {}, {}
    for c, n in (("massive-esc", a.massive_esc), ("massive-es", a.massive_es)):
        por[c], meta[c] = elegir(todo[c][0], c, n)
    lx, mlx = lexboe(a.hub_nuevas)
    por["lexboe"], meta["lexboe"] = elegir(lx, "lexboe", a.lexboe)
    meta = {c: {"disponibles": meta[c], "en_fondo2": len(por[c])} for c in por}
    def filas(c, rs):
        out = []
        for r in rs:
            f = a_fila(r, c, etiqueta=False)
            f["id"] = "fondo2:" + r["id"]
            out.append((f, {"id": f["id"], "conjunto": c, "etiqueta": r["etiqueta"]}))
        return out
    mitad = len(por["massive-esc"]) // 2
    b1 = intercalar([f for f, _ in filas("lexboe", por["lexboe"]) + filas("massive-esc", por["massive-esc"][:mitad])])
    b2 = intercalar([f for f, _ in filas("massive-es", por["massive-es"]) + filas("massive-esc", por["massive-esc"][mitad:])])
    fuente = [g for c in por for _, g in filas(c, por[c])]
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    escribir(sal / "fondo2.jsonl", b1 + b2)
    escribir(sal / "fondo2-fuente.jsonl", fuente)
    meta["_total"] = len(b1) + len(b2)
    meta["_bloque1"] = len(b1)
    (sal / "fondo2.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


def cmd_nuevas(a):
    """R4 (a): el test de las tareas nuevas en formato de cliente (con conjunto y etiqueta, que el cliente no lee)."""
    from model.alt.p0_medir import cargar_bateria
    filas = [a_fila(r, c) for c, (_, ev) in sorted(cargar_bateria(a.nuevas).items()) for r in ev]
    escribir(Path(a.salida), intercalar(filas))
    print(json.dumps({c: sum(1 for f in filas if f["conjunto"] == c) for c in sorted({f["conjunto"] for f in filas})}))


# ---------------------------------------------------------------- medición
def llm_pred(ruta, filas, permisivo=False):
    """{id: (top1 o None, pmax, valida)}; sin respuesta = inválida."""
    from model.alt.llm_techo import permisivo as perm
    from model.alt.llm_techo import top1
    res = {r["id"]: r for r in leer(ruta)}
    out = {}
    for f in filas:
        r = res.get(f["id"])
        if r is None:
            out[f["id"]] = (None, 0.0, False)
            continue
        p = r["probs"] if r["valida"] else (perm(r.get("raw", ""), f["opciones"]) if permisivo else None)
        t = top1(p, f["opciones"], f["tipo"]) if p else None
        out[f["id"]] = (t, max(p.values()) if p else 0.0, bool(r["valida"]))
    return out


def ponderada(ok_por, pesos):
    return sum(pesos[c] * sum(v) / len(v) for c, v in ok_por.items()) / sum(pesos[c] for c in ok_por)


def boot(ok_por, pesos, remuestreos=2000, semilla=0, ref=None):
    """IC95 de la exactitud ponderada (o de la diferencia con `ref`, por pares), remuestreando dentro de cada tarea."""
    rng = random.Random(semilla)
    dif = {c: [x - (ref[c][i] if ref else 0.0) for i, x in enumerate(v)] for c, v in ok_por.items()}
    vals = []
    for _ in range(remuestreos):
        vals.append(ponderada({c: [v[rng.randrange(len(v))] for _ in v] for c, v in dif.items()}, pesos))
    vals.sort()
    return [round(ponderada(dif, pesos), 4), round(vals[int(0.025 * remuestreos)], 4), round(vals[int(0.975 * remuestreos) - 1], 4)]


def auroc(conf, ok):
    from model.alt.p0_medir import auroc as a
    r = a(conf, ok)
    return None if r is None else round(r, 4)


def referencias(a, ids_por):
    """{modelo: {conjunto: {id: acierto}}} para kNN (guardado y recalculado), L2, cruzado y fusión (todo el soporte, w fuera de tarea)."""
    import numpy as np
    from model.alt import fusion_knn_l2 as fk
    datos = fk.cargar(a.bateria, a.pred, a.vec)
    r = fk.medir(datos, semillas=1, modo="loglineal", ks=(), verbose=False)["_por_k"]["todo"]
    out = {m: {} for m in ("knn", "knn_recalculado", "l2", "ce", "fusion")}
    for c, d in datos.items():
        idx = {i: j for j, i in enumerate(d["ids"])}
        cols = {"knn": fk.acierto(d["Pk_guardado"], d), "knn_recalculado": r[c]["ok_knn"][0], "l2": fk.acierto(d["Pl"], d),
                "ce": fk.acierto(d["Pc"], d), "fusion": r[c]["ok_fus"][0]}
        for m, v in cols.items():
            out[m][c] = {i: float(np.asarray(v)[idx[i]]) for i in ids_por.get(c, [])}
    return out


def cmd_medir(a):
    from model.alt.p0_medir import mcnemar
    mu = Path(a.muestra)
    filas = leer(mu / "filas.jsonl")
    meta = json.loads((mu / "muestra.json").read_text(encoding="utf-8"))
    pesos = {c: v["n_bateria"] for c, v in meta.items() if not c.startswith("_")}
    tareas = sorted(pesos)
    por = {c: [f for f in filas if f["conjunto"] == c] for c in tareas}
    ids_por = {c: [f["id"] for f in v] for c, v in por.items()}
    out = {"criterios": CRIT, "muestra": meta, "pesos": pesos}

    def oks(pred, sub=None):
        return {c: [1.0 if pred[f["id"]][0] == f["etiqueta"] else 0.0 for f in por[c] if sub is None or f["id"] in sub] for c in tareas}

    bf = llm_pred(a.bf16, filas)
    bfp = llm_pred(a.bf16, filas, permisivo=True)
    ok_bf, ok_bfp = oks(bf), oks(bfp)
    refs = referencias(a, ids_por) if a.vec else {}
    tab = {}
    for c in tareas:
        n = len(por[c])
        t = {"n": n, "estricta": boot({c: ok_bf[c]}, {c: 1}), "permisiva": boot({c: ok_bfp[c]}, {c: 1}),
             "invalidas": sum(1 for f in por[c] if not bf[f["id"]][2])}
        conf = [bf[f["id"]][1] if bf[f["id"]][2] else 0.0 for f in por[c]]
        t["auroc_pmax"] = auroc(conf, ok_bf[c])
        t["auroc_validez"] = auroc([1.0 if bf[f["id"]][2] else 0.0 for f in por[c]], ok_bf[c])
        for m, v in refs.items():
            ref = [v[c][f["id"]] for f in por[c]]
            t[m] = round(sum(ref) / n, 4)
            t["llm_menos_" + m] = boot({c: ok_bf[c]}, {c: 1}, ref={c: ref})
            t["mcnemar_" + m] = mcnemar(ok_bf[c], ref)
        tab[c] = t
    g = {"estricta": boot(ok_bf, pesos), "permisiva": boot(ok_bfp, pesos),
         "sin_ponderar_estricta": round(sum(map(sum, ok_bf.values())) / sum(map(len, ok_bf.values())), 4)}
    allok = [x for c in tareas for x in ok_bf[c]]
    g["auroc_pmax_mezcla"] = auroc([bf[f["id"]][1] if bf[f["id"]][2] else 0.0 for c in tareas for f in por[c]], allok)
    g["auroc_validez_mezcla"] = auroc([1.0 if bf[f["id"]][2] else 0.0 for c in tareas for f in por[c]], allok)
    vals = [tab[c]["auroc_pmax"] for c in tareas if tab[c]["auroc_pmax"] is not None]
    g["auroc_pmax_media_tareas"] = round(sum(vals) / len(vals), 4) if vals else None
    for m, v in refs.items():
        ref = {c: [v[c][f["id"]] for f in por[c]] for c in tareas}
        g[m] = round(ponderada(ref, pesos), 4)
        g["llm_menos_" + m] = boot(ok_bf, pesos, ref=ref)
    out["por_tarea"], out["agrupada_ponderada"] = tab, g
    if a.q4:
        hechas = {r["id"] for r in leer(a.q4)}   # el fichero Q4 puede venir cortado por tiempo: solo las filas con respuesta
        sub = [f for f in leer(mu / "filas-q4.jsonl") if f["id"] in hechas]
        sid = {f["id"] for f in sub}
        q4 = llm_pred(a.q4, sub)
        oq = {c: v for c, v in oks(q4, sid).items() if v}
        ob = {c: v for c, v in oks(bf, sid).items() if v}
        pq = {c: pesos[c] for c in oq}
        out["q4"] = {"n": len(sub), "n_fichero": len(leer(mu / "filas-q4.jsonl")), "invalidas": sum(1 for f in sub if not q4[f["id"]][2]),
                     "q4_ponderada": boot(oq, pq), "bf16_mismas_filas": boot(ob, pq), "q4_menos_bf16": boot(oq, pq, ref=ob),
                     "por_tarea": {c: {"n": len(oq[c]), "q4": round(sum(oq[c]) / len(oq[c]), 4), "bf16": round(sum(ob[c]) / len(ob[c]), 4),
                                       "mcnemar": mcnemar(oq[c], ob[c]), "q4_menos_bf16": boot({c: oq[c]}, {c: 1}, ref={c: ob[c]}),
                                       "invalidas_q4": sum(1 for f in sub if f["conjunto"] == c and not q4[f["id"]][2]),
                                       "invalidas_bf16": sum(1 for f in sub if f["conjunto"] == c and not bf[f["id"]][2])} for c in oq},
                     "mcnemar_agrupado": mcnemar([x for c in sorted(oq) for x in oq[c]], [x for c in sorted(ob) for x in ob[c]]),
                     "acuerdo_top1": round(sum(1 for f in sub if q4[f["id"]][0] == bf[f["id"]][0]) / len(sub), 4)}
    out["veredicto"] = veredicto(out)
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["veredicto"], ensure_ascii=False, indent=1))


def veredicto(out):
    g, h = out["agrupada_ponderada"], out["por_tarea"].get("hatecheck-es", {})
    v = {"agrupada": g["estricta"][0], "c_agrupada": g["estricta"][0] >= CRIT["agrupada_min"]}
    if "knn" in h:
        v["hatecheck_llm_menos_knn"] = round(h["estricta"][0] - max(h["knn"], h["knn_recalculado"]), 4)
        v["c_hatecheck"] = v["hatecheck_llm_menos_knn"] >= CRIT["hatecheck_sobre_knn"]
    if "q4" in out:
        v["q4_menos_bf16"] = out["q4"]["q4_menos_bf16"]
        v["c_q4"] = -out["q4"]["q4_menos_bf16"][0] <= CRIT["q4_perdida_max"]
    if "fusion" in g:
        v["fusion_mismas_filas"] = g["fusion"]
        v["refutado"] = g["estricta"][0] < g["fusion"]
    pasa = all(v.get(k) for k in ("c_agrupada", "c_hatecheck", "c_q4"))
    completo = all(k in v for k in ("c_hatecheck", "c_q4", "refutado"))
    v["estado"] = "REFUTADO" if v.get("refutado") else ("PASA" if pasa else ("NO PASA" if completo else "INCOMPLETO"))
    return v


def percentiles(ms):
    v = sorted(ms)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]  # noqa: E731
    return {"n": len(v), "p50_ms": q(0.5), "p95_ms": q(0.95), "media_ms": round(sum(v) / len(v))}


def cmd_pequeno(a):
    """Paso 5: LLM más pequeño en la misma muestra que el paso 2, frente a Qwen3-4B bf16 (por pares) y con CRIT5."""
    from model.alt.p0_medir import mcnemar
    mu = Path(a.muestra)
    filas = leer(mu / "filas.jsonl")
    meta = json.loads((mu / "muestra.json").read_text(encoding="utf-8"))
    pesos = {c: v["n_bateria"] for c, v in meta.items() if not c.startswith("_")}
    tareas = sorted(pesos)
    por = {c: [f for f in filas if f["conjunto"] == c] for c in tareas}
    tarea_de = {f["id"]: f["conjunto"] for f in filas}

    def oks(pred):
        return {c: [1.0 if pred[f["id"]][0] == f["etiqueta"] else 0.0 for f in por[c]] for c in tareas}

    ref = llm_pred(a.ref, filas)
    ok_ref = oks(ref)
    lat = dict(x.split("=", 1) for x in (a.lat or []))
    cr = CRIT5L if a.criterio == "r5" else CRIT5
    out = {"criterios": cr, "criterio_sha256": hashlib.sha256(json.dumps(cr, sort_keys=True).encode()).hexdigest(), "referencia_4b": {"agrupada": boot(ok_ref, pesos)}, "modelos": {}}
    for nom, ruta in (x.split("=", 1) for x in a.modelo):
        pr, prp = llm_pred(ruta, filas), llm_pred(ruta, filas, permisivo=True)
        ok, okp = oks(pr), oks(prp)
        res = leer(ruta)
        m = {"respondidas": len({r["id"] for r in res} & set(tarea_de)), "n": len(filas),
             "agrupada": boot(ok, pesos), "permisiva": boot(okp, pesos), "menos_4b": boot(ok, pesos, ref=ok_ref),
             "mcnemar_4b": mcnemar([x for c in tareas for x in ok[c]], [x for c in tareas for x in ok_ref[c]]),
             "acuerdo_top1_4b": round(sum(1 for f in filas if pr[f["id"]][0] == ref[f["id"]][0]) / len(filas), 4),
             "por_tarea": {c: {"n": len(por[c]), "estricta": boot({c: ok[c]}, {c: 1}), "cuatro_b": round(sum(ok_ref[c]) / len(ok_ref[c]), 4),
                               "menos_4b": boot({c: ok[c]}, {c: 1}, ref={c: ok_ref[c]}), "mcnemar_4b": mcnemar(ok[c], ok_ref[c]),
                               "invalidas": sum(1 for f in por[c] if not pr[f["id"]][2])} for c in tareas},
             "latencia_concurrente": percentiles([r["ms"] for r in res if r["id"] in tarea_de])}
        if nom in lat:
            lr = [r for r in leer(lat[nom]) if r["id"] in tarea_de]
            m["latencia_un_hilo"] = percentiles([r["ms"] for r in lr])
            m["latencia_un_hilo_por_tarea"] = {c: percentiles([r["ms"] for r in lr if tarea_de[r["id"]] == c]) for c in tareas}
        g, h = m["agrupada"][0], m["por_tarea"].get("hatecheck-es", {}).get("estricta", [None])[0]
        v = {"agrupada": g, "hatecheck": h, "completo": m["respondidas"] == m["n"]}
        v["estado"] = ("REFUTADO" if g < cr["refutado_agrupada_bajo"] else
                       "PASA" if g >= cr["pasa_agrupada_min"] and h is not None and h >= cr["pasa_hatecheck_min"] else "NO PASA")
        if not v["completo"] and v["estado"] != "PASA":
            v["estado"] += " (INCOMPLETO: faltan %d filas, contadas como fallo)" % (m["n"] - m["respondidas"])
        m["veredicto"] = v
        out["modelos"][nom] = m
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["veredicto"] for k, v in out["modelos"].items()}, ensure_ascii=False, indent=1))


def cmd_fondo_resumen(a):
    from model.alt.llm_techo import top1
    fuente = {r["id"]: r for r in leer(Path(a.fondo) / "fondo-fuente.jsonl")}
    filas = {r["id"]: r for r in leer(Path(a.fondo) / "fondo.jsonl")}
    res = {r["id"]: r for r in leer(a.l3)}
    out = {}
    for c in sorted({r["conjunto"] for r in fuente.values()}):
        ids = [i for i, r in fuente.items() if r["conjunto"] == c]
        hechas = [i for i in ids if i in res]
        val = [i for i in hechas if res[i]["valida"]]
        ok = [1.0 if top1(res[i]["probs"], filas[i]["opciones"], filas[i]["tipo"]) == fuente[i]["etiqueta"] else 0.0 for i in val]
        out[c] = {"en_fondo": len(ids), "etiquetadas": len(hechas), "validas": len(val),
                  "acierto_frente_a_la_fuente": round(sum(ok) / len(ok), 4) if ok else None,
                  "con_pmax_ge_0_9": sum(1 for i in val if max(res[i]["probs"].values()) >= 0.9)}
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("muestra")
    m.add_argument("--bateria", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--por-tarea", type=int, default=250)
    m.add_argument("--completa", default="hatecheck-es")
    m.add_argument("--q4", help="tamaños de la submuestra Q4, p. ej. hatecheck-es=150,massive-es=40")
    m.add_argument("--semilla", type=int, default=0)
    f = sp.add_parser("fondo")
    f.add_argument("--hub", required=True)
    f.add_argument("--bateria", required=True)
    f.add_argument("--salida", required=True)
    f.add_argument("--por-tarea", type=int, default=2000)
    f.add_argument("--bajar", action="store_true")
    d = sp.add_parser("medir")
    d.add_argument("--muestra", required=True)
    d.add_argument("--bf16", required=True)
    d.add_argument("--q4")
    d.add_argument("--bateria", required=True)
    d.add_argument("--pred")
    d.add_argument("--vec")
    d.add_argument("--salida", required=True)
    r = sp.add_parser("fondo-resumen")
    r.add_argument("--fondo", required=True)
    r.add_argument("--l3", required=True)
    r.add_argument("--salida", required=True)
    f2 = sp.add_parser("fondo2")
    f2.add_argument("--hub", required=True)
    f2.add_argument("--hub-nuevas", required=True)
    f2.add_argument("--bateria", required=True)
    f2.add_argument("--nuevas", required=True)
    f2.add_argument("--fondo-previo", required=True)
    f2.add_argument("--salida", required=True)
    f2.add_argument("--massive-es", type=int, default=6000)
    f2.add_argument("--massive-esc", type=int, default=8000)
    f2.add_argument("--lexboe", type=int, default=4000)
    nv = sp.add_parser("nuevas")
    nv.add_argument("--nuevas", required=True)
    nv.add_argument("--salida", required=True)
    q = sp.add_parser("pequeno")
    q.add_argument("--muestra", required=True)
    q.add_argument("--ref", required=True, help="l3_bateria.jsonl de Qwen3-4B bf16 (paso 2)")
    q.add_argument("--modelo", action="append", required=True, help="nombre=l3_x.jsonl (la muestra entera, 32 hilos)")
    q.add_argument("--lat", action="append", help="nombre=lat_x.jsonl (subconjunto con un hilo)")
    q.add_argument("--salida", required=True)
    q.add_argument("--criterio", choices=("r2", "r5"), default="r2", help="r5: CRIT5L (método por logprobs, mismos umbrales)")
    a = ap.parse_args(argv)
    {"muestra": cmd_muestra, "fondo": cmd_fondo, "medir": cmd_medir, "fondo-resumen": cmd_fondo_resumen, "pequeno": cmd_pequeno, "fondo2": cmd_fondo2, "nuevas": cmd_nuevas}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
