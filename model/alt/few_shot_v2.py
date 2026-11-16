"""R10-c: ¿mejora el few-shot real el soporte etiquetado por el LLM? Seis tareas de DESARROLLO de la batería v2 (nunca las selladas).

    python -m model.alt.few_shot_v2 preparar --hub HUB --dev DIR_DEV --manifiesto M.json --salida DIR     (i7, sin torch)
    python -m model.alt.few_shot_v2 ordenar --fondo fondo.jsonl --salida reetiqueta.jsonl [--ventana 720]   (orden de la g5 + estimación)
    python -m model.alt.few_shot_v2 armar-dev32 --dev DIR_DEV --oro32 oro32.jsonl --salida DIR              (train = oro32, test = dev)
    python -m model.alt.few_shot_v2 filtrar-fondo --fondo fondo.jsonl --lp lp_reetiqueta.jsonl --salida fondo_lp.jsonl     (solo lo etiquetado)
    python -m model.alt.few_shot_v2 vectores-falsos --bateria DIR --fondo fondo.jsonl --salida DIR_VEC      (SOLO prueba en seco, sin torch)
    python -m model.alt.few_shot_v2 medir --bateria DIR --vec DIR_VEC --fondo F --fondo-fuente FF --lp LP --crit crit-r10-c.json --crit-sha SHA --salida J
    python -m model.alt.few_shot_v2 criterio

Datos (solo medir, nunca entrenar; los textos de train de origen nunca son del test ni del soporte de oro de 8, por texto normalizado):
  oro32.jsonl        hasta 32 por clase, contiene los 8 de oro del .jsonl de dev (mismos ids); el resto sale del train de origen
  fondo.jsonl        textos de train SIN etiqueta (hasta 2 000 por tarea, distintos de oro32), formato de cliente de `llm_bateria.a_fila`
                     (id `fondo-v2:<tarea>:<n>`, conjunto = nombre de la tarea), barajados con semilla e intercalados por tarea
  fondo-fuente.jsonl la etiqueta de la fuente de cada texto del fondo (NO va a la g5): sirve para medir el acierto del etiquetador
Medición (e5 @ 614241f6, transformers 5.22.2): kNN (32 vecinos, T = 0,05, `fusion_knn_l2.knn`) y cabeza de prototipos (centroide normalizado por
clase, coseno, softmax T = 0,05), con soporte oro8, oro32, oro8+LLM, oro32+LLM y LLM solo. Criterio congelado en `crit-r10-c.json`.
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOPE_CAR = 3000
SINTETICAS = ("finanzas-documentos", "horeca-resenas")        # generadas por LLM (marcadas en los recuentos)
CRIT_RUTA = Path(__file__).with_name("crit-r10-c.json")
CRIT = {
    "nombre": "CRIT_R10_C",
    "sistema_principal": "kNN (32 vecinos, T = 0,05, fusion_knn_l2.knn) sobre e5-small @ 614241f6, transformers 5.22.2",
    "informado_al_lado": "cabeza de prototipos (centroide normalizado por clase, coseno, softmax T = 0,05)",
    "delta": "macro(oro32 + LLM) - macro(oro32), macro = media de la exactitud de las 6 tareas dev, en puntos porcentuales",
    "pasa": {"delta_min": 4.0, "caida_max_por_tarea": 2.0},
    "refutado": {"delta_menor_que": 1.0},
    "no_pasa": "todo lo demas",
    "se_informa_tambien": ["k = 8 (oro8 + LLM frente a oro8)", "macro sin sinteticas (%s)" % ", ".join(SINTETICAS), "prototipos", "IC95 por bootstrap sobre tareas",
                           "+-1 y MAE en las tareas score", "acierto del etiquetador contra la etiqueta de la fuente"],
    "etiqueta_llm": "top-1 de las probabilidades por logprobs de Qwen3-4B-Instruct-2507 (mediana CORAL en score); solo respuestas validas",
    "soportes": ["oro8", "oro32", "oro8+llm", "oro32+llm", "llm"],
    "tareas_sin_fondo": "una tarea sin textos de fondo (poco train) cuenta con delta 0 en el macro",
}
CRIT_SHA = hashlib.sha256(json.dumps(CRIT, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def escribir(p, filas):
    with open(p, "w", encoding="utf-8") as fh:
        for r in filas:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def sha256_fichero(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------ preparar
def candidatos(items, filas_dev, clases, semilla):
    """Textos de train de origen utilizables: train oficial (o todo el pool si no hay split), cortados como en dev, sin los del test ni los
    del soporte de oro de dev (texto normalizado), sin duplicados ni textos con dos etiquetas, solo clases de la tarea. Devuelve {clase: [item]}."""
    tiene = any(i.get("split") for i in items)
    pool = [i for i in items if i.get("split") == "train"] if tiene else list(items)
    fuera = {norm(r["texto"]) for r in filas_dev}
    et = defaultdict(set)
    for i in pool:
        et[norm(i["texto"][:TOPE_CAR])].add(i["etiqueta"])
    rng = random.Random("cand-%s" % semilla)
    rng.shuffle(pool)
    vistos, por = set(), defaultdict(list)
    for i in pool:
        t = (i["texto"] or "").strip()[:TOPE_CAR]
        n = norm(t)
        if not t or n in fuera or n in vistos or len(et[n]) > 1 or i["etiqueta"] not in clases:
            continue
        vistos.add(n)
        por[i["etiqueta"]].append(dict(i, texto=t))
    return por, len(pool), tiene


def preparar(hub, dev, salida, manifiesto=None, semilla=0, k_oro=32, max_fondo=2000):
    from model.alt import bateria_v2 as bv
    from model.alt.llm_bateria import a_fila, intercalar
    tareas = {t["nombre"]: t for t in bv.TAREAS}
    rev = {t["nombre"]: t["rev"] for t in json.loads(Path(manifiesto).read_text(encoding="utf-8"))["tareas"]} if manifiesto else {}
    oro, fondo, fuente, cuenta = [], [], [], {}
    por_tarea = {}
    for p in sorted(Path(dev).glob("*.jsonl")):
        corto = p.stem
        filas = leer(p)
        t = tareas[corto]
        raiz = bv.snapshot(hub, t["repo"])
        if rev and rev.get(corto) != raiz.name:
            raise SystemExit("%s: la instantánea %s no es la del manifiesto %s" % (corto, raiz.name, rev.get(corto)))
        items, _ = t["loader"](raiz)
        plant = next(r for r in filas if r["split"] == "train")
        clases = plant["opciones"]
        oro8 = [r for r in filas if r["split"] == "train"]
        por, n_pool, tiene = candidatos(items, filas, set(clases), semilla)
        # oro32: los 8 de dev + hasta 24 más del train por clase
        extra = []
        usados = defaultdict(int)
        for r in oro8:
            usados[r["etiqueta"]] += 1
        resto = {}
        for c in clases:
            v = por.get(c, [])
            n = max(0, k_oro - usados[c])
            extra += list(v[:n])
            resto[c] = v[n:]
        base = {k: plant[k] for k in ("tarea", "tipo", "pregunta", "opciones", "descripciones", "idioma", "licencia", "uso_comercial")}
        for r in oro8:
            oro.append(dict(r, oro8=True))
        for j, i in enumerate(extra):
            oro.append(dict(base, id="%s:oro32:%d" % (corto, j), texto=i["texto"], etiqueta=i["etiqueta"], split="train", fuente=t["repo"] + " oro32", oro8=False))
        rng = random.Random("fondo-%s-%d" % (corto, semilla))
        libres = [i for c in clases for i in resto[c]]
        rng.shuffle(libres)
        libres = libres[:max_fondo]
        for j, i in enumerate(libres):
            r = dict(base, id="fondo-v2:%s:%d" % (corto, j), texto=i["texto"])
            por_tarea.setdefault(corto, []).append(a_fila(r, corto, etiqueta=False))
            fuente.append(dict(id=r["id"], conjunto=corto, etiqueta=i["etiqueta"]))
        n_oro_cl = Counter(r["etiqueta"] for r in oro if r["id"].startswith(corto + ":"))
        cuenta[corto] = dict(sintetico=corto in SINTETICAS, traducida=(corto == "jql-valor-educativo"), clases=len(clases), oro8=len(oro8),
                             oro32=sum(n_oro_cl.values()), oro32_clases_completas=sum(1 for c in clases if n_oro_cl[c] >= k_oro),
                             oro32_min_por_clase=min(n_oro_cl[c] for c in clases), pool_train_origen=n_pool, split_train_oficial=tiene,
                             candidatos_unicos=sum(len(v) for v in por.values()), fondo=len(libres),
                             aviso=("sin train suficiente: oro32 = oro8 y sin fondo" if len(libres) < 50 and sum(len(v) for v in por.values()) < 50 else None))
    for corto in por_tarea:
        random.Random("orden-%d" % semilla).shuffle(por_tarea[corto])
    fondo = intercalar([f for v in por_tarea.values() for f in v])
    sal = Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    escribir(sal / "oro32.jsonl", oro)
    escribir(sal / "fondo.jsonl", fondo)
    escribir(sal / "fondo-fuente.jsonl", fuente)
    (sal / "recuentos.json").write_text(json.dumps(cuenta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return cuenta


# ------------------------------------------------------------------------------------------------------------ orden para la g5
def coste_texto(n_opciones):
    """Segundos por texto en la g5 (cifras del coordinador: ≈ 0,05 con ≤ 10 opciones, ≈ 1,2 con 60); lineal entre medias."""
    if n_opciones <= 10:
        return 0.05
    return min(1.2, 0.05 + 1.15 * (n_opciones - 10) / 50.0)


def ordenar(filas, por_clase=100, tope_tarea=2000):
    """Orden para que lo más útil quepa en la ventana. Fase 1: las tareas baratas (coste < 0,2 s) hasta `por_clase` textos por clase
    (≤ tope_tarea), intercaladas. Fase 2: se alternan, por coste acumulado, las tareas caras (muchas opciones) y el resto de las baratas,
    para que ninguna se quede sin textos mientras la otra gasta toda la ventana."""
    por = defaultdict(list)
    for f in filas:
        por[f["conjunto"]].append(f)
    cost = {c: coste_texto(len(v[0]["opciones"])) for c, v in por.items()}
    baratas = sorted(c for c in por if cost[c] < 0.2)
    caras = sorted(c for c in por if cost[c] >= 0.2)
    f1, f2b = [], []
    for c in baratas:
        n = min(len(por[c]), tope_tarea, por_clase * len(por[c][0]["opciones"]))
        f1.append(por[c][:n])
        f2b.append(por[c][n:])
    inter = lambda ls: [x for t in zip(*[l + [None] * (max(map(len, ls)) - len(l)) for l in ls]) for x in t if x is not None] if ls else []  # noqa: E731
    fase1 = inter(f1)
    extras = inter(f2b)
    cola = inter([por[c] for c in caras]) if caras else []
    out, t_ex, t_ca = list(fase1), 0.0, 0.0
    i = j = 0
    while i < len(extras) or j < len(cola):
        if j >= len(cola) or (i < len(extras) and t_ex <= t_ca):
            out.append(extras[i]); t_ex += cost[extras[i]["conjunto"]]; i += 1
        else:
            out.append(cola[j]); t_ca += cost[cola[j]["conjunto"]]; j += 1
    return out, cost


def estimar(filas, cost, ventana_s):
    """{tarea: textos hechos dentro de `ventana_s`}, segundos totales y segundos hasta cada punto."""
    t, hechos, total = 0.0, Counter(), Counter(f["conjunto"] for f in filas)
    for f in filas:
        c = cost[f["conjunto"]]
        if t + c > ventana_s:
            break
        t += c
        hechos[f["conjunto"]] += 1
    return dict(hechos), dict(total), sum(cost[f["conjunto"]] for f in filas), t


def cmd_ordenar(a):
    filas = leer(a.fondo)
    out, cost = ordenar(filas)
    escribir(a.salida, out)
    hechos, total, t_total, t_usado = estimar(out, cost, a.ventana)
    print(json.dumps(dict(textos=len(out), segundos_totales_estimados=round(t_total), ventana_s=a.ventana, usado_s=round(t_usado), coste_por_texto={c: round(v, 3) for c, v in sorted(cost.items())},
                          caben={c: "%d de %d" % (hechos.get(c, 0), n) for c, n in sorted(total.items())}), ensure_ascii=False, indent=1))


# ------------------------------------------------------------------------------------------------------------ dev32 y vectores falsos
def armar_dev32(dev, oro32, salida):
    """Un jsonl por tarea con train = oro32 y test = test de dev (para que `knn_soporte_llm vectores` codifique oro32 y test)."""
    sal = Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    oro = defaultdict(list)
    for r in leer(oro32):
        oro[r["id"].split(":")[0]].append(r)
    n = {}
    for p in sorted(Path(dev).glob("*.jsonl")):
        if p.name.startswith("."):
            continue
        test = [r for r in leer(p) if r["split"] == "test"]
        escribir(sal / p.name, [dict(r, split="train") for r in oro[p.stem]] + test)
        n[p.stem] = (len(oro[p.stem]), len(test))
    return n


def vec_falso(texto, d=384):
    import numpy as np
    v = np.zeros(d)
    for w in re.findall(r"\w+", texto.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % d] += 1.0 if (h >> 20) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def filtrar_fondo(fondo, lp, salida):
    """Solo los textos del fondo con respuesta válida del LLM, en el orden del fondo (para no codificar lo que no se etiquetó)."""
    ok = {r["id"] for r in leer(lp) if r.get("valida") and r.get("probs")}
    filas = [r for r in leer(fondo) if r["id"] in ok]
    escribir(salida, filas)
    return len(filas)


def cmd_vectores_falsos(a):
    import numpy as np
    from model.alt.p0_medir import cargar_bateria
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    for c, (sop, ev) in cargar_bateria(a.bateria).items():
        np.savez(sal / (c + ".npz"), Xtr=np.array([vec_falso(r["texto"]) for r in sop]).reshape(len(sop), 384), Xte=np.array([vec_falso(r["texto"]) for r in ev]).reshape(len(ev), 384),
                 idtr=np.array([r["id"] for r in sop]), idte=np.array([r["id"] for r in ev]), transformers=np.array("falso"))
    fo = leer(a.fondo)[:a.max_fondo]
    np.savez(sal / "fondo.npz", X=np.array([vec_falso(r["texto"]) for r in fo]).reshape(len(fo), 384), ids=np.array([r["id"] for r in fo]), transformers=np.array("falso"))


# ------------------------------------------------------------------------------------------------------------ medición
def top1(probs, ops, tipo):
    from model.alt.llm_techo import top1 as t
    return t(probs, ops, tipo)


def probs_knn(Xte, Xs, ys, opciones):
    from model.alt.fusion_knn_l2 import knn
    P, _, _ = knn(Xte, Xs, ys, opciones)
    return [dict(zip(o, P[i, :len(o)])) for i, o in enumerate(opciones)]


def probs_proto(Xte, Xs, ys, opciones, temp=0.05):
    import numpy as np
    cl = sorted(set(ys))
    cent = np.zeros((len(cl), Xs.shape[1]))
    for j, y in enumerate(cl):
        v = Xs[[i for i, t in enumerate(ys) if t == y]].mean(0)
        cent[j] = v / max(np.linalg.norm(v), 1e-12)
    S = Xte @ cent.T if cl else np.zeros((len(Xte), 0))
    col = {y: j for j, y in enumerate(cl)}
    out = []
    for i, ops in enumerate(opciones):
        tiene = [o for o in ops if o in col]
        if not tiene:
            out.append({o: 1.0 / len(ops) for o in ops})
            continue
        s = np.array([S[i, col[o]] for o in tiene]) / temp
        e = np.exp(s - s.max())
        p = dict.fromkeys(ops, 0.0)
        p.update(dict(zip(tiene, (e / e.sum()).tolist())))
        out.append(p)
    return out


def metricas(preds, golds, tipo):
    n = len(golds)
    m = dict(n=n, acc=sum(p == g for p, g in zip(preds, golds)) / n if n else 0.0)
    if tipo == "score" and n:
        d = [abs(int(p) - int(g)) for p, g in zip(preds, golds)]
        m["mas_menos_1"] = sum(x <= 1 for x in d) / n
        m["mae"] = sum(d) / n
    return m


def boot_tareas(vals, remuestreos=2000, semilla=0):
    """Media de las tareas con IC95 por bootstrap sobre tareas (remuestreo con reposición de las tareas)."""
    rng = random.Random(semilla)
    v = list(vals)
    ms = sorted(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for _ in range(remuestreos))
    return [round(sum(v) / len(v), 3), round(ms[int(0.025 * remuestreos)], 3), round(ms[int(0.975 * remuestreos) - 1], 3)]


def veredicto(delta, caidas, crit):
    """delta en puntos; caidas = {tarea: cambio en puntos (negativo = cae)}."""
    peor = min(caidas.values()) if caidas else 0.0
    if delta >= crit["pasa"]["delta_min"] and peor >= -crit["pasa"]["caida_max_por_tarea"]:
        return "PASA"
    if delta < crit["refutado"]["delta_menor_que"]:
        return "REFUTADO"
    return "NO PASA"


def cargar_dev32(bateria, vec, permitir_falso):
    import numpy as np
    from model.alt.p0_medir import cargar_bateria
    datos = {}
    for c, (sop, ev) in cargar_bateria(bateria).items():
        z = np.load(Path(vec) / (c + ".npz"))
        tv = str(z["transformers"])
        if tv != "5.22.2" and not (permitir_falso and tv == "falso"):
            raise SystemExit("%s: vectores con transformers %s, se exigen 5.22.2" % (c, tv))
        assert [str(x) for x in z["idtr"]] == [r["id"] for r in sop] and [str(x) for x in z["idte"]] == [r["id"] for r in ev], c
        datos[c] = dict(sop=sop, ev=ev, Xtr=z["Xtr"], Xte=z["Xte"])
    return datos


def cmd_medir(a):
    import numpy as np
    if sha256_fichero(a.crit) != a.crit_sha:
        raise SystemExit("el criterio %s no es el congelado (sha256 %s)" % (a.crit, sha256_fichero(a.crit)))
    crit = json.loads(Path(a.crit).read_text(encoding="utf-8"))
    if crit != CRIT:
        raise SystemExit("crit-r10-c.json distinto del CRIT del código")
    datos = cargar_dev32(a.bateria, a.vec, a.permitir_falso)
    fo = leer(a.fondo)
    z = np.load(Path(a.vec) / "fondo.npz")
    if [str(x) for x in z["ids"]] != [r["id"] for r in fo][:len(z["ids"])]:
        raise SystemExit("los vectores del fondo no coinciden con fondo.jsonl")
    fo = fo[:len(z["ids"])]
    Xf = z["X"]
    res = {r["id"]: r for r in leer(a.lp)}
    ffu = leer(a.fondo_fuente)
    fu = {r["id"]: r["etiqueta"] for r in ffu}
    en_fondo = Counter(r["conjunto"] for r in ffu)
    llm = defaultdict(list)                      # tarea -> [(indice en fondo, etiqueta del LLM)]
    for i, f in enumerate(fo):
        r = res.get(f["id"])
        if r and r.get("valida") and r.get("probs"):
            t = top1(r["probs"], f["opciones"], f["tipo"])
            if t is not None:
                llm[f["conjunto"]].append((i, t))
    tareas = sorted(datos)
    por = {}
    etiquetador = {}
    for c in tareas:
        d = datos[c]
        oro_ids8 = [i for i, r in enumerate(d["sop"]) if r.get("oro8")]
        if not oro_ids8:
            oro_ids8 = list(range(min(len(d["sop"]), 8)))
        ops = [r["opciones"] for r in d["ev"]]
        gold = [r["etiqueta"] for r in d["ev"]]
        tipo = d["ev"][0]["tipo"]
        yo = [r["etiqueta"] for r in d["sop"]]
        ll = llm.get(c, [])
        XL = Xf[[i for i, _ in ll]] if ll else np.zeros((0, d["Xtr"].shape[1]))
        yl = [t for _, t in ll]
        etiquetador[c] = dict(en_fondo=en_fondo.get(c, 0), recibidos_de_la_g5=sum(1 for r in res.values() if r["id"].startswith("fondo-v2:%s:" % c)), etiquetados=len(ll),
                              acierto_vs_fuente=(round(sum(1 for i, t in ll if fu.get(fo[i]["id"]) == t) / len(ll), 4) if ll else None))
        sup = {"oro8": (d["Xtr"][oro_ids8], [yo[i] for i in oro_ids8]), "oro32": (d["Xtr"], yo),
               "oro8+llm": (np.vstack([d["Xtr"][oro_ids8], XL]), [yo[i] for i in oro_ids8] + yl),
               "oro32+llm": (np.vstack([d["Xtr"], XL]), yo + yl), "llm": (XL, yl)}
        por[c] = {}
        for sis, fun in (("knn", probs_knn), ("proto", probs_proto)):
            for nom, (X, y) in sup.items():
                P = fun(d["Xte"], X, y, ops)
                preds = [top1(p, o, tipo) for p, o in zip(P, ops)]
                por[c][sis + "/" + nom] = metricas(preds, gold, tipo)
    sint = [c for c in tareas if c in SINTETICAS]
    reales = [c for c in tareas if c not in SINTETICAS]
    out = dict(criterio_sha256=a.crit_sha, tareas=tareas, sinteticas=sint, etiquetador=etiquetador, por_tarea=por, macro={}, delta={})
    for sis in ("knn", "proto"):
        for nom in CRIT["soportes"]:
            v = [por[c][sis + "/" + nom]["acc"] for c in tareas]
            out["macro"][sis + "/" + nom] = dict(todas=boot_tareas(v), sin_sinteticas=boot_tareas([por[c][sis + "/" + nom]["acc"] for c in reales]) if reales else None)
        for base, con in (("oro32", "oro32+llm"), ("oro8", "oro8+llm")):
            dd = {c: 100.0 * (por[c]["%s/%s" % (sis, con)]["acc"] - por[c]["%s/%s" % (sis, base)]["acc"]) for c in tareas}
            out["delta"]["%s/%s" % (sis, con)] = dict(
                delta_puntos=boot_tareas([dd[c] for c in tareas]), delta_sin_sinteticas=boot_tareas([dd[c] for c in reales]) if reales else None, por_tarea={c: round(v, 2) for c, v in dd.items()})
    dd = out["delta"]["knn/oro32+llm"]
    out["veredicto"] = veredicto(dd["delta_puntos"][0], dd["por_tarea"], crit)
    out["veredicto_informativo_k8"] = veredicto(out["delta"]["knn/oro8+llm"]["delta_puntos"][0], out["delta"]["knn/oro8+llm"]["por_tarea"], crit)
    out["veredicto_prototipos_oro32"] = veredicto(out["delta"]["proto/oro32+llm"]["delta_puntos"][0], out["delta"]["proto/oro32+llm"]["por_tarea"], crit)
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(veredicto=out["veredicto"], delta_oro32=dd["delta_puntos"], delta_oro8=out["delta"]["knn/oro8+llm"]["delta_puntos"], macro={k: v["todas"][0] for k, v in out["macro"].items()}), ensure_ascii=False))


def escribir_criterio(ruta=CRIT_RUTA):
    Path(ruta).write_text(json.dumps(CRIT, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_fichero(ruta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("preparar")
    p.add_argument("--hub", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--manifiesto")
    p.add_argument("--salida", required=True)
    o = sp.add_parser("ordenar")
    o.add_argument("--fondo", required=True)
    o.add_argument("--salida", required=True)
    o.add_argument("--ventana", type=float, default=720.0, help="segundos de etiquetado útiles en la g5 (35 min − arranque − 12 de cierre)")
    d = sp.add_parser("armar-dev32")
    d.add_argument("--dev", required=True)
    d.add_argument("--oro32", required=True)
    d.add_argument("--salida", required=True)
    ff = sp.add_parser("filtrar-fondo")
    ff.add_argument("--fondo", required=True)
    ff.add_argument("--lp", required=True)
    ff.add_argument("--salida", required=True)
    v = sp.add_parser("vectores-falsos")
    v.add_argument("--bateria", required=True)
    v.add_argument("--fondo", required=True)
    v.add_argument("--salida", required=True)
    v.add_argument("--max-fondo", type=int, default=10 ** 9)
    m = sp.add_parser("medir")
    m.add_argument("--bateria", required=True)
    m.add_argument("--vec", required=True)
    m.add_argument("--fondo", required=True)
    m.add_argument("--fondo-fuente", required=True)
    m.add_argument("--lp", required=True)
    m.add_argument("--crit", required=True)
    m.add_argument("--crit-sha", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--permitir-falso", action="store_true")
    c = sp.add_parser("criterio")
    c.add_argument("--escribir", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "criterio":
        print(escribir_criterio() if a.escribir else sha256_fichero(CRIT_RUTA))
        return 0
    if a.cmd == "preparar":
        print(json.dumps(preparar(a.hub, a.dev, a.salida, a.manifiesto), ensure_ascii=False, indent=1))
    else:
        {"ordenar": cmd_ordenar, "armar-dev32": lambda x: print(armar_dev32(x.dev, x.oro32, x.salida)),
                "filtrar-fondo": lambda x: print(filtrar_fondo(x.fondo, x.lp, x.salida)), "vectores-falsos": cmd_vectores_falsos, "medir": cmd_medir}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
