"""R10 apertura: la apertura única y registrada de las 8 tareas SELLADAS de la batería v2, con el sistema final congelado y sus líneas base.

    python -m model.alt.apertura_v2 sellar    --manifiesto M --selladas DIR --salida DIR --motivo TXT --registro-previo F --registro-nuevo F
    python -m model.alt.apertura_v2 contar    --datos DIR
    python -m model.alt.apertura_v2 verificar --crit crit-r10-apertura.json --crit-sha SHA --datos DIR --modo real|ensayo
    python -m model.alt.apertura_v2 preparar  --datos DIR --salida DIR --crit crit-r10-apertura.json --modo real|ensayo
    python -m model.alt.apertura_v2 medir     --datos DIR --pred DIR [--pred DIR2 ...] --crit crit-r10-apertura.json --salida cifras.json --modo real|ensayo

`sellar` (en el empaquetado, NO en la instancia) es la única vía de lectura de las selladas: pasa por `bateria_v2.abrir_sellada` (que anota la apertura en el
registro ANTES de leer) y se niega si el registro ya trae una apertura de cualquiera de esos ficheros. `preparar` (en la instancia) saca de DIR de datos
(<tarea>.jsonl con split train = soporte de oro y test) las filas por tarea y la rúbrica MECÁNICA de las tareas score, sin que nadie lea las filas:
descripción de cada nivel = la propia de la opción si las filas la traen (`descripciones`), si no «Etiqueta N = nivel N de K.»; ejemplo = la primera fila del
soporte de ese nivel por id (orden natural), recortada a 130 caracteres con «…». `medir` compone el sistema FINAL congelado (choice: fs8-knn; score: rúbrica
+ umbrales isotónicos ajustados en el soporte, excluyendo del ajuste las filas-ejemplo) y mide con las MISMAS funciones de `lineas_base_v2.py`
(exactitud, QWK, MAE, ±1, IC por tareas y por filas), por tarea, macro por tipo (choice / score) y global, diferencias final − cada línea base con IC95,
y la cobertura. El veredicto solo vale si el final y el 4B zero-shot cubren el 100 % de TODAS las tareas selladas; si no, «no medido».
"""
import argparse
import hashlib
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb
from model.alt.auditoria import boot_par
from model.alt.lineas_base_v2b import ids_ejemplos, umbrales
from model.alt.lineas_base_v2d import macro_score

AQUI = Path(__file__).parent
EJ_MAX = 130
SISTEMAS = ["mayoritaria", "knn", "zs", "fs8-azar", "final"]
BASES = ["mayoritaria", "knn", "zs", "fs8-azar"]


def sha256_fichero(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clave_natural(s):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


def cargar_crit(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- empaquetado: la única apertura registrada
def registradas(*registros):
    out = set()
    for reg in registros:
        p = Path(reg)
        if p.exists():
            out |= {json.loads(l)["fichero"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    return out


def cmd_sellar(a):
    from model.alt import bateria_v2 as bv
    man = json.loads(Path(a.manifiesto).read_text(encoding="utf-8"))
    sel = [t for t in man["tareas"] if t["particion"] == "sellada"]
    ficheros = {t["nombre"] + ".jsonl": t for t in sel}
    ya = sorted(set(ficheros) & registradas(a.registro_previo, a.registro_nuevo))
    if ya:
        raise SystemExit("apertura única: el registro ya trae una apertura de %s; no se vuelve a abrir" % ", ".join(ya))
    rutas = {}
    for n, t in ficheros.items():
        r = Path(a.selladas) / n
        if not r.is_file():
            raise SystemExit("falta %s en %s" % (n, a.selladas))
        if bv.sha256_fichero(r) != t["sha256"]:   # solo el hash, antes de abrir: si no coincide con el manifiesto no se abre nada
            raise SystemExit("%s no coincide con el sha256 del manifiesto: no se abre" % n)
        rutas[n] = r
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    nt = ns = 0
    for n in sorted(rutas):   # TODA lectura pasa por abrir_sellada (anota antes de leer)
        filas = bv.abrir_sellada(rutas[n], a.motivo, manifiesto=a.manifiesto, registro=a.registro_nuevo)
        (sal / n).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas), encoding="utf-8")
        if sha256_fichero(sal / n) != ficheros[n]["sha256"]:
            raise SystemExit("%s: reescrito no coincide con el sha256 del manifiesto (apertura ya registrada)" % n)
        nt += sum(1 for r in filas if r["split"] == "test")
        ns += sum(1 for r in filas if r["split"] == "train")
    (sal / "conteo.txt").write_text("%d %d %d\n" % (len(rutas), nt, ns), encoding="utf-8")
    print("abiertas %d selladas (test=%d soporte=%d); registro nuevo: %s" % (len(rutas), nt, ns, a.registro_nuevo))


def cmd_contar(a):
    dev = lb.cargar_dev(a.datos)
    print(len(dev), sum(len(e) for _, e in dev.values()), sum(len(s) for s, _ in dev.values()))


# ---------------------------------------------------------------- comprobaciones de entrada (en la instancia)
def esperados(crit, modo):
    return crit["selladas_sha256"] if modo == "real" else crit["dev_ensayo_sha256"]


def cmd_verificar(a):
    texto = Path(a.crit).read_text(encoding="utf-8")
    mal = []
    if sha256_fichero(a.crit) != a.crit_sha:
        mal.append("criterio")
    if a.modo == "real" and re.search(r'__[A-Z0-9_]{4,}__', texto):
        mal.append("criterio sin congelar (quedan marcadores __...__)")
    esp = esperados(json.loads(texto), a.modo)
    hay = {f.name: f for f in Path(a.datos).glob("*.jsonl") if not f.name.startswith(".")}
    if a.modo == "real" and sorted(hay) != sorted(esp):
        mal.append("ficheros %s distintos de los %d congelados" % (sorted(hay), len(esp)))
    mal += [n for n, f in hay.items() if n not in esp or sha256_fichero(f) != esp[n]]
    if not hay:
        mal.append("sin datos")
    print("MAL:", mal) if mal else print("criterio y %d ficheros de datos OK (modo %s)" % (len(hay), a.modo))
    sys.exit(1 if mal else 0)


# ---------------------------------------------------------------- rúbrica mecánica
def recortar(txt, n=EJ_MAX):
    t = " ".join((txt or "").split())
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


def rubrica_mecanica(dev):
    """{tarea: {pregunta, niveles: {o: {descripcion, descripcion_origen, ejemplo, fuente_ejemplo}}}} de las tareas score, SOLO desde el soporte de oro."""
    out = {}
    for t, (sop, ev) in sorted(dev.items()):
        if ev[0]["tipo"] != "score":
            continue
        ops = lb.orden(ev[0])
        propias = (sop[0].get("descripciones") or {}) if sop else {}
        niv = {}
        for o in ops:
            filas = sorted((r for r in sop if r["etiqueta"] == o), key=lambda r: clave_natural(r["id"]))
            if not filas:
                raise SystemExit("%s: el soporte no trae ninguna fila del nivel %s" % (t, o))
            d = propias.get(o)
            origen = "opcion" if isinstance(d, str) and d.strip() else "plantilla"
            niv[o] = {"descripcion": d.strip() if origen == "opcion" else "Etiqueta %s = nivel %s de %d." % (o, o, len(ops)), "descripcion_origen": origen,
                      "ejemplo": recortar(filas[0]["texto"]), "fuente_ejemplo": filas[0]["id"]}
        out[t] = {"pregunta": sop[0]["pregunta"], "niveles": niv}
    return out


def cmd_preparar(a):
    crit = cargar_crit(a.crit)
    dev = lb.cargar_dev(a.datos)
    esp = esperados(crit, a.modo)
    if (a.modo == "real" and sorted(dev) != sorted(n[:-6] for n in esp)) or any(t + ".jsonl" not in esp for t in dev):
        raise SystemExit("las tareas de %s no son las congeladas: %s" % (a.datos, sorted(dev)))
    if a.modo == "real":
        for t, (sop, ev) in dev.items():
            m = crit["tareas_selladas"][t]
            if (len(ev), len(sop), len(ev[0]["opciones"])) != (m["n_test"], m["n_soporte"], m["clases"]):
                raise SystemExit("%s: recuentos distintos de los del manifiesto" % t)
    buf = io.StringIO()
    with redirect_stdout(buf):
        lb.cmd_filas(argparse.Namespace(dev=a.datos, salida=a.salida))
    sal = Path(a.salida)
    rub = rubrica_mecanica(dev)
    for t, (sop, ev) in dev.items():
        if t not in rub:
            continue
        por = {o: sum(1 for r in sop if r["etiqueta"] == o) for o in lb.orden(ev[0])}
        if min(por.values()) < crit["umbrales"]["oro_min_por_nivel"]:
            raise SystemExit("%s: menos de %d de oro en algún nivel: %s" % (t, crit["umbrales"]["oro_min_por_nivel"], por))
        for sub, filas in (("test", ev), ("soporte", sop)):
            d = sal / "rub" / sub
            d.mkdir(parents=True, exist_ok=True)
            with open(d / (t + ".jsonl"), "w", encoding="utf-8") as fh:
                for f in filas:
                    g = dict(f)
                    g["descripciones"] = {o: "%s Ejemplo: «%s»" % (rub[t]["niveles"][o]["descripcion"], rub[t]["niveles"][o]["ejemplo"]) for o in f["opciones"]}
                    fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    (sal / "rubrica-mecanica.json").write_text(json.dumps({"regla": crit["rubrica_mecanica"]["regla"], "tareas": rub}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    umb = int(__import__("os").environ.get("CHISPA_TOPK_UMBRAL", crit["topk"]["umbral_opciones"]))
    ch = sorted(t for t, (_, ev) in dev.items() if ev[0]["tipo"] != "score")
    (sal / "choice.txt").write_text(" ".join(ch), encoding="utf-8")
    (sal / "score.txt").write_text(" ".join(sorted(rub)), encoding="utf-8")
    (sal / "topk.txt").write_text(" ".join(t for t in ch if len(dev[t][1][0]["opciones"]) > umb), encoding="utf-8")
    print(buf.getvalue().strip())


# ---------------------------------------------------------------- medir
def buscar(dirs):
    def f(s):
        def g(t):
            for d in dirs:
                p = Path(d) / s / (t + ".jsonl")
                if p.exists():
                    return p
            return None
        return g
    return f


def macro_tipo(c, tipo):
    v = [d["exactitud"] for d in c["tareas"].values() if d["tipo"] == tipo]
    return {"valor": round(float(np.mean(v)), 4), "ic95_tareas": lb.ic_tareas(v), "n_tareas": len(v)} if v else None


def dif_grupo(oks_a, oks_b, tareas):
    """final − base sobre `tareas`: dif macro, IC95 sobre tareas (lb.ic_tareas) y bootstrap por filas dentro de tarea (auditoria.boot_par: [est, lo, hi])."""
    if not tareas:
        return None
    d = {t: oks_a[t] - oks_b[t] for t in tareas}
    m = [float(x.mean()) for x in d.values()]
    return {"dif": round(float(np.mean(m)), 4), "ic95_tareas": lb.ic_tareas(m), "ic95_filas": boot_par(d), "n_tareas": len(m), "por_tarea": {t: round(x, 4) for t, x in zip(d, m)}}


def margenes(dev, f_knn):
    """{tarea: array del margen p1 − p2 del kNN por fila de test (nan si falta)} (la señal de la cascada de E)."""
    out = {}
    for t, (_, ev) in dev.items():
        p = f_knn(t)
        pr = {r["id"]: r for r in lb.leer(p)} if p else {}
        m = []
        for r in ev:
            x = pr.get(r["id"])
            q = sorted((float(v) for v in x["probs"].values()), reverse=True) if x and x.get("probs") else []
            m.append(q[0] - (q[1] if len(q) > 1 else 0.0) if q else float("nan"))
        out[t] = np.array(m)
    return out


def cmd_medir(a):
    crit_bytes = Path(a.crit).read_bytes()
    crit = json.loads(crit_bytes)
    dev = lb.cargar_dev(a.datos)
    dirs = [Path(d) for d in a.pred]
    buscar_s = buscar(dirs)
    ex = ids_ejemplos({"tareas": rubrica_mecanica(dev)})
    # mayoritaria: la clase mayoritaria del soporte de oro (empate: el id menor), sin modelo
    for t, (sop, ev) in dev.items():
        m = lb.mayoritaria(sop, ev)["clase"]
        (dirs[0] / "mayoritaria").mkdir(parents=True, exist_ok=True)
        (dirs[0] / "mayoritaria" / (t + ".jsonl")).write_text("".join(json.dumps({"id": r["id"], "probs": {o: float(o == m) for o in r["opciones"]}, "valida": True, "ms": 0.0}) + "\n" for r in ev), encoding="utf-8")
    out = {"criterio_sha256": hashlib.sha256(crit_bytes).hexdigest(), "modo": a.modo, "cascada_en_final": crit["config"]["cascada_en_final"], "sistemas": {}, "umbrales": {}}
    tipo = {t: ev[0]["tipo"] for t, (_, ev) in dev.items()}
    esp, sin_umbral = {}, set()
    for t, (sop, ev) in dev.items():
        if tipo[t] != "score":
            continue
        f = buscar_s("rub-sop")(t)
        try:
            if f is None:
                raise ValueError("falta rub-sop")
            esp[t] = umbrales(sop, {r["id"]: r for r in lb.leer(f)}, ex)
            out["umbrales"][t] = esp[t]
        except ValueError as e:   # «no medido»: esa tarea queda sin cubrir, no se mide con otra lectura
            sin_umbral.add(t)
            out["umbrales"][t] = str(e)

    def fuente_final(t):
        if tipo[t] == "choice":
            return buscar_s("fs8-knn")(t)
        return None if t in sin_umbral else buscar_s("rub")(t)

    fuentes = {"mayoritaria": buscar_s("mayoritaria"), "knn": buscar_s("knn-e5-k8"), "zs": buscar_s("zs"), "fs8-azar": buscar_s("fs8-azar"), "final": fuente_final}
    oks = {}
    for s, fu in fuentes.items():
        out["sistemas"][s], oks[s] = lb.cifras_sistema(dev, fu, set(), esp if s == "final" else False)
        c = out["sistemas"][s]
        c["macro_score"] = macro_score(c) if any(d["tipo"] == "score" for d in c["tareas"].values()) else None
        c["macro_tipo"] = {k: macro_tipo(c, k) for k in ("choice", "score")}
    # cascada informativa (E): kNN + margen; escala a L3 (el final) si margen < u; u de dev, tal cual
    u = crit["config"]["cascada"]["u"]
    mg = margenes(dev, fuentes["knn"])
    cas = {t: np.where(np.isnan(mg[t]) | (mg[t] < u), oks["final"][t], oks["knn"][t]) for t in dev}
    esc = {t: float((np.isnan(mg[t]) | (mg[t] < u)).mean()) for t in dev}
    out["cascada_informativa"] = {"u": u, "l3": "final", "en_final": crit["config"]["cascada_en_final"],
                                  "por_tarea": {t: {"cascada": round(float(cas[t].mean()), 4), "escalado": round(esc[t], 4), "knn": round(float(oks["knn"][t].mean()), 4),
                                                    "final": round(float(oks["final"][t].mean()), 4), "tipo": tipo[t]} for t in dev},
                                  "macro": {k: {"cascada": round(float(np.mean([cas[t].mean() for t in dev if tipo[t] == k])), 4), "escalado": round(float(np.mean([esc[t] for t in dev if tipo[t] == k])), 4)}
                                            for k in ("choice", "score") if any(tipo[t] == k for t in dev)}}
    if crit["config"]["cascada_en_final"]:   # congelado en falso; si se voltea antes de congelar, el final ES la cascada
        oks["final"] = cas
    grupos = {"choice": [t for t in dev if tipo[t] == "choice"], "score": [t for t in dev if tipo[t] == "score"], "global": list(dev)}
    out["diferencias_final_menos"] = {b: {g: dif_grupo(oks["final"], oks[b], ts) for g, ts in grupos.items()} for b in BASES}
    out["macro_global"] = {s: {"valor": out["sistemas"][s]["macro"]["valor"], "ic95_tareas": out["sistemas"][s]["macro"]["ic95_tareas"]} for s in SISTEMAS}
    cob = {s: {t: out["sistemas"][s]["tareas"][t]["cobertura"] for t in dev} for s in SISTEMAS}
    out["cobertura"] = cob
    falta = sorted("%s/%s" % (s, t) for s in ("final", "zs") for t, c in cob[s].items() if c < 1.0)
    v = {"estado": "no medido" if falta else "medido", "motivo": ("cobertura < 100 %% en: %s" % ", ".join(falta)) if falta else "final y 4B zs cubren el 100 % de las tareas selladas",
         "invalidas": {s: out["sistemas"][s]["invalidas_total"] for s in ("final", "zs")}}
    if not falta:
        v["lectura"] = {g: {"final": _macro(out, "final", g), "zs": _macro(out, "zs", g), "mayoritaria": _macro(out, "mayoritaria", g),
                            "final_menos_zs": out["diferencias_final_menos"]["zs"][g], "final_menos_mayoritaria": out["diferencias_final_menos"]["mayoritaria"][g]} for g in ("choice", "score", "global")}
    out["veredicto"] = v
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("veredicto:", v["estado"], "|", v["motivo"])
    for s in SISTEMAS:
        c = out["sistemas"][s]
        print("%-12s macro %.4f choice %s score %s cobertura %.3f invalidas %d" % (s, c["macro"]["valor"], (c["macro_tipo"]["choice"] or {}).get("valor"), (c["macro_tipo"]["score"] or {}).get("valor"), c["cobertura_total"], c["invalidas_total"]))
    for b in BASES:
        g = out["diferencias_final_menos"][b]["global"]
        print("final - %-12s global %+.4f IC95 tareas %s" % (b, g["dif"], g["ic95_tareas"]))


def _macro(out, s, g):
    c = out["sistemas"][s]
    return c["macro_tipo"][g] if g != "global" else {"valor": c["macro"]["valor"], "ic95_tareas": c["macro"]["ic95_tareas"], "n_tareas": c["macro"]["n_tareas"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sellar")
    for k in ("manifiesto", "selladas", "salida", "motivo", "registro-previo", "registro-nuevo"):
        s.add_argument("--" + k, required=True)
    c = sub.add_parser("contar")
    c.add_argument("--datos", required=True)
    v = sub.add_parser("verificar")
    for k in ("crit", "crit-sha", "datos", "modo"):
        v.add_argument("--" + k, required=True)
    p = sub.add_parser("preparar")
    for k in ("datos", "salida", "crit", "modo"):
        p.add_argument("--" + k, required=True)
    m = sub.add_parser("medir")
    m.add_argument("--datos", required=True)
    m.add_argument("--pred", action="append", required=True)
    m.add_argument("--crit", default=str(AQUI / "crit-r10-apertura.json"))
    m.add_argument("--salida", required=True)
    m.add_argument("--modo", default="real")
    a = ap.parse_args(argv)
    {"sellar": cmd_sellar, "contar": cmd_contar, "verificar": cmd_verificar, "preparar": cmd_preparar, "medir": cmd_medir}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
