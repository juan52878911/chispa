"""R10-t5e2e: cliente de la prueba de punta a punta del kNN de L2 (con codificador e5) y L3 por logprobs. Solo biblioteca estándar.

    python3 r10_t5_cliente.py e2e   --url URL --filas consultas.jsonl --salida DIR [--clientes 16] [--n N] [--ref 0.7233]
    python3 r10_t5_cliente.py carga --url URL --filas consultas.jsonl --salida DIR --clientes 64 --dura 50

`consultas.jsonl` es el fixture de banco-paridad con texto (model/alt/consultas_con_texto.py): id, tarea, tipo, pregunta, texto, opciones,
etiqueta y la decisión de Python (l2, l3, escala, motivo, final). Cada fila va por /v1/decide con su texto real (el servidor codifica con e5).
e2e: exactitud de la cascada por tarea y macro sin hatecheck-es (frente a --ref, la de Python con la l3 del fixture), fracción escalada por tarea
     y por motivo (de /metrics: chispa_knn_escaladas_total), acuerdo con la decisión de escalado de Python y p50/p95/p99 por capa con las cabeceras
     X-Chispa-Cod-Ms (codificador e5), X-Chispa-KNN-Ms, X-Chispa-L3-Espera-Ms y X-Chispa-L3-Ms, más el total del cliente. Escribe e2e-filas.jsonl y e2e-resumen.json.
carga: N clientes durante `--dura` s con filas al azar: peticiones/s, percentiles del total y de cada capa, códigos HTTP, descartes. Escribe carga-cN-resumen.json.
"""
import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r10_cliente import Cliente, contador, delta_cont, delta_scalar, metricas, pct, resumen_ms  # noqa: E402

CABS = {"cod_ms": "x-chispa-cod-ms", "knn_ms": "x-chispa-knn-ms", "l3_espera_ms": "x-chispa-l3-espera-ms", "l3_ms": "x-chispa-l3-ms"}


DESC = Path(__file__).with_name("r10_t5e2e_descripciones.json")


def leer(ruta, n=None):
    """Filas del fixture con texto; si no traen las descripciones de las opciones se completan por tarea (r10_t5e2e_descripciones.json): el prompt de
    L3 las lleva y sin ellas es otro (T5b-e2e: −3,2 puntos de macro por mandar las consultas sin descripciones)."""
    f = [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]
    desc = json.load(open(DESC, encoding="utf-8")) if DESC.exists() else {}
    for r in f:
        if not r.get("descripciones") and desc.get(r["tarea"]):
            r["descripciones"] = desc[r["tarea"]]
    return f[:n] if n else f


def num(cab, k):
    try:
        return float(cab[k])
    except (KeyError, TypeError, ValueError):
        return None


def una(c, f):
    cod, r, ms = c.decidir(f)
    out = {"id": f["id"], "tarea": f["tarea"], "http": cod, "ms": round(ms, 3), "capa": r.get("layer"), "answer": r.get("answer"), "escalate": r.get("escalate"),
           "motivo_http": r.get("escalate_reason"), "etiqueta": f.get("etiqueta"), "escala_py": f.get("escala"), "motivo_py": f.get("motivo"), "l2_py": f.get("l2")}
    for k, h in CABS.items():
        out[k] = num(c.cab, h)
    return out


def en_paralelo(filas, k, url):
    out, idx, cerrojo = [None] * len(filas), [0], threading.Lock()

    def hilo():
        c = Cliente(url)
        while True:
            with cerrojo:
                i = idx[0]
                idx[0] += 1
            if i >= len(filas):
                return
            out[i] = una(c, filas[i])

    hs = [threading.Thread(target=hilo) for _ in range(k)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    return out


def lat(rs, k):
    return resumen_ms([r[k] for r in rs if r[k] is not None])


def exactitud(rs):
    por = {}
    for r in rs:
        t = por.setdefault(r["tarea"], [0, 0])
        t[1] += 1
        t[0] += r["http"] == 200 and r["answer"] is not None and str(r["answer"]) == str(r["etiqueta"])
    acc = {t: round(a / n, 4) for t, (a, n) in por.items()}
    sin = [v for t, v in acc.items() if t != "hatecheck-es"]
    return {"por_tarea": acc, "n_por_tarea": {t: n for t, (_, n) in por.items()}, "macro_sin_hatecheck": round(sum(sin) / len(sin), 4) if sin else None,
            "macro_todas": round(sum(acc.values()) / len(acc), 4) if acc else None}


def escaladas_por_tarea_motivo(antes, despues):
    """{tarea: {motivo: n}} del delta de chispa_knn_escaladas_total{tarea,motivo}."""
    import re
    def leer_(t):
        return {(a, b): float(v) for a, b, v in re.findall(r'^chispa_knn_escaladas_total\{tarea="([^"]*)",motivo="([^"]*)"\} (\S+)$', t, re.M)}
    a, d = leer_(antes), leer_(despues)
    out = {}
    for (t, m), v in d.items():
        out.setdefault(t, {})[m] = v - a.get((t, m), 0)
    return out


def filas_knn(antes, despues):
    import re
    def leer_(t):
        return {k: float(v) for k, v in re.findall(r'^chispa_knn_filas_total\{tarea="([^"]*)"\} (\S+)$', t, re.M)}
    a, d = leer_(antes), leer_(despues)
    return {k: v - a.get(k, 0) for k, v in d.items()}


def e2e(filas, a):
    cli = Cliente(a.url)
    antes = metricas(cli)
    t0 = time.time()
    rs = en_paralelo(filas, a.clientes, a.url)
    seg = time.time() - t0
    despues = metricas(cli)
    pre = "e2e-" + a.etiqueta if a.etiqueta else "e2e"
    with open(a.salida + "/%s-filas.jsonl" % pre, "w", encoding="utf-8") as o:
        for r in rs:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    ex = exactitud(rs)
    esc = escaladas_por_tarea_motivo(antes, despues)
    nk = filas_knn(antes, despues)
    por_tarea = {}
    for t in sorted({r["tarea"] for r in rs}):
        rt = [r for r in rs if r["tarea"] == t]
        a3 = sum(1 for r in rt if r["capa"] in ("L3", "escalar"))
        por_tarea[t] = {"n": len(rt), "fraccion_a_l3": round(a3 / len(rt), 4), "acuerdo_escala_con_python": round(sum(1 for r in rt if (r["capa"] in ("L3", "escalar")) == bool(r["escala_py"])) / len(rt), 4),
                        "motivos_knn": esc.get(t, {}), "filas_knn": nk.get(t)}
    res = {"n": len(rs), "segundos": round(seg, 1), "clientes": a.clientes, "exactitud": ex, "referencia_python_macro_sin_hatecheck": a.ref,
           "diferencia_macro_vs_python": None if ex["macro_sin_hatecheck"] is None else round(ex["macro_sin_hatecheck"] - a.ref, 4),
           "http": {str(k): sum(1 for r in rs if r["http"] == k) for k in {r["http"] for r in rs}},
           "capas": {str(k): sum(1 for r in rs if r["capa"] == k) for k in {r["capa"] for r in rs}},
           "por_tarea": por_tarea, "latencia_ms": {"total_cliente": resumen_ms([r["ms"] for r in rs if r["http"] == 200]), **{k: lat(rs, k) for k in CABS}},
           "l3_llamadas": delta_cont(contador(antes, "chispa_l3_llamadas_total"), contador(despues, "chispa_l3_llamadas_total")),
           "descartes": delta_cont(contador(antes, "chispa_l3_descartes_total"), contador(despues, "chispa_l3_descartes_total")),
           "knn_sin_vector": delta_scalar(antes, despues, "chispa_knn_sin_vector_total"),
           "l3_filas_parciales": delta_scalar(antes, despues, "chispa_l3_filas_parciales_total")}
    if a.timeout_fila_ms:   # filas que agotaron el tope de la fila (parciales con piso o sin L3): su respuesta puede diferir de la de Python por eso, no por el prompt
        tf = [r for r in rs if r["l3_ms"] is not None and r["l3_ms"] >= 0.98 * a.timeout_fila_ms]
        res["filas_por_timeout"] = {"timeout_fila_ms": a.timeout_fila_ms, "n": len(tf), "por_tarea": {t: sum(1 for r in tf if r["tarea"] == t) for t in sorted({r["tarea"] for r in tf})},
                                    "ids": [r["id"] for r in tf][:400]}
    return res, pre + "-resumen.json"


def carga(filas, a):
    cli = Cliente(a.url)
    antes = metricas(cli)
    fin, cerrojo, regs = time.time() + a.dura, threading.Lock(), []

    def hilo(i):
        c, rnd = Cliente(a.url), random.Random(i)
        while time.time() < fin:
            r = una(c, filas[rnd.randrange(len(filas))])
            with cerrojo:
                regs.append(r)

    t0 = time.time()
    hs = [threading.Thread(target=hilo, args=(i,)) for i in range(a.clientes)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    seg = time.time() - t0
    despues = metricas(cli)
    ok = [r for r in regs if r["http"] == 200]
    res = {"clientes": a.clientes, "segundos": round(seg, 1), "peticiones": len(regs), "por_segundo": round(len(regs) / seg, 2),
           "http": {str(k): sum(1 for r in regs if r["http"] == k) for k in {r["http"] for r in regs}},
           "capas": {str(k): sum(1 for r in regs if r["capa"] == k) for k in {r["capa"] for r in regs}},
           "latencia_ms": {"total_cliente": resumen_ms([r["ms"] for r in ok]), **{k: lat(regs, k) for k in CABS}},
           "descartes": delta_cont(contador(antes, "chispa_l3_descartes_total"), contador(despues, "chispa_l3_descartes_total")),
           "reintentos": delta_cont(contador(antes, "chispa_l3_reintentos_total"), contador(despues, "chispa_l3_reintentos_total")),
           "circuito_aperturas": delta_scalar(antes, despues, "chispa_l3_circuito_aperturas_total")}
    return res, "carga-c%d-resumen.json" % a.clientes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fase", choices=("e2e", "carga"))
    ap.add_argument("--url", required=True)
    ap.add_argument("--filas", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--clientes", type=int, default=16)
    ap.add_argument("--dura", type=float, default=50)
    ap.add_argument("--etiqueta", default="", help="e2e: nombre de la pasada; escribe e2e-<etiqueta>-filas.jsonl y e2e-<etiqueta>-resumen.json")
    ap.add_argument("--n", type=int)
    ap.add_argument("--muestra", type=int, help="N filas repartidas a intervalos regulares por todo el fichero (todas las tareas)")
    ap.add_argument("--timeout-fila-ms", type=float, default=0, help="timeout_fila del servidor (ms): las filas cuyo tiempo de L3 llega a él se listan como timeout")
    ap.add_argument("--ref", type=float, default=0.7233, help="macro sin hatecheck de la cascada de Python (referencia.json)")
    a = ap.parse_args()
    filas = leer(a.filas, a.n)
    if a.muestra and a.muestra < len(filas):
        filas = filas[::len(filas) // a.muestra][:a.muestra]
    res, nombre = {"e2e": e2e, "carga": carga}[a.fase](filas, a)
    json.dump(res, open(a.salida + "/" + nombre, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
