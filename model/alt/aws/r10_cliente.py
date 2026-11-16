"""R10-caudal (sobre r9_cliente.py): cliente de carga del servidor Go (/v1/decide), solo biblioteca estándar.

    python3 r9_cliente.py FASE --url http://127.0.0.1:8090 --filas filas.jsonl --salida res/ [--n N] [--clientes K] [--dura S] [--etiqueta X]

FASE:
  a  paridad L3: el servidor corre con `forzar_l3` (L3 para todas las filas). Una pregunta por petición, `--clientes` en paralelo.
     Escribe a-l3.jsonl ({id, conjunto, probs, valida, ms, capa, answer, etiqueta}) y a-resumen.json (exactitud por tarea y agrupada
     ponderada con los pesos de las tareas; por `answer` del servidor y por argmax de probs). Una fila inválida cuenta como fallo.
  b  cascada por defecto del servidor (L1/L2 -> L3 según su decide), 1 cliente. Escribe b-filas.jsonl y b-resumen.json: exactitud,
     fracción por capa y escalada, p50/p95 por capa que contestó (ms del cliente y ms del servidor) y por capa desde /metrics
     (chispa_capa_latencia_segundos, delta antes/después) y total.
  (R10: `--intercalado N` toma N filas repartidas por tarea (una de cada conjunto por turno); en `c`, `--etiqueta X` escribe c-X-*.json(l);
  los percentiles de espera en el semáforo y de latencia de L3 salen de las cabeceras X-Chispa-L3-Espera-Ms y X-Chispa-L3-Ms de cada
  respuesta, no del histograma de /metrics.)
  c  cola: `--clientes` (32) en paralelo durante `--dura` segundos sobre filas al azar. Escribe c-peticiones.jsonl y c-resumen.json:
     p50/p95/p99 total (cliente), espera en la cola de L3 (chispa_l3_espera_segundos) y latencia de L3, códigos HTTP, escaladas
     (incompleta = ErrSinL3), llamadas de L3 por resultado.
  d  tope de tiempo: una fila de massive-es con más opciones, `--rep` (5) veces. Escribe d-<etiqueta>.json con lo que respondió el
     servidor (capa, escalate, escalate_reason, probs, answer, ms) y el tiempo del cliente. Se corre dos veces (otra `--etiqueta`)
     contra servidores con distinto `timeout_fila`.
"""
import argparse
import http.client
import json
import math
import random
import re
import threading
import time
import urllib.parse

PESOS = {"hatecheck-es": 2000, "inyecciones": 116, "jailbreak": 400, "massive-es": 1503, "massive-esc": 1438, "tuits-es": 1194}


def leer(ruta, n=None):
    filas = [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]
    return filas[:n] if n else filas


def pregunta(f):
    ops = f["opciones"]
    desc = f.get("descripciones") or {}
    return {"state": f["texto"], "questions": [{
        "id": "q", "question": f["pregunta"], "kind": f["tipo"],
        "options": [dict({"id": str(o)}, **({"description": desc[o]} if desc.get(o) else {})) for o in ops]}]}


class Cliente:
    """Conexión persistente al servidor; reconecta si la cierran."""

    def __init__(self, url, timeout=300):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.timeout, self.c, self.cab = u.hostname, u.port or 80, timeout, None, {}

    def _con(self):
        if self.c is None:
            self.c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self.c

    def pedir(self, metodo, ruta, cuerpo=None):
        """(código, cuerpo bytes, ms). Un solo reintento si la conexión keep-alive se había cerrado."""
        datos = json.dumps(cuerpo, ensure_ascii=False).encode() if cuerpo is not None else None
        for intento in (0, 1):
            t0 = time.perf_counter()
            try:
                c = self._con()
                c.request(metodo, ruta, body=datos, headers={"Content-Type": "application/json"})
                r = c.getresponse()
                b = r.read()
                self.cab = {k.lower(): v for k, v in r.getheaders()}
                return r.status, b, (time.perf_counter() - t0) * 1000
            except (http.client.HTTPException, OSError):
                self.c = None
                if intento:
                    return 0, b"", (time.perf_counter() - t0) * 1000
        return 0, b"", 0.0

    def decidir(self, f):
        cod, b, ms = self.pedir("POST", "/v1/decide", pregunta(f))
        try:
            r = json.loads(b)["results"][0] if cod == 200 else {}
        except (ValueError, KeyError, IndexError):
            r, cod = {}, -cod or -1
        return cod, r, ms


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    i, j = int(math.floor(k)), int(math.ceil(k))
    return round(xs[i] + (xs[j] - xs[i]) * (k - i), 3)


def resumen_ms(xs):
    return {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99), "media": round(sum(xs) / len(xs), 3) if xs else None}


def argmax(probs, opciones):
    mejor = None
    for o in opciones:
        if o in probs and (mejor is None or probs[o] > probs[mejor]):
            mejor = o
    return mejor


# ---------- /metrics: histogramas por delta ----------
def metricas(cli):
    cod, b, _ = cli.pedir("GET", "/metrics")
    return b.decode("utf-8", "replace") if cod == 200 else ""


def histos(texto, nombre):
    """{etiqueta: {'b': [(le, acumulado)], 'sum': s, 'n': n}} de un histograma Prometheus (etiqueta = capa o '')."""
    out = {}
    pat = re.compile(r"^%s_(bucket|sum|count)(?:\{([^}]*)\})? ([^ ]+)$" % re.escape(nombre), re.M)
    for tipo, lab, val in pat.findall(texto):
        capa = (re.search(r'capa="([^"]*)"', lab) or [None, ""])[1]
        h = out.setdefault(capa, {"b": [], "sum": 0.0, "n": 0})
        if tipo == "bucket":
            h["b"].append((float(re.search(r'le="([^"]*)"', lab).group(1).replace("+Inf", "inf")), float(val)))
        elif tipo == "sum":
            h["sum"] = float(val)
        else:
            h["n"] = int(float(val))
    return out


def delta_histo(a, b):
    """Histograma b - a (mismo conjunto de buckets)."""
    if not b:
        return None
    ba = dict(a["b"]) if a else {}
    return {"b": [(le, v - ba.get(le, 0.0)) for le, v in b["b"]], "sum": b["sum"] - (a["sum"] if a else 0.0), "n": b["n"] - (a["n"] if a else 0)}


def cuantil_histo(h, p):
    """Cuantil p (0-100) en ms por interpolación lineal dentro del bucket; None si vacío. El último bucket abierto devuelve su cota inferior."""
    if not h or h["n"] <= 0:
        return None
    objetivo, prev_le, prev_acc = h["n"] * p / 100, 0.0, 0.0
    for le, acc in h["b"]:
        if acc >= objetivo:
            if math.isinf(le):
                return round(prev_le * 1000, 3)
            frac = (objetivo - prev_acc) / (acc - prev_acc) if acc > prev_acc else 1.0
            return round((prev_le + (le - prev_le) * frac) * 1000, 3)
        prev_le, prev_acc = le, acc
    return None


def resumen_histo(h):
    if not h or h["n"] <= 0:
        return {"n": 0}
    return {"n": h["n"], "media": round(h["sum"] / h["n"] * 1000, 3), "p50": cuantil_histo(h, 50), "p95": cuantil_histo(h, 95), "p99": cuantil_histo(h, 99)}


def deltas(antes, despues, nombre):
    a, d = histos(antes, nombre), histos(despues, nombre)
    return {k: resumen_histo(delta_histo(a.get(k), d[k])) for k in d}


def contador(texto, nombre):
    return {m[0]: float(m[1]) for m in re.findall(r'^%s\{[^}]*?"([^"]*)"\} ([^ ]+)$' % re.escape(nombre), texto, re.M)}


def delta_cont(a, b):
    return {k: b[k] - a.get(k, 0) for k in b}


# ---------- fases ----------
def en_paralelo(filas, k, trabajo):
    """Reparte filas entre k hilos (cola compartida). trabajo(cliente, fila) -> dict; devuelve los dicts en el orden de las filas."""
    out, idx, cerrojo = [None] * len(filas), [0], threading.Lock()

    def hilo():
        c = Cliente(ARGS.url)
        while True:
            with cerrojo:
                i = idx[0]
                idx[0] += 1
            if i >= len(filas):
                return
            out[i] = trabajo(c, filas[i])

    hs = [threading.Thread(target=hilo) for _ in range(k)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    return out


def num(cab, k):
    try:
        return float(cab[k])
    except (KeyError, TypeError, ValueError):
        return None


def fila_res(f, cod, r, ms, cab=None):
    probs = r.get("probs") or {}
    ops = [str(o) for o in f["opciones"]]
    valida = cod == 200 and bool(probs) and all(o in probs for o in ops)
    return {"id": f["id"], "conjunto": f["conjunto"], "probs": probs if valida else None, "valida": valida, "ms": round(ms, 3),
            "capa": r.get("layer"), "ms_servidor": r.get("ms"), "escalate": r.get("escalate"), "motivo": r.get("escalate_reason"),
            "answer": r.get("answer"), "etiqueta": f.get("etiqueta"), "http": cod,
            "l3_espera_ms": num(cab or {}, "x-chispa-l3-espera-ms"), "l3_ms": num(cab or {}, "x-chispa-l3-ms")}


def exactitud(rs, filas):
    """Por tarea y agrupada ponderada (pesos fijos por tarea, renormalizados sobre las presentes)."""
    por = {}
    for r, f in zip(rs, filas):
        t = por.setdefault(f["conjunto"], {"n": 0, "ok_answer": 0, "ok_argmax": 0, "validas": 0})
        t["n"] += 1
        et = str(f.get("etiqueta"))
        if r["valida"]:
            t["validas"] += 1
            t["ok_argmax"] += argmax(r["probs"], [str(o) for o in f["opciones"]]) == et
            t["ok_answer"] += str(r["answer"]) == et
    res = {}
    for c, t in por.items():
        res[c] = {"n": t["n"], "validas": t["validas"], "exactitud_answer": round(t["ok_answer"] / t["n"], 4),
                  "exactitud_argmax": round(t["ok_argmax"] / t["n"], 4), "peso": PESOS.get(c)}
    pes = {c: v for c, v in res.items() if v["peso"]}
    w = sum(v["peso"] for v in pes.values())
    agr = {m: round(sum(v["peso"] * v[m] for v in pes.values()) / w, 4) if w else None for m in ("exactitud_answer", "exactitud_argmax")}
    return {"por_tarea": res, "agrupada_ponderada": agr, "tareas_sin_peso": sorted(set(res) - set(pes)), "n": len(filas)}


def fase_a(filas):
    t0 = time.time()
    rs = en_paralelo(filas, ARGS.clientes, lambda c, f: fila_res(f, *c.decidir(f), c.cab))
    with open(ARGS.salida + "/a-l3.jsonl", "w", encoding="utf-8") as o:
        for r in rs:
            o.write(json.dumps({k: r[k] for k in ("id", "probs", "valida", "ms", "capa")}, ensure_ascii=False) + "\n")
    res = exactitud(rs, filas)
    res.update({"segundos": round(time.time() - t0, 1), "clientes": ARGS.clientes, "ms_cliente": resumen_ms([r["ms"] for r in rs]),
                "capas": contar([r["capa"] for r in rs]), "invalidas": sum(not r["valida"] for r in rs)})
    return res


def contar(xs):
    d = {}
    for x in xs:
        d[str(x)] = d.get(str(x), 0) + 1
    return d


def fase_b(filas):
    cli = Cliente(ARGS.url)
    antes = metricas(cli)
    rs = []
    t0 = time.time()
    for f in filas:
        rs.append(fila_res(f, *cli.decidir(f), cli.cab))
    despues = metricas(cli)
    with open(ARGS.salida + "/b-filas.jsonl", "w", encoding="utf-8") as o:
        for r in rs:
            o.write(json.dumps({k: r[k] for k in ("id", "probs", "valida", "ms", "capa", "ms_servidor", "escalate", "motivo")}, ensure_ascii=False) + "\n")
    # exactitud: con la respuesta aunque la cascada haya escalado (answer = mejor pista); aparte, solo las que contestó una capa
    res = exactitud([dict(r, valida=r["valida"] or bool(r["probs"])) for r in rs], filas)
    contestadas = [(r, f) for r, f in zip(rs, filas) if r["capa"] != "escalar" and r["valida"]]
    res["solo_contestadas"] = exactitud([r for r, _ in contestadas], [f for _, f in contestadas]) if contestadas else None
    capas = contar([r["capa"] for r in rs])
    res.update({"segundos": round(time.time() - t0, 1), "capas": capas, "fraccion_por_capa": {k: round(v / len(rs), 4) for k, v in capas.items()},
                "fraccion_escalada": round(capas.get("escalar", 0) / len(rs), 4),
                "total_ms_cliente": resumen_ms([r["ms"] for r in rs]),
                "por_capa_que_contesta_ms_cliente": {c: resumen_ms([r["ms"] for r in rs if str(r["capa"]) == c]) for c in capas},
                "por_capa_que_contesta_ms_servidor": {c: resumen_ms([r["ms_servidor"] for r in rs if str(r["capa"]) == c and r["ms_servidor"] is not None]) for c in capas},
                "capa_metrics_ms_delta": deltas(antes, despues, "chispa_capa_latencia_segundos"),
                "l3_metrics_ms_delta": deltas(antes, despues, "chispa_l3_latencia_segundos"),
                "l3_llamadas": delta_cont(contador(antes, "chispa_l3_llamadas_total"), contador(despues, "chispa_l3_llamadas_total"))})
    return res


def scalar(texto, nombre):
    m = re.search(r"^%s ([^ ]+)$" % re.escape(nombre), texto, re.M)
    return float(m.group(1)) if m else None


def delta_scalar(a, b, nombre):
    x, y = scalar(a, nombre), scalar(b, nombre)
    return None if y is None else y - (x or 0)


def intercalar(filas, n):
    """n filas repartidas por tarea: una de cada conjunto por turno, en el orden en que aparecen."""
    por, orden = {}, []
    for f in filas:
        if f["conjunto"] not in por:
            orden.append(f["conjunto"])
        por.setdefault(f["conjunto"], []).append(f)
    out, i = [], 0
    while len(out) < n and any(por.values()):
        for c in orden:
            if por[c] and len(out) < n:
                out.append(por[c].pop(0))
        i += 1
    return out


def fase_c(filas):
    cli = Cliente(ARGS.url)
    antes = metricas(cli)
    fin, cerrojo, regs = time.time() + ARGS.dura, threading.Lock(), []

    def hilo(i):
        c, rnd = Cliente(ARGS.url), random.Random(i)
        while time.time() < fin:
            f = filas[rnd.randrange(len(filas))]
            r = fila_res(f, *c.decidir(f), c.cab)
            with cerrojo:
                regs.append({k: r[k] for k in ("id", "ms", "capa", "http", "escalate", "motivo", "valida", "l3_espera_ms", "l3_ms")})

    t0 = time.time()
    hs = [threading.Thread(target=hilo, args=(i,)) for i in range(ARGS.clientes)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    seg = time.time() - t0
    despues = metricas(cli)
    pre = "c-%s" % ARGS.etiqueta if ARGS.etiqueta != "normal" else "c"
    with open(ARGS.salida + "/%s-peticiones.jsonl" % pre, "w", encoding="utf-8") as o:
        for r in regs:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"clientes": ARGS.clientes, "segundos": round(seg, 1), "peticiones": len(regs), "por_segundo": round(len(regs) / seg, 2),
            "total_ms_cliente": resumen_ms([r["ms"] for r in regs if r["http"] == 200]),
            "http": contar([r["http"] for r in regs]), "capas": contar([r["capa"] for r in regs]),
            "escaladas_por_motivo": contar([r["motivo"] for r in regs if r["escalate"]]),
            "invalidas": sum(not r["valida"] for r in regs),
            "espera_semaforo_l3_ms": resumen_ms([r["l3_espera_ms"] for r in regs if r["l3_espera_ms"] is not None]),
            "latencia_l3_ms": resumen_ms([r["l3_ms"] for r in regs if r["l3_ms"] is not None]),
            "espera_histograma_ms": deltas(antes, despues, "chispa_l3_espera_segundos"),  # orientativo: solo para contrastar
            "descartes": delta_cont(contador(antes, "chispa_l3_descartes_total"), contador(despues, "chispa_l3_descartes_total")),
            "reintentos": delta_cont(contador(antes, "chispa_l3_reintentos_total"), contador(despues, "chispa_l3_reintentos_total")),
            "circuito_aperturas": delta_scalar(antes, despues, "chispa_l3_circuito_aperturas_total"),
            "semaforo_capacidad": scalar(despues, "chispa_l3_semaforo_capacidad"),
            "l3_llamadas": delta_cont(contador(antes, "chispa_l3_llamadas_total"), contador(despues, "chispa_l3_llamadas_total"))}


def fase_d(filas):
    cand = [f for f in filas if f["conjunto"] == "massive-es"] or filas
    f = max(cand, key=lambda f: len(f["opciones"]))
    cli = Cliente(ARGS.url)
    antes = metricas(cli)
    reps = []
    for _ in range(ARGS.rep):
        cod, r, ms = cli.decidir(f)
        reps.append({"http": cod, "ms_cliente": round(ms, 1), "capa": r.get("layer"), "escalate": r.get("escalate"), "motivo": r.get("escalate_reason"),
                     "ms_servidor": r.get("ms"), "n_probs": len(r.get("probs") or {}), "answer": r.get("answer"),
                     "prob_max": max((r.get("probs") or {"": None}).values(), default=None),
                     "probs_distintas": len(set(round(v, 6) for v in (r.get("probs") or {}).values()))})
    despues = metricas(cli)
    return {"etiqueta": ARGS.etiqueta, "fila": f["id"], "n_opciones": len(f["opciones"]), "etiqueta_oro": f.get("etiqueta"), "repeticiones": reps,
            "l3_llamadas": delta_cont(contador(antes, "chispa_l3_llamadas_total"), contador(despues, "chispa_l3_llamadas_total")),
            "l3_latencia_ms": deltas(antes, despues, "chispa_l3_latencia_segundos")}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fase", choices="abcd")
    ap.add_argument("--url", required=True)
    ap.add_argument("--filas", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--n", type=int, help="solo las primeras N filas (prueba en seco)")
    ap.add_argument("--intercalado", type=int, help="N filas repartidas por tarea (tras --n si se pasa)")
    ap.add_argument("--clientes", type=int, default=None)
    ap.add_argument("--dura", type=float, default=180, help="fase c: segundos")
    ap.add_argument("--rep", type=int, default=5, help="fase d: repeticiones")
    ap.add_argument("--etiqueta", default="normal", help="fase d: nombre de la variante")
    ARGS = ap.parse_args()
    if ARGS.clientes is None:
        ARGS.clientes = {"a": 16, "b": 1, "c": 32, "d": 1}[ARGS.fase]
    filas = leer(ARGS.filas, ARGS.n)
    if ARGS.intercalado:
        filas = intercalar(filas, ARGS.intercalado)
    res = {"a": fase_a, "b": fase_b, "c": fase_c, "d": fase_d}[ARGS.fase](filas)
    nombre = "d-%s.json" % ARGS.etiqueta if ARGS.fase == "d" else ("c-%s-resumen.json" % ARGS.etiqueta if ARGS.fase == "c" and ARGS.etiqueta != "normal" else "%s-resumen.json" % ARGS.fase)
    json.dump(res, open(ARGS.salida + "/" + nombre, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
