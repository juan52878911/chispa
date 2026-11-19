"""Techo local (W4): ¿un LLM abierto más grande, servido en local, supera al L3 actual (Qwen3-4B-Instruct-2507 por logprobs) en dev v2? Solo biblioteca estándar y numpy.

    python -m model.alt.techo_local preparar --dev DIR_DEV --salida T --rubricas FICHERO|mecanica
    python -m model.alt.techo_local dividir  --origen DIR --destino-b DIR_B --destino-a DIR_A      (partición por auditoria.mitad: B = 1, A = 0)
    python -m model.alt.techo_local bench    --t T --url URL --modelo M --condicion zs|fs8-knn --concurrencia C --n N --salida bench.json [--formato harmony]
    python -m model.alt.techo_local licencia --readme FICHERO [--licencia-fichero F] --modelo M --salida lic.json [--permitidas apache-2.0,mit]
    python -m model.alt.techo_local medir    --dev DIR_DEV --pred DIR_PRED --rubrica rubrica.json --salida cifras.json [--licencias DIR] [--crit crit-techo-local.json]

Sistemas por modelo, en DIR_PRED/<modelo>/<sistema>/<tarea>.jsonl (salida de `techo_local_cliente`):
  zs        zero-shot por logprobs en las 6 tareas (finanzas con las 10 candidatas del kNN-e5, como D); score: mediana CORAL como T3
  fs8-knn   8 demostraciones por fila elegidas por kNN-e5, tareas choice (la configuración que ganó en sellado)
  rub-sop   rúbrica, soporte de oro de las tareas score (solo para ajustar los umbrales)      rub  rúbrica, test de las tareas score
  «fs8» (el sistema que decide) = choice con fs8-knn + score con rub y umbrales isotónicos ajustados con rub-sop (regla de T3b/apertura; las filas-ejemplo de la rúbrica fuera del ajuste).
Medida: exactitud macro de las 6 tareas en la mitad B (la misma partición que D); A y todo dev, informativas. Diferencia pareada frente al 4B de la MISMA corrida con IC95 por
bootstrap sobre filas dentro de cada tarea (auditoria.boot_par, 2000 remuestreos, semilla 0). El caudal sale de `bench` (req/s con C peticiones concurrentes, fs8-knn).
Informativo de dev: confirmarlo exige la batería v3.
"""
import argparse
import hashlib
import io
import json
import re
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb
from model.alt import lineas_base_v2b as lb2b
from model.alt.auditoria import boot_par, mitad

PARTES = {"B": lambda r: mitad(r["id"]) == 1, "A": lambda r: mitad(r["id"]) == 0, "todo": lambda r: True}
CRIT = Path(__file__).with_name("crit-techo-local.json")


def sha256_fichero(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def escribir(path, filas):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas), encoding="utf-8")


# ---------------------------------------------------------------- preparar
def rubrica_de(dev, spec):
    if spec == "mecanica":   # solo pruebas en seco: la regla mecánica de la apertura sobre el soporte
        from model.alt.apertura_v2 import rubrica_mecanica
        return {"tareas": rubrica_mecanica(dev)}
    return json.loads(Path(spec).read_text(encoding="utf-8"))


def cmd_preparar(a):
    dev = lb.cargar_dev(a.dev)
    sal = Path(a.salida)
    buf = io.StringIO()
    with redirect_stdout(buf):
        lb.cmd_filas(argparse.Namespace(dev=a.dev, salida=a.salida))
    rub = rubrica_de(dev, a.rubricas)
    score = sorted(t for t, (_, ev) in dev.items() if ev[0]["tipo"] == "score")
    if sorted(rub["tareas"]) != score:
        raise SystemExit("la rúbrica trae %s y las tareas score de dev son %s" % (sorted(rub["tareas"]), score))
    (sal / "rubrica.json").write_text(json.dumps(rub, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with redirect_stdout(io.StringIO()):
        lb2b.cmd_rubrica(argparse.Namespace(rubricas=str(sal / "rubrica.json"), dev=a.dev, salida=a.salida))
    ch = sorted(t for t, (_, ev) in dev.items() if ev[0]["tipo"] != "score")
    (sal / "choice.txt").write_text(" ".join(ch), encoding="utf-8")
    (sal / "score.txt").write_text(" ".join(score), encoding="utf-8")
    (sal / "topk.txt").write_text(" ".join(t for t in ch if len(dev[t][1][0]["opciones"]) > a.umbral_opciones), encoding="utf-8")
    print(buf.getvalue().strip())


def cmd_dividir(a):
    n = [0, 0]
    for f in sorted(Path(a.origen).glob("*.jsonl")):
        filas = lb.leer(f)
        b = [r for r in filas if PARTES["B"](r)]
        aa = [r for r in filas if PARTES["A"](r)]
        escribir(Path(a.destino_b) / f.name, b)
        escribir(Path(a.destino_a) / f.name, aa)
        n[0] += len(b)
        n[1] += len(aa)
    print("B=%d A=%d" % tuple(n))


# ---------------------------------------------------------------- licencia
def cmd_licencia(a):
    """Lee la clave `license:` del encabezado YAML de la tarjeta del modelo; guarda el texto y su sha256. Sale con 3 si no es de las permitidas."""
    txt = Path(a.readme).read_text(encoding="utf-8")
    cab = re.match(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", txt, re.S)
    m = re.search(r"^license:\s*['\"]?([A-Za-z0-9.\-_]+)['\"]?\s*$", cab.group(1) if cab else "", re.M)
    lic = m.group(1).lower() if m else None
    out = {"modelo": a.modelo, "licencia": lic, "permitidas": a.permitidas.split(","), "tarjeta_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(), "tarjeta_bytes": len(txt.encode("utf-8")),
           "fichero_licencia_sha256": sha256_fichero(a.licencia_fichero) if a.licencia_fichero and Path(a.licencia_fichero).exists() else None}
    out["permitida"] = lic in out["permitidas"]
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("%s: licencia %s -> %s" % (a.modelo, lic, "se mide" if out["permitida"] else "NO se mide"))
    return 0 if out["permitida"] else 3


# ---------------------------------------------------------------- bench: latencia y caudal
def filas_bench(T, n, con_demos=True):
    """n filas choice de la mitad A (reparto por turnos entre las tareas; finanzas y demás tareas con topk, con sus 10 candidatas), sus soportes y sus demos."""
    T = Path(T)
    ch = (T / "choice.txt").read_text().split()
    tk = set((T / "topk.txt").read_text().split())
    por, sop, demos = {}, {}, {}
    for t in ch:
        base = T / "topk" if t in tk else T
        f = base / "testA" / (t + ".jsonl")
        por[t] = lb.leer(f if f.exists() else base / "test" / (t + ".jsonl"))
        sop[t] = lb.leer(base / "soporte" / (t + ".jsonl")) if con_demos else []
        demos[t] = lb.leer(T / "demos" / (t + ".jsonl")) if con_demos else []
    filas, i = [], 0
    while len(filas) < n and any(por[t][i:i + 1] for t in ch):
        filas += [por[t][i] for t in ch if i < len(por[t])]
        i += 1
    filas = filas[:n]
    ids = {r["id"] for r in filas}
    return filas, [r for t in ch for r in sop[t]], [d for t in ch for d in demos[t] if d["id"] in ids]


def pctl(v, q):
    return round(float(np.percentile(v, q)), 1) if len(v) else None


def cmd_bench(a):
    from model.alt import techo_local_cliente as cl
    filas, sop, demos = filas_bench(a.t, a.n, a.condicion == "fs8-knn")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        escribir(d / "filas.jsonl", filas)
        arg = ["--formato", a.formato, "--filas", str(d / "filas.jsonl"), "--modelo", a.modelo, "--salida", str(d / "sal.jsonl"), "--timeout", "600", "--reintentos", "1"]
        arg += sum((["--url", a.url] for _ in range(a.concurrencia)), [])
        if a.condicion == "fs8-knn":
            escribir(d / "soporte.jsonl", sop)
            escribir(d / "demos.jsonl", demos)
            arg += ["--ejemplos", str(d / "soporte.jsonl"), "--demos", str(d / "demos.jsonl"), "--k", "8"]
        t0 = time.perf_counter()
        cl.main(arg)
        seg = time.perf_counter() - t0
        res = lb.leer(d / "sal.jsonl") if (d / "sal.jsonl").exists() else []
    val = [r for r in res if r.get("valida")]
    ms = [r["ms"] for r in val]
    out = {"modelo": a.modelo, "condicion": a.condicion, "concurrencia": a.concurrencia, "n": len(filas), "respondidas": len(res), "validas": len(val), "segundos": round(seg, 2),
           "req_s": round(len(val) / seg, 3) if seg > 0 else 0.0, "p50_ms": pctl(ms, 50), "p95_ms": pctl(ms, 95), "formato": a.formato}
    Path(a.salida).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("bench", json.dumps(out))


# ---------------------------------------------------------------- medir
def fuentes_modelo(P, dev, tipo):
    """(final, zs): funciones tarea -> ruta de predicciones."""
    final = lambda t: P / ("fs8-knn" if tipo[t] == "choice" else "rub") / (t + ".jsonl")  # noqa: E731
    zs = lambda t: P / "zs" / (t + ".jsonl")  # noqa: E731
    return final, zs


def umbrales_modelo(P, dev, tipo, ex):
    esp, sin = {}, {}
    for t, (sop, ev) in dev.items():
        if tipo[t] != "score":
            continue
        f = P / "rub-sop" / (t + ".jsonl")
        try:
            if not f.exists():
                raise ValueError("falta rub-sop")
            esp[t] = lb2b.umbrales(sop, {r["id"]: r for r in lb.leer(f)}, ex)
        except ValueError as e:   # esa tarea queda sin cubrir («no medido»): no se mide con otra lectura
            sin[t] = str(e)
    return esp, sin


def caudal(P, crit):
    c = crit["caudal"]
    f = P / ("bench-%s-c%d.json" % (c["condicion"], c["concurrencia"]))
    return json.loads(f.read_text()) if f.exists() else None


def cmd_medir(a):
    crit_bytes = Path(a.crit).read_bytes()
    crit = json.loads(crit_bytes)
    dev = lb.cargar_dev(a.dev)
    if sorted(dev) != sorted(crit["tareas_dev"]):
        raise SystemExit("las tareas de %s no son las de dev del criterio" % a.dev)
    sint, ref, pred = set(crit["sinteticas"]), crit["referencia"], Path(a.pred)
    tipo = {t: ev[0]["tipo"] for t, (_, ev) in dev.items()}
    ex = lb2b.ids_ejemplos(json.loads(Path(a.rubrica).read_text(encoding="utf-8")))
    out = {"criterio_sha256": hashlib.sha256(crit_bytes).hexdigest(), "referencia": ref, "sinteticas": sorted(sint), "modelos": {}, "pareadas": {}, "veredicto_por_candidato": {}}
    oks = {}
    for m in crit["modelos"]:
        P = pred / m
        if not P.exists():
            out["modelos"][m] = {"estado": "sin predicciones"}
            continue
        esp, sin = umbrales_modelo(P, dev, tipo, ex)
        final, zs = fuentes_modelo(P, dev, tipo)
        ff = lambda t, final=final, sin=sin: Path("/nonexistent") if t in sin else final(t)  # noqa: E731
        o = {"umbrales": {**esp, **sin}, "fs8": {}, "zs": {}}
        for pn, filtro in PARTES.items():
            dv = {t: (s, [r for r in ev if filtro(r)]) for t, (s, ev) in dev.items()}
            for sis, fu, e in (("fs8", ff, esp), ("zs", zs, False)):
                c, ok = lb.cifras_sistema(dv, fu, sint, e)
                c["macro_score"] = lb2b.macro_score(c) if any(d["tipo"] == "score" for d in c["tareas"].values()) else None
                o[sis][pn] = c
                oks[(m, sis, pn)] = ok
        o["caudal"] = caudal(P, crit)
        lic = Path(a.licencias) / (m + ".json") if a.licencias else None
        o["licencia"] = json.loads(lic.read_text()) if lic and lic.exists() else None
        out["modelos"][m] = o
    cand = [m for m in crit["modelos"] if m != ref]
    for m in cand:
        for sis in ("fs8", "zs"):
            for pn in PARTES:
                if (m, sis, pn) in oks and (ref, sis, pn) in oks:
                    d = {t: oks[(m, sis, pn)][t] - oks[(ref, sis, pn)][t] for t in dev}
                    est, lo, hi = boot_par(d, crit["ic95"]["remuestreos"], crit["ic95"]["semilla"])
                    out["pareadas"].setdefault(m, {}).setdefault(sis, {})[pn] = {"macro_dif": est, "ic95": [lo, hi], "por_tarea": {t: round(float(v.mean()), 4) for t, v in d.items()}}
    ch = crit["cambiar"]
    for m in cand:
        v = {"modelo": m, "motivos": [], "faltan": []}
        o = out["modelos"].get(m, {})
        if "fs8" not in o:
            v["faltan"].append("predicciones")
        else:
            if not (o["licencia"] and o["licencia"].get("permitida")):
                v["faltan"].append("licencia (no comprobada o no es Apache-2.0/MIT)")
            cob = min(o["fs8"]["B"]["cobertura_total"], out["modelos"].get(ref, {}).get("fs8", {}).get("B", {}).get("cobertura_total", 0.0))
            v["cobertura_B"] = cob
            if cob < 1.0:
                v["faltan"].append("cobertura en B < 100 %%: %s" % cob)
            cq = o["caudal"]
            if not cq or cq["validas"] < ch["bench_validas_min"] * cq["n"]:
                v["faltan"].append("caudal (bench ausente o con muchas filas inválidas)")
        if v["faltan"]:
            v["veredicto"] = "no medido"
        else:
            p = out["pareadas"][m]["fs8"]["B"]
            cq = out["modelos"][m]["caudal"]
            v.update({"macro_dif": p["macro_dif"], "ic95": p["ic95"], "req_s": cq["req_s"]})
            if p["macro_dif"] < ch["macro_min"]:
                v["motivos"].append("diferencia %+.4f < %+.2f" % (p["macro_dif"], ch["macro_min"]))
            if not p["ic95"][0] > 0:
                v["motivos"].append("el límite inferior del IC95 pareado (%+.4f) no es > 0" % p["ic95"][0])
            if cq["req_s"] < ch["req_s_min"]:
                v["motivos"].append("caudal %.2f req/s < %.1f" % (cq["req_s"], ch["req_s_min"]))
            v["veredicto"] = "CAMBIAR" if not v["motivos"] else "EL 4B SE QUEDA"
        out["veredicto_por_candidato"][m] = v
    ganan = sorted((v for v in out["veredicto_por_candidato"].values() if v["veredicto"] == "CAMBIAR"), key=lambda v: -v["macro_dif"])
    sin = [m for m, v in out["veredicto_por_candidato"].items() if v["veredicto"] == "no medido"]
    out["veredicto"] = {"decision": ("CAMBIAR el L3 a %s" % ganan[0]["modelo"]) if ganan else ("no medido" if len(sin) == len(cand) else "EL 4B SE QUEDA"), "sin_medir": sin,
                        "nota": "informativo de dev: confirmarlo exige la batería v3"}
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("veredicto:", out["veredicto"]["decision"], "| sin medir:", sin or "-")
    for m, o in out["modelos"].items():
        if "fs8" not in o:
            print("%-14s sin predicciones" % m)
            continue
        print("%-14s fs8 B macro %.4f (A %.4f, todo %.4f) | zs B %.4f | cobertura B %.3f | caudal %s req/s" % (m, o["fs8"]["B"]["macro"]["valor"], o["fs8"]["A"]["macro"]["valor"], o["fs8"]["todo"]["macro"]["valor"],
              o["zs"]["B"]["macro"]["valor"], o["fs8"]["B"]["cobertura_total"], (o["caudal"] or {}).get("req_s")))
    for m, v in out["veredicto_por_candidato"].items():
        print("%-14s %s %s" % (m, v["veredicto"], "; ".join(v["motivos"] + v["faltan"])))
        if m in out["pareadas"]:
            p = out["pareadas"][m]["fs8"]["B"]
            print("%-14s fs8 B dif %+.4f IC95 [%+.4f, %+.4f]" % ("", p["macro_dif"], p["ic95"][0], p["ic95"][1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preparar")
    p.add_argument("--dev", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--rubricas", required=True, help="fichero de rúbricas (T3b) o «mecanica» (solo en seco)")
    p.add_argument("--umbral-opciones", type=int, default=20)
    p = sub.add_parser("dividir")
    p.add_argument("--origen", required=True)
    p.add_argument("--destino-b", required=True)
    p.add_argument("--destino-a", required=True)
    p = sub.add_parser("bench")
    p.add_argument("--t", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--modelo", required=True)
    p.add_argument("--condicion", choices=["zs", "fs8-knn"], required=True)
    p.add_argument("--concurrencia", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--formato", choices=["chat", "harmony"], default="chat")
    p = sub.add_parser("licencia")
    p.add_argument("--readme", required=True)
    p.add_argument("--licencia-fichero")
    p.add_argument("--modelo", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--permitidas", default="apache-2.0,mit")
    p = sub.add_parser("medir")
    p.add_argument("--dev", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--rubrica", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--licencias")
    p.add_argument("--crit", default=str(CRIT))
    a = ap.parse_args(argv)
    r = {"preparar": cmd_preparar, "dividir": cmd_dividir, "bench": cmd_bench, "licencia": cmd_licencia, "medir": cmd_medir}[a.cmd](a)
    return r or 0


if __name__ == "__main__":
    sys.exit(main())
