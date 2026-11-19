"""L2-qwen3emb: aplica el criterio congelado (`crit-l2-qwen3emb.json`) a lo que midió el ud. Solo numpy (la lectura de las filas es la de R11).

    python -m model.alt.qwen3emb_medir cifras --res DIR_RES --dev DIR_DEV --crit crit-l2-qwen3emb.json --salida cifras.json

DIR_RES (lo que escribe model/alt/aws/l2qwen3emb_fases.sh):
  paridad-tokens.json, paridad-fp32.json, paridad-int8.json   (test Go de paridad: tokens 200/200 y coseno por texto frente a transformers fp32)
  e2e-<fp32|int8>-v<32|8>-filas.jsonl y -resumen.json          (r10_t5_cliente.py e2e por HTTP; las filas llevan `probs`)
  rss.json {condicion: kB}, tamanos.json {…}, onnx.json (qwen3emb-onnx.json del exportador)
Cada fila del e2e se lee como en R11 (`lineas_base_v2.top1`: argmax, y en `score` la mediana CORAL sobre las probabilidades); el macro es la media de
las exactitudes por tarea. Veredictos: tokens, coseno (fp32), e2e (|macro − R11| ≤ 0,005 con cobertura 100 %) por condición; int8 elegible solo si queda a ≤ 0,005
del fp32 con los mismos vecinos; nota de latencia si la p50 del modo elegido pasa de 150 ms.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from model.alt.lineas_base_v2 import cargar_dev, evaluar_tarea

TOL = 0.005
LAT_MS = 150.0


def leer_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def macro_e2e(filas_jsonl, dev):
    """(macro, por_tarea, cobertura, n) de las filas servidas por HTTP, leídas como R11."""
    por = {}
    for l in open(filas_jsonl, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            por.setdefault(r["tarea"], {})[r["id"]] = r
    acc, cub, n = {}, 0, 0
    for t, (_, ev) in sorted(cargar_dev(dev).items()):
        preds = {i: {"probs": r["probs"], "valida": True} for i, r in por.get(t, {}).items() if r.get("http") == 200 and r.get("probs")}
        if not por.get(t):
            continue
        ok, _, _, presentes = evaluar_tarea(ev, preds)
        acc[t] = round(float(ok.mean()), 4)
        cub += presentes
        n += len(ev)
    macro = round(float(np.mean(list(acc.values()))), 4) if acc else None
    return macro, acc, (cub / n if n else 0.0), n


def lat(res, k):
    d = (res or {}).get("latencia_ms", {}).get(k) or {}
    return {"p50": d.get("p50"), "p95": d.get("p95"), "n": d.get("n")}


def cifras(res, dev, crit, completo=True):
    res, crit = Path(res), json.load(open(crit, encoding="utf-8"))
    ref = crit["referencia_python"]["macro_dev_knn_qwen"]
    out = {"criterio": crit["id"], "condiciones": {}, "informativo": {}}
    tk, p32, p8 = leer_json(res / "paridad-tokens.json"), leer_json(res / "paridad-fp32.json"), leer_json(res / "paridad-int8.json")
    tokens = tk or (p32 or {}).get("tokens")
    out["condiciones"]["tokens"] = {"pasa": bool(tokens and tokens.get("n") == 200 and tokens.get("iguales") == 200), "iguales": (tokens or {}).get("iguales"), "n": (tokens or {}).get("n")}
    cos = (p32 or {}).get("coseno") or {}
    out["condiciones"]["coseno"] = {"pasa": bool(cos) and cos.get("min", 0) >= 0.999 and cos.get("n_bajo_0.999", 1) == 0, "min": cos.get("min"), "n_bajo_0.999": cos.get("n_bajo_0.999")}
    if p8:
        out["informativo"]["coseno_int8"] = p8.get("coseno")
    e2e = {}
    for modo in ("fp32", "int8"):
        for v in (32, 8):
            f = res / ("e2e-%s-v%d-filas.jsonl" % (modo, v))
            nombre = "%s-v%d" % (modo, v)
            if not f.exists():
                out["condiciones"]["e2e-" + nombre] = {"pasa": None, "motivo": "no medido"}
                continue
            m, acc, cob, n = macro_e2e(f, dev)
            r = ref["v%d" % v]
            e2e[nombre] = m
            c = {"macro": m, "referencia_r11": r, "dif": round(m - r, 4), "cobertura": round(cob, 4), "n": n, "por_tarea": acc,
                 "pasa": bool(m is not None and abs(m - r) <= TOL + 1e-9 and cob >= 1.0)}
            rs = leer_json(res / ("e2e-%s-v%d-resumen.json" % (modo, v)))
            c["latencia_ms"] = {"total_cliente": lat(rs, "total_cliente"), "codificador": lat(rs, "cod_ms"), "knn": lat(rs, "knn_ms")}
            out["condiciones"]["e2e-" + nombre] = c
    # int8 solo si queda a ≤ 0,005 del fp32 con los mismos vecinos
    int8 = {}
    for v in (32, 8):
        a, b = e2e.get("fp32-v%d" % v), e2e.get("int8-v%d" % v)
        int8[v] = None if a is None or b is None else abs(b - a) <= TOL + 1e-9
    out["condiciones"]["int8_vs_fp32"] = {"v32": int8[32], "v8": int8[8], "regla": "|macro_int8 − macro_fp32| ≤ 0,005 con los mismos vecinos"}
    # elección: vecinos de mayor macro e2e (fp32) en dev, empate → 32; modo int8 si cumple en ese vecino
    if "fp32-v32" in e2e and "fp32-v8" in e2e:
        vec = 8 if e2e["fp32-v8"] > e2e["fp32-v32"] else 32
        modo = "int8" if int8[vec] else "fp32"
        c = out["condiciones"]["e2e-%s-v%d" % (modo, vec)]
        p50 = (c.get("latencia_ms") or {}).get("total_cliente", {}).get("p50")
        out["eleccion"] = {"vecinos": vec, "modo": modo, "p50_ms": p50,
                           "nota_producto": None if p50 is None or p50 <= LAT_MS else "recomendado con GPU o cuando la latencia no importe"}
    for n in ("rss", "tamanos", "onnx"):
        v = leer_json(res / (n + ".json"))
        if v is not None:
            out["informativo"][n] = v
    req = [out["condiciones"][k]["pasa"] for k in out["condiciones"] if k in ("tokens", "coseno") or k.startswith("e2e-fp32")]
    out["pasa_todo"] = bool(req) and all(x is True for x in req)
    nv = leer_json(res / "int8-no-viable.json")
    if nv is not None:
        out["informativo"]["int8"] = nv
        for v in (32, 8):
            out["condiciones"]["e2e-int8-v%d" % v] = {"pasa": None, "motivo": "int8 no viable"}
    modos = ("fp32",) if nv is not None else ("fp32", "int8")
    out["completo"] = all(out["condiciones"]["e2e-%s-v%d" % (m, v)].get("pasa") is not None for m in modos for v in (32, 8))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["cifras"])
    ap.add_argument("--res", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--crit", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args(argv)
    o = cifras(a.res, a.dev, a.crit)
    Path(a.salida).write_text(json.dumps(o, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("pasa") for k, v in o["condiciones"].items() if isinstance(v, dict) and "pasa" in v}, ensure_ascii=False), "pasa_todo=%s" % o["pasa_todo"])


if __name__ == "__main__":
    main()
