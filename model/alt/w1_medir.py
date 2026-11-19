"""W1 (servidor-ejemplos): cifras de la medida fs8 por el servidor (Go) frente a Python, con el criterio de model/alt/crit-w1-fs8-e2e.json.

    python3 -m model.alt.w1_medir --filas consultas.jsonl --go e2e-w1-filas.jsonl --py DIR_PRED_PY --crit crit-w1-fs8-e2e.json [--salida cifras.json]

--filas    consultas.jsonl de `w1_filas` (las filas medidas, con tipo, opciones y etiqueta).
--go       e2e-<etiqueta>-filas.jsonl de `r10_t5_cliente.py e2e` (una fila por consulta: tarea, id, http, probs, answer).
--py       directorio con <tarea>.jsonl de `llm_logprobs.py` (id, probs, valida).
Las dos lecturas usan la MISMA regla, `lineas_base_v2.top1(probs, r)` con esp=False (mediana CORAL en score, argmax en el resto); la `answer` del servidor solo se da como informativo.
Cobertura = filas con respuesta válida (Go: HTTP 200 con probs; Python: valida) entre las filas medidas; el criterio pide 100 % en las dos.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from model.alt.lineas_base_v2 import evaluar_tarea


def leer_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def macro(acc):
    return round(float(np.mean(list(acc.values()))), 4) if acc else None


def cifras(filas, go, py_dir, crit):
    tol = crit["criterio"]["tolerancia_macro"]
    cob_min = crit["criterio"]["cobertura_minima"]
    por_tarea = {}
    for r in filas:
        por_tarea.setdefault(r["tarea"], []).append(r)
    go_por = {}
    for r in go:
        go_por.setdefault(r["tarea"], {})[r["id"]] = r
    acc_go, acc_py, cob_go, cob_py, acuerdo, ans_srv, por = {}, {}, {}, {}, {}, {}, {}
    for t, ev in sorted(por_tarea.items()):
        preds_go = {i: {"probs": r["probs"], "valida": True} for i, r in go_por.get(t, {}).items() if r.get("http") == 200 and r.get("probs")}
        ruta = Path(py_dir) / (t + ".jsonl")
        preds_py = {r["id"]: r for r in leer_jsonl(ruta)} if ruta.exists() else {}
        ok_g, pred_g, _, n_g = evaluar_tarea(ev, preds_go)
        ok_p, pred_p, _, n_p = evaluar_tarea(ev, preds_py)
        validas_p = sum(1 for r in ev if preds_py.get(r["id"], {}).get("valida"))
        acc_go[t], acc_py[t] = round(float(ok_g.mean()), 4), round(float(ok_p.mean()), 4)
        cob_go[t], cob_py[t] = round(n_g / len(ev), 4), round(validas_p / len(ev), 4)
        acuerdo[t] = round(sum(1 for a, b in zip(pred_g, pred_p) if a is not None and a == b) / len(ev), 4)
        ans_srv[t] = round(sum(1 for r in ev if str(go_por.get(t, {}).get(r["id"], {}).get("answer")) == str(r["etiqueta"])) / len(ev), 4)
        por[t] = {"n": len(ev), "go": acc_go[t], "python": acc_py[t], "dif": round(acc_go[t] - acc_py[t], 4), "cobertura_go": cob_go[t], "cobertura_python": cob_py[t],
                  "acuerdo_fila_a_fila": acuerdo[t]}
    mg, mp = macro(acc_go), macro(acc_py)
    cg = min(cob_go.values()) if cob_go else 0.0
    cp = min(cob_py.values()) if cob_py else 0.0
    dif = None if mg is None or mp is None else round(mg - mp, 4)
    pasa_dif = dif is not None and abs(dif) <= tol + 1e-9
    pasa_cob = cg >= cob_min and cp >= cob_min
    return {"criterio": crit["id"], "n": sum(len(v) for v in por_tarea.values()), "macro_go": mg, "macro_python": mp, "dif": dif, "tolerancia": tol,
            "cobertura_minima_por_tarea": {"go": cg, "python": cp}, "por_tarea": por,
            "condiciones": {"dif_macro": {"pasa": pasa_dif}, "cobertura": {"pasa": pasa_cob}},
            "informativo": {"macro_answer_del_servidor": macro(ans_srv), "answer_del_servidor_por_tarea": ans_srv},
            "pasa_todo": bool(pasa_dif and pasa_cob)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--filas", required=True)
    ap.add_argument("--go", required=True)
    ap.add_argument("--py", required=True)
    ap.add_argument("--crit", required=True)
    ap.add_argument("--salida")
    a = ap.parse_args(argv)
    out = cifras(leer_jsonl(a.filas), leer_jsonl(a.go), a.py, json.load(open(a.crit, encoding="utf-8")))
    txt = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
    if a.salida:
        Path(a.salida).write_text(txt, encoding="utf-8")
    print("macro Go %s, Python %s, dif %s (tolerancia %s), cobertura mínima Go %s Python %s: %s" % (
        out["macro_go"], out["macro_python"], out["dif"], out["tolerancia"], out["cobertura_minima_por_tarea"]["go"], out["cobertura_minima_por_tarea"]["python"],
        "PASA" if out["pasa_todo"] else "NO PASA"))


if __name__ == "__main__":
    main()
