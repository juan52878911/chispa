"""Techo del LLM en JevBench: exactitud con IC95 sobre los 231 casos, en lectura estricta y permisiva, y frente a L2 y al cruzado.

    python3 llm_techo.py --filas jev_filas.jsonl --llm l3_jevbench.jsonl --base baseline-top1.json [--antes l3_jevbench_parcial.jsonl] --salida DIR

Solo biblioteca estándar. `--llm` es la salida de `llm_jevbench.py` (una línea por caso, en el orden de `--filas`).
  - estricta: la respuesta debe cumplir el contrato de L3 (todos los ids, suma 1±0,05); si no, el caso cuenta como fallo.
  - permisiva: se renormalizan las probabilidades que haya (ids ausentes = 0); solo es fallo si no hay JSON o todo es 0.
`--antes` (la salida libre de la v1, parcial) permite contar cuántos de esos casos pasan de inválidos a válidos con la restricción.
Las respuestas de Jev en JevBench solo sirven para medir (MCA 2.3(b)).
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def leer(r):
    return [json.loads(l) for l in open(r, encoding="utf-8") if l.strip()]


def top1(probs, ops, tipo):
    if not probs or sum(probs.get(o, 0.0) for o in ops) <= 0:
        return None
    if tipo == "score":   # mediana CORAL: primer nivel cuyo acumulado llega a 0,5
        niv = sorted(ops, key=int)
        acc = 0.0
        tot = sum(probs.get(o, 0.0) for o in niv)
        for o in niv:
            acc += probs.get(o, 0.0) / tot
            if acc >= 0.5:
                return o
    return max(ops, key=lambda o: probs.get(o, 0.0))


def permisivo(raw, ops):
    try:
        i, j = raw.find("{"), raw.rfind("}")
        pr = json.loads(raw[i:j + 1])["probs"]
        pr = {k: float(v) for k, v in pr.items() if k in ops and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0}
        s = sum(pr.values())
        return {o: pr.get(o, 0.0) / s for o in ops} if s > 0 else None
    except (ValueError, KeyError, AttributeError, TypeError):
        return None


def ic95(ok, n=2000):
    rng = random.Random(0)
    m = sorted(sum(ok[rng.randrange(len(ok))] for _ in ok) / len(ok) for _ in range(n))
    return [round(sum(ok) / len(ok), 4), round(m[int(0.025 * n)], 4), round(m[int(0.975 * n)] - 0.0, 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--filas", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--antes")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    filas, llm = leer(a.filas), {r["id"]: r for r in leer(a.llm)}
    base = json.load(open(a.base, encoding="utf-8"))
    res = {"casos": len(filas), "con_respuesta_del_llm": sum(1 for f in filas if f["id"] in llm)}
    estricto, perm, invalidos, por_tipo = [], [], 0, {}
    for f in filas:
        l = llm.get(f["id"])
        if l is None:
            estricto.append(0.0)
            perm.append(0.0)
            continue
        if not l["valida"]:
            invalidos += 1
        e = 1.0 if l["valida"] and top1(l["probs"], f["opciones"], f["tipo"]) == f["etiqueta"] else 0.0
        pp = l["probs"] if l["valida"] else permisivo(l.get("raw", ""), f["opciones"])
        p = 1.0 if top1(pp, f["opciones"], f["tipo"]) == f["etiqueta"] else 0.0
        estricto.append(e)
        perm.append(p)
        t = por_tipo.setdefault(f["tarea"], [0, 0, 0])
        t[0] += 1
        t[1] += int(e)
        t[2] += int(p)
    res["invalidas_estricto"] = invalidos
    res["tasa_invalidas_estricto"] = round(invalidos / len(filas), 4)
    res["errores"] = dict(Counter((l["error"][:30] or "ninguno") for l in llm.values()))
    res["exactitud_estricta"] = dict(zip(("acc", "lo", "hi"), ic95(estricto)))
    res["exactitud_permisiva"] = dict(zip(("acc", "lo", "hi"), ic95(perm)))
    for m in ("l2", "ce-fp32-onnx"):
        ok = [1.0 if base[m].get(f["id"], [None])[0] == f["etiqueta"] else 0.0 for f in filas]
        res["referencia_" + m] = dict(zip(("acc", "lo", "hi"), ic95(ok)))
    res["por_tarea"] = {t: {"n": v[0], "estricta": round(v[1] / v[0], 3), "permisiva": round(v[2] / v[0], 3)} for t, v in sorted(por_tipo.items())}
    res["criterio_menor_de_0_60"] = {"estricta": res["exactitud_estricta"]["acc"] < 0.60, "permisiva": res["exactitud_permisiva"]["acc"] < 0.60,
                                     "nota": "si el techo del LLM sale < 0,60, JevBench deja de ser criterio de paso para codificadores"}
    if a.antes:
        ant = {r["id"]: r for r in leer(a.antes)}
        res["frente_a_la_v1_libre"] = {"casos_comunes": sum(1 for i in ant if i in llm),
                                       "invalidos_antes": sum(1 for i, r in ant.items() if i in llm and not r["valida"]),
                                       "invalidos_ahora": sum(1 for i, r in ant.items() if i in llm and not llm[i]["valida"])}
    d = Path(a.salida)
    d.mkdir(parents=True, exist_ok=True)
    (d / "llm_techo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "por_tarea"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
