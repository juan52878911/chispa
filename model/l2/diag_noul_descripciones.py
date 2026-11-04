"""Diagnóstico de noul: ¿la descripción de «false» llega al modelo? (`ModeloL2` con la cabeza `una` solo codifica la de «true».)

    python -m model.l2.diag_noul_descripciones --ckpt mejor.pt --filas hatecheck.jsonl [--tarea hatecheck] [--max-filas N] \\
        [--salida diag_noul.json] [--device cpu]

Sobre las filas noul (por defecto todas las que traiga el jsonl; `--tarea` filtra) compara la predicción del modelo con:
- `original`: las filas tal cual;
- `false_cambiada`: la descripción de «false» sustituida por otra (la de «true» invertida en el orden de las palabras). Con la cabeza
  `una` esa descripción no entra en ningún texto de torre, así que las probabilidades deben salir IDÉNTICAS (si cambian, algo la usa);
- `intercambiada`: las descripciones de «true» y «false» intercambiadas. Un modelo que lea las descripciones debería voltear sus
  predicciones; uno que no las use cambia poco.
Por comparación da: fracción de filas cuya etiqueta predicha (p(true) ≥ 0,5) cambia, diferencia media y máxima de p(true), y cuántas
filas voltean (la etiqueta predicha pasa a ser la contraria). Solo imprime agregados, nunca texto de las filas.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

from . import datos as D


def p_true(modelo, enc, filas, lote=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(filas), lote):
            probs, _ = modelo(filas[i:i + lote], enc, entrenar=False)
            out += [float(p[r["opciones"].index("true")]) for p, r in zip(probs, filas[i:i + lote])]
    return out


def _invertir(texto):
    return " ".join(reversed(str(texto).split())) or "(sin descripción)"


def variante(fila, modo):
    f = copy.deepcopy(fila)
    d = f.setdefault("descripciones", {})
    if modo == "false_cambiada":
        d["false"] = _invertir(d.get("true", "")) + " (otra)"
    elif modo == "intercambiada":
        d["true"], d["false"] = d.get("false", ""), d.get("true", "")
    elif modo != "original":
        raise ValueError(modo)
    return f


def comparar(pa, pb, umbral=0.5):
    """Agregados de p(true) entre dos pasadas sobre las mismas filas."""
    n = len(pa)
    if n == 0:
        return {"n": 0}
    dif = [abs(a - b) for a, b in zip(pa, pb)]
    cambian = [(a >= umbral) != (b >= umbral) for a, b in zip(pa, pb)]
    return {"n": n, "frac_etiqueta_cambia": sum(cambian) / n, "dif_media": sum(dif) / n, "dif_max": max(dif),
            "frac_identicas": sum(d == 0.0 for d in dif) / n, "media_p_true_a": sum(pa) / n, "media_p_true_b": sum(pb) / n}


def diagnosticar(modelo, enc, filas, lote=64):
    filas = [r for r in filas if r["tipo"] == "noul" and "true" in r["opciones"] and "false" in r["opciones"]]
    base = p_true(modelo, enc, filas, lote)
    res = {"cabeza_noul": modelo.cabeza_noul, "n": len(filas)}
    for modo in ("false_cambiada", "intercambiada"):
        res[modo] = comparar(base, p_true(modelo, enc, [variante(r, modo) for r in filas], lote))
    res["lectura"] = ("la descripción de «false» NO llega al modelo" if res["false_cambiada"].get("frac_identicas", 0) == 1.0
                      else "la descripción de «false» cambia las probabilidades")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--filas", nargs="+", required=True, help="jsonl con filas del esquema (tipo noul, opciones false/true, descripciones)")
    ap.add_argument("--tarea", default=None)
    ap.add_argument("--max-filas", type=int, default=0)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    from .evaluar import cargar_modelo
    from .modelo import MAX_TOK, Codificador
    modelo, ck = cargar_modelo(a.ckpt, a.device)
    filas = [r for r in D.leer(a.filas) if r["tipo"] == "noul" and (a.tarea is None or r["tarea"] == a.tarea)]
    if a.max_filas:
        filas = filas[:a.max_filas]
    enc = Codificador(device=a.device, max_tok=ck.get("max_tok", MAX_TOK))
    enc.cargar_enc(ck.get("enc"))
    res = diagnosticar(modelo, enc, filas)
    print("cabeza noul %s, %d filas" % (res["cabeza_noul"], res["n"]))
    for modo in ("false_cambiada", "intercambiada"):
        r = res[modo]
        print("%-15s etiqueta cambia en %.1f %%, |Δp(true)| medio %.4f, máx %.4f, idénticas %.1f %%" % (
            modo, 100 * r.get("frac_etiqueta_cambia", 0), r.get("dif_media", 0), r.get("dif_max", 0), 100 * r.get("frac_identicas", 0)))
    print(res["lectura"])
    if a.salida:
        Path(a.salida).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
