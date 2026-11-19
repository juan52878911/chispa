"""W1: fixture de paridad Go↔Python de la regla de respuesta del camino kNN.

Escribe `server/testdata/w1_score_regla.json`: filas {tipo, opciones, probs, esperada} con `esperada` = `lineas_base_v2.top1(probs, r)` (esp=False: mediana CORAL en
score, argmax en el resto; None si la fila es inválida). Incluye empates, colas EXACTAMENTE en 0,5 y opciones fuera de orden. Determinista (semilla fija).

    python -m model.alt.w1_fixture_score [--salida RUTA] [--comprobar]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb

SALIDA = Path(__file__).resolve().parents[2] / "server" / "testdata" / "w1_score_regla.json"


def casos_a_mano():
    f = []
    # score: colas exactamente en 0,5 (la cola NO pasa de 0,5 → no sube) y empates
    for ops, p in [
        (["1", "2", "3"], [0.5, 0.25, 0.25]),
        (["1", "2", "3"], [0.25, 0.25, 0.5]),
        (["1", "2", "3", "4"], [0.25, 0.25, 0.25, 0.25]),
        (["1", "2", "3", "4"], [0.5, 0.0, 0.0, 0.5]),
        (["1", "2"], [0.5, 0.5]),
        (["1", "2", "3", "4", "5"], [0.1, 0.2, 0.4, 0.2, 0.1]),
        (["1", "2", "3", "4", "5"], [0.0, 0.0, 0.0, 0.0, 1.0]),
        (["1", "2", "3", "4", "5"], [1.0, 0.0, 0.0, 0.0, 0.0]),
        (["1", "2", "3", "4", "5"], [0.3, 0.0, 0.0, 0.0, 0.7]),
        (["1", "2", "3"], [0.0, 0.0, 0.0]),   # inválida
        (["3", "1", "2"], [0.25, 0.5, 0.25]),  # opciones fuera de orden: se ordenan por int
        (["10", "2", "1"], [0.2, 0.3, 0.5]),    # int, no texto: 1 < 2 < 10
        (["0", "1", "2"], [0.4, 0.2, 0.4]),
        (["1", "2", "3"], [3.0, 2.0, 1.0]),      # sin normalizar: suman 6
    ]:
        f.append(("score", ops, p))
    f += [("choice", ["a", "b", "c"], [0.2, 0.5, 0.3]), ("choice", ["a", "b", "c"], [0.4, 0.4, 0.2]),
          ("choice", ["a", "b"], [0.0, 0.0]), ("noul", ["false", "true"], [0.5, 0.5]), ("noul", ["false", "true"], [0.1, 0.9])]
    return f


def casos_aleatorios(n=400, semilla=20261119):
    rs = np.random.RandomState(semilla)
    out = []
    for i in range(n):
        k = int(rs.randint(2, 8))
        if i % 3 == 0:   # probabilidades cuantizadas a 1/8: muchos empates y colas en 0,5
            p = rs.multinomial(8, [1.0 / k] * k) / 8.0
        else:
            p = rs.dirichlet([0.5] * k)
        tipo = "score" if i % 4 else "choice"
        ops = [str(j + 1) for j in range(k)] if tipo == "score" else [chr(97 + j) for j in range(k)]
        if tipo == "score" and i % 5 == 0:
            ops = list(rs.permutation(ops))
        out.append((tipo, [str(o) for o in ops], [float(x) for x in p]))
    return out


def construir():
    filas = []
    for tipo, ops, p in casos_a_mano() + casos_aleatorios():
        r = {"tipo": tipo, "opciones": ops}
        e = lb.top1({o: x for o, x in zip(ops, p)}, r, esp=False)
        filas.append({"tipo": tipo, "opciones": ops, "probs": p, "esperada": e})
    return filas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--comprobar", action="store_true", help="falla si el fichero no coincide con lo que genera hoy lb.top1")
    a = ap.parse_args(argv)
    txt = json.dumps(construir(), ensure_ascii=False, separators=(",", ":")) + "\n"
    if a.comprobar:
        if Path(a.salida).read_text(encoding="utf-8") != txt:
            sys.exit("%s no coincide con lineas_base_v2.top1: regenéralo" % a.salida)
        return
    Path(a.salida).write_text(txt, encoding="utf-8")
    print("escrito %s (%d filas)" % (a.salida, txt.count("},{") + 1))


if __name__ == "__main__":
    main()
