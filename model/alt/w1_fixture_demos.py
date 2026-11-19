"""W1b: fixture de la elección de demostraciones fs8 por consulta (kNN sobre el soporte): `lineas_base_v2_pred.demos_fila` frente a `SeleccionarDemos` de Go.

Escribe server/testdata/w1_demos_knn.json: casos sintéticos con vectores normalizados (float32, redondeados a 6 decimales), consultas y, para cada una, los índices del
soporte que elige Python (en el orden del prompt). Incluye empates exactos (vectores repetidos), soportes con menos de k ejemplos, con k clases o menos y con más.
Determinista.

    python -m model.alt.w1_fixture_demos [--salida FICHERO] [--comprobar]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from model.alt.lineas_base_v2_pred import demos_fila

SALIDA = Path(__file__).resolve().parents[2] / "server" / "testdata" / "w1_demos_knn.json"
# (nombre, filas del soporte, clases, k, dim, filas repetidas a propósito, consultas)
CASOS = [("mas_clases_que_k", 40, 12, 8, 16, 0, 6), ("pocas_clases", 40, 3, 8, 16, 0, 6), ("k_clases_justas", 30, 8, 8, 12, 0, 5),
         ("soporte_menor_que_k", 5, 3, 8, 8, 0, 4), ("empates_exactos", 24, 4, 8, 6, 10, 6), ("dos_clases_sin_un_tipo", 20, 2, 3, 10, 0, 4),
         ("etiquetas_numericas", 36, 11, 8, 14, 4, 5), ("k_uno", 12, 4, 1, 8, 0, 3)]


def etiquetas(n_cl):
    base = ["10", "2", "1", "ñ", "Á", "b", "a", "9", "100", "B", "z", "0"]
    return base[:n_cl]


def casos():
    rng = np.random.default_rng(20261119)
    out = []
    for nombre, n, ncl, k, dim, rep, nq in CASOS:
        cl = etiquetas(ncl)
        ys = [cl[i % ncl] if i < ncl else cl[int(rng.integers(ncl))] for i in range(n)]
        X = rng.normal(size=(n, dim))
        for j in range(rep):   # el soporte repite filas: empates exactos en la similitud
            X[n - 1 - j] = X[j]
        Q = rng.normal(size=(nq, dim))
        if rep:
            Q[0] = X[0]
        X, Q = [(m / np.linalg.norm(m, axis=1, keepdims=True)).round(6).astype(np.float32) for m in (X, Q)]
        S = Q @ X.T
        out.append({"nombre": nombre, "k": k, "ys": ys, "X": X.tolist(), "Q": Q.tolist(), "esperado": [[int(i) for i in demos_fila(S[j], ys, k)] for j in range(nq)]})
    return out


def texto():
    return json.dumps({"casos": casos()}, ensure_ascii=False, separators=(",", ":")) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--comprobar", action="store_true")
    a = ap.parse_args(argv)
    if a.comprobar:
        if Path(a.salida).read_text(encoding="utf-8") != texto():
            sys.exit("%s no coincide con lo que genera hoy Python: regenéralo" % a.salida)
    else:
        Path(a.salida).write_text(texto(), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
