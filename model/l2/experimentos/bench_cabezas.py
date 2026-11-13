"""Microbenchmark del coste en CPU de las cabezas de score (E2) por decisión.

    python -m model.l2.experimentos.bench_cabezas [--iter 20000] [--hilos 1]

Una decisión de score = un forward de la cabeza sobre su entrada: un escalar (`coral1`) o el vector de K similitudes MaxSim (las demás),
como en `ModeloL2.forward`. Mide µs por llamada (mediana de 5 tandas de `--iter`) de las cuatro cabezas para K = 4 y 5, en
`torch.no_grad()`, y lo compara con una decisión completa de L2 (e5 + ranuras + torre) cuando hay e5 en la máquina (`--con-e5`).
"""
import argparse
import statistics
import sys
import time

import torch

from model.l2.modelo import CLASES_CABEZA

CABEZAS = CLASES_CABEZA


def medir(h, it, k):
    x = torch.tensor([0.7]) if isinstance(h, CLASES_CABEZA["coral1"]) else torch.linspace(-0.5, 0.9, k).unsqueeze(0)
    tandas = []
    with torch.no_grad():
        for _ in range(200):
            h(x)
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(it):
                h(x)
            tandas.append((time.perf_counter() - t0) / it * 1e6)
    return statistics.median(tandas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iter", type=int, default=20000)
    ap.add_argument("--hilos", type=int, default=1)
    ap.add_argument("--con-e5", action="store_true", help="mide también una decisión completa de L2 (necesita e5)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.hilos)
    res = {}
    for k in (4, 5):
        for nombre, cls in CABEZAS.items():
            res[(nombre, k)] = medir(cls(k).eval(), a.iter, k)
    print("µs por decisión (forward de la cabeza, %d hilo(s), mediana de 5 tandas de %d)" % (a.hilos, a.iter))
    print("%-12s %8s %8s %10s" % ("cabeza", "K=4", "K=5", "vs coral"))
    for nombre in CABEZAS:
        print("%-12s %8.1f %8.1f %9.2fx" % (nombre, res[(nombre, 4)], res[(nombre, 5)], (res[(nombre, 4)] + res[(nombre, 5)]) / (res[("coral1", 4)] + res[("coral1", 5)])))
    for nombre in ("coral", "acumulados", "grueso_fino"):
        print("añadido por decisión de %s sobre coral1: %+.1f µs (K=4), %+.1f µs (K=5)" % (nombre, res[(nombre, 4)] - res[("coral1", 4)], res[(nombre, 5)] - res[("coral1", 5)]))
    if a.con_e5:
        from model.l2.exportar import modelo_aleatorio
        from model.l2.modelo import Codificador
        from model.l2.tests.falso import fila
        m, enc = modelo_aleatorio(), Codificador(max_tok=128)
        f = fila(0, "score", k=4)
        for cab in ("coral1", "acumulados"):
            m.cabeza_score = cab
            m.coral.clear()
            with torch.no_grad():
                m([f], enc, entrenar=False)
                t0 = time.perf_counter()
                for i in range(20):
                    m([dict(f, texto="mensaje distinto número %d con algo de texto para que e5 trabaje" % i)], enc, entrenar=False)
            print("decisión completa de L2 (e5 + ranuras + torre + cabeza %s), score K=4: %.1f ms (textos de torre ya en caché)" % (cab, (time.perf_counter() - t0) / 20 * 1e3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
