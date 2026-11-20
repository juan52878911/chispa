"""W1: fixture de la elección de ejemplos fs8 que hace server/cmd/chispa-banco-qwen en Go (Mersenne Twister de CPython) frente a Python.

Escribe en server/cmd/chispa-banco-qwen/testdata/: `soporte.jsonl` (soporte sintético con tildes, comillas, saltos de línea y filas de test) y, para varias (semilla, k),
`ejemplos-S-K.jsonl`: lo que escribe `generar_banco.py` (`llm_logprobs.elegir_ejemplos` + json.dumps con ensure_ascii=False) sobre ese soporte. Determinista.

    python -m model.alt.w1_fixture_ejemplos [--salida DIR] [--comprobar]
"""
import argparse
import json
import sys
from pathlib import Path

from model.alt import llm_logprobs as lp

SALIDA = Path(__file__).resolve().parents[2] / "server" / "cmd" / "chispa-banco-qwen" / "testdata"
CASOS = [(20261115, 8), (20261115, 3), (7, 8), (2 ** 33 + 5, 8), (0, 5), (20261115, 40)]
COLS = ("id", "pregunta", "tipo", "texto", "opciones", "etiqueta")


def soporte():
    ops = ["a", "b", "c", "d", "e"]
    filas = []
    for i in range(37):
        c = ops[(i * 3 + i // 5) % 5]
        filas.append({"tarea": "t", "id": "r%03d" % ((i * 7) % 37), "etiqueta": c, "pregunta": "¿Cuál es «la» clase?", "tipo": "choice",
                      "texto": "Reseña %d: \"muy bueno\"\ncon salto, tab\t y ñandú ✓ <b>&</b> \\ barra" % i, "opciones": ops})
    for i in range(3):   # filas de test: nunca son oro
        filas.append({"tarea": "t", "id": "x%03d" % i, "etiqueta": "a", "pregunta": "¿Cuál es «la» clase?", "tipo": "choice", "texto": "test %d" % i, "opciones": ops, "split": "test"})
    return filas


def construir():
    s = soporte()
    out = {"soporte.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in s)}
    oro = [r for r in s if r.get("split", "train") == "train"]
    for sem, k in CASOS:
        ejs = lp.elegir_ejemplos(oro, k, sem)
        out["ejemplos-%d-%d.jsonl" % (sem, k)] = "".join(json.dumps({c: e[c] for c in COLS}, ensure_ascii=False) + "\n" for e in ejs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--comprobar", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.salida)
    for n, txt in construir().items():
        if a.comprobar:
            if (d / n).read_text(encoding="utf-8") != txt:
                sys.exit("%s no coincide con lo que genera hoy Python: regenéralo" % (d / n))
        else:
            d.mkdir(parents=True, exist_ok=True)
            (d / n).write_text(txt, encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
