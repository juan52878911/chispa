"""T5b: añade el texto de la consulta al fixture `consultas.jsonl` de banco-paridad (que solo trae id, tarea, opciones, etiqueta y la decisión de Python).

    python3 consultas_con_texto.py --fixture DIR_BANCO_PARIDAD --bat DIR1[,DIR2] --salida consultas.jsonl

El texto, la pregunta y el tipo se buscan por id en los `*.jsonl` de las baterías (p3-eval/full/bat y tareas-nuevas/bat en el i7). Salida: el
mismo orden y las mismas filas del fixture (la fila i corresponde al vector i de consultas.npy) con los campos extra `texto`, `pregunta`
y `tipo`. Solo biblioteca estándar; falla si algún id no aparece o si las opciones no coinciden."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--bat", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args(argv)
    por_id = {}
    for d in a.bat.split(","):
        for f in sorted(Path(d).glob("*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    por_id[r["id"]] = r
    n = 0
    with open(a.salida, "w", encoding="utf-8") as o:
        for l in open(Path(a.fixture) / "consultas.jsonl", encoding="utf-8"):
            if not l.strip():
                continue
            c = json.loads(l)
            r = por_id.get(c["id"])
            if r is None:
                sys.exit("sin texto para %s" % c["id"])
            if list(r["opciones"]) != list(c["opciones"]):
                sys.exit("%s: las opciones de la batería y del fixture difieren" % c["id"])
            c.update({"texto": r["texto"], "pregunta": r["pregunta"], "tipo": r.get("tipo", "choice")})
            o.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += 1
    print("%d consultas con texto" % n)


if __name__ == "__main__":
    main()
