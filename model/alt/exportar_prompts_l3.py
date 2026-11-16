"""T5c: referencia de Python del prompt de L3: los mensajes (`llm_logprobs.preparar`) y las etiquetas de filas del fixture de banco-paridad que la cascada escala a L3.

    python3 -m model.alt.exportar_prompts_l3 --fixture DIR_BANCO_PARIDAD --bat DIR1[,DIR2] --salida prompts.jsonl [--por-tarea 25] [--todas]

Elige, de cada tarea, las primeras `--por-tarea` filas con `escala: true` en el fixture (o todas con --todas) y escribe una línea por fila con lo que
necesita el test de paridad del servidor Go: id, tarea, tipo, pregunta, texto, opciones, descripciones (de la batería: el fixture no las trae), y los
`mensajes` (sistema y usuario) y `etiquetas` (orden de las opciones) que genera `llm_logprobs.preparar(fila)` con la fila de la batería, que es
la que usó R5/R7 para etiquetar. Solo biblioteca estándar."""
import argparse
import json
import sys
from pathlib import Path

from model.alt import llm_logprobs as lp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--bat", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--por-tarea", type=int, default=25)
    ap.add_argument("--todas", action="store_true")
    a = ap.parse_args(argv)
    bat = {}
    for d in a.bat.split(","):
        for f in sorted(Path(d).glob("*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    bat[r["id"]] = r
    cont, n = {}, 0
    with open(a.salida, "w", encoding="utf-8") as o:
        for l in open(Path(a.fixture) / "consultas.jsonl", encoding="utf-8"):
            if not l.strip():
                continue
            c = json.loads(l)
            if not c["escala"] or (not a.todas and cont.get(c["tarea"], 0) >= a.por_tarea):
                continue
            f = bat[c["id"]]
            try:
                msgs, etq = lp.preparar(f)
            except ValueError:   # más de 62 opciones: no hay prompt
                continue
            cont[c["tarea"]] = cont.get(c["tarea"], 0) + 1
            o.write(json.dumps({"id": c["id"], "tarea": c["tarea"], "tipo": f["tipo"], "pregunta": f["pregunta"], "texto": f["texto"], "opciones": f["opciones"],
                                "descripciones": f.get("descripciones") or {}, "mensajes": msgs, "etiquetas": list(etq)}, ensure_ascii=False) + "\n")
            n += 1
    print("%d filas: %s" % (n, cont))


if __name__ == "__main__":
    sys.exit(main())
