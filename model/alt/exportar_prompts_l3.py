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
    ap.add_argument("--fs-k", type=int, default=0, help="few-shot: escribe también `mensajes_fs` (llm_logprobs con --ejemplos --k K) y --fs-salida con los ejemplos elegidos por tarea")
    ap.add_argument("--fs-semilla", type=int, default=lp.SEMILLA_EJEMPLOS if hasattr(lp, "SEMILLA_EJEMPLOS") else 20261115)
    ap.add_argument("--fs-salida", help="JSON {tarea: [ejemplos]}: los que el servidor lee de <banco>.ejemplos.jsonl")
    a = ap.parse_args(argv)
    bat, oro = {}, {}
    for d in a.bat.split(","):
        for f in sorted(Path(d).glob("*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    bat[r["id"]] = r
                    if r.get("split") == "train":
                        oro.setdefault(f.stem, []).append(r)
    ids = {json.loads(l)["id"] for l in open(Path(a.fixture) / "consultas.jsonl", encoding="utf-8") if l.strip()}
    ejem, turnos = {}, {}
    if a.fs_k:   # el soporte de oro de cada tarea (split train de la batería) sin las filas que se puntúan, como exige llm_logprobs
        for t, sop in oro.items():
            sop = [r for r in sop if r["id"] not in ids]
            if sop:
                ejem[t] = lp.elegir_ejemplos(sop, a.fs_k, a.fs_semilla)
                turnos[t] = lp.turnos_ejemplos(ejem[t], 1200)
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
            fs = lp.con_ejemplos(msgs, turnos[c["tarea"]]) if c["tarea"] in turnos else None
            o.write(json.dumps({"id": c["id"], "tarea": c["tarea"], "tipo": f["tipo"], "pregunta": f["pregunta"], "texto": f["texto"], "opciones": f["opciones"],
                                "descripciones": f.get("descripciones") or {}, "mensajes": msgs, "etiquetas": list(etq), "mensajes_fs": fs}, ensure_ascii=False) + "\n")
            n += 1
    if a.fs_salida:
        cols = ("id", "pregunta", "tipo", "texto", "opciones", "etiqueta")
        json.dump({t: [{k: e[k] for k in cols} for e in es] for t, es in ejem.items()}, open(a.fs_salida, "w", encoding="utf-8"), ensure_ascii=False)
    print("%d filas: %s; ejemplos fs: %s" % (n, cont, {t: len(e) for t, e in ejem.items()}))


if __name__ == "__main__":
    sys.exit(main())
