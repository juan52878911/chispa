"""W1b: demostraciones fs8 por consulta (kNN) en la medida del ud: la muestra para la comprobación independiente, la lectura del registro del servidor y la comparación con Python.

    python3 -m model.alt.w1_demos muestra --consultas consultas.jsonl --por-tarea M --salida muestra.jsonl
    python3 -m model.alt.w1_demos escribir --log demos-log.jsonl --test DIR_TEST --salida DIR_DEMOS --salida-test DIR_TEST_CON_DEMOS
    python3 -m model.alt.w1_demos comparar --log demos-log.jsonl --muestra muestra.jsonl --bancos DIR_BANCOS (--vectores DIR_VQ | --hash-dim N) -k 8 --salida seleccion.json

muestra   las M primeras filas por tarea (por el orden del fichero) en el formato del soporte de `chispa-banco-qwen` (tarea, id, etiqueta, pregunta, tipo, opciones, texto): se codifican con
          esa herramienta (`-k 0`) y salen `banco-<tarea>.npy` con los vectores de las consultas, por la misma ruta de codificación que el servidor.
escribir  del registro del servidor (`chispa-server -demos-log`: {id, tarea, demos: [ids]}) saca, por tarea, `<tarea>.jsonl` con {id, ejemplos: [ids]} (el formato de `llm_logprobs.py --demos`)
          y las filas de test que tienen registro (las demás no se pueden puntuar con las mismas demostraciones y cuentan como sin cobertura en Python).
comparar  la elección INDEPENDIENTE de Python (`lineas_base_v2_pred.demos_fila` sobre los vectores del banco y los de la muestra) frente a las demostraciones del registro del servidor, misma fila
          y mismo orden. Con --hash-dim (solo la prueba en seco) los vectores de la muestra salen de la función hash del codificador de prueba del servidor.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from model.alt.lineas_base_v2_pred import demos_fila


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def escribir(ruta, filas):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_muestra(a):
    cuenta, out = {}, []
    for r in leer(a.consultas):
        if cuenta.get(r["tarea"], 0) < a.por_tarea:
            cuenta[r["tarea"]] = cuenta.get(r["tarea"], 0) + 1
            out.append({k: r[k] for k in ("tarea", "id", "etiqueta", "pregunta", "tipo", "opciones", "texto")})
    escribir(a.salida, out)
    print("muestra %d filas de %d tareas" % (len(out), len(cuenta)))


def registro(ruta):
    """id -> (tarea, demos); si una fila se repite, vale la última."""
    out = {}
    for r in leer(ruta):
        out[(r["tarea"], r["id"])] = r["demos"]
    return out


def cmd_escribir(a):
    reg = registro(a.log)
    n = 0
    for p in sorted(Path(a.test).glob("*.jsonl")):
        t = p.stem
        filas = [r for r in leer(p) if (t, r["id"]) in reg]
        escribir(Path(a.salida) / (t + ".jsonl"), [{"id": r["id"], "ejemplos": reg[(t, r["id"])]} for r in filas])
        escribir(Path(a.salida_test) / (t + ".jsonl"), filas)
        n += len(filas)
    print("demos de %d filas" % n)


def cmd_comparar(a):
    reg = registro(a.log)
    muestra = leer(a.muestra)
    por_tarea = {}
    for r in muestra:
        por_tarea.setdefault(r["tarea"], []).append(r)
    res, tot_ok, tot = {}, 0, 0
    for t, filas in sorted(por_tarea.items()):
        pool = leer(Path(a.bancos) / ("banco-%s.ejemplos.jsonl" % t))
        if any("fila" not in e for e in pool):
            raise SystemExit("banco-%s.ejemplos.jsonl no está en formato pool (sin `fila`)" % t)
        pool.sort(key=lambda e: e["fila"])
        X = np.load(Path(a.bancos) / ("banco-%s.npy" % t)).astype(np.float32)[[e["fila"] for e in pool]]
        ys = [e["etiqueta"] for e in pool]
        if a.hash_dim:
            from model.alt.w1_banco_falso import vector
            Q = np.stack([vector(r["texto"], a.hash_dim) for r in filas])
        else:
            Q = np.load(Path(a.vectores) / ("banco-%s.npy" % t)).astype(np.float32)
            if len(Q) != len(filas):
                raise SystemExit("%s: %d vectores de consulta y %d filas en la muestra" % (t, len(Q), len(filas)))
        S = Q @ X.T
        ok = faltan = 0
        distintas = []
        for j, r in enumerate(filas):
            py = [pool[i]["id"] for i in demos_fila(S[j], ys, a.k)]
            go = reg.get((t, r["id"]))
            if go is None:
                faltan += 1
            elif go == py:
                ok += 1
            else:
                distintas.append({"id": r["id"], "go": go, "python": py})
        res[t] = {"n": len(filas), "iguales": ok, "sin_registro": faltan, "acuerdo": round(ok / len(filas), 4), "distintas": distintas[:3]}
        tot_ok, tot = tot_ok + ok, tot + len(filas)
    out = {"k": a.k, "n": tot, "iguales": tot_ok, "acuerdo": round(tot_ok / tot, 4) if tot else 0.0, "por_tarea": res}
    if a.salida:
        Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("selección de demostraciones: %d de %d iguales a Python (%.4f)" % (tot_ok, tot, out["acuerdo"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("muestra")
    m.add_argument("--consultas", required=True)
    m.add_argument("--por-tarea", type=int, required=True)
    m.add_argument("--salida", required=True)
    e = sp.add_parser("escribir")
    e.add_argument("--log", required=True)
    e.add_argument("--test", required=True)
    e.add_argument("--salida", required=True)
    e.add_argument("--salida-test", required=True)
    c = sp.add_parser("comparar")
    c.add_argument("--log", required=True)
    c.add_argument("--muestra", required=True)
    c.add_argument("--bancos", required=True)
    c.add_argument("--vectores")
    c.add_argument("--hash-dim", type=int, default=0)
    c.add_argument("-k", type=int, default=8)
    c.add_argument("--salida")
    a = ap.parse_args(argv)
    if a.cmd == "comparar" and not (a.vectores or a.hash_dim):
        ap.error("comparar exige --vectores o --hash-dim")
    {"muestra": cmd_muestra, "escribir": cmd_escribir, "comparar": cmd_comparar}[a.cmd](a)


if __name__ == "__main__":
    main()
