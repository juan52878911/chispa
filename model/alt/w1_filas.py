"""W1 (servidor-ejemplos): prepara las entradas de la medida fs8 por el servidor frente a Python, desde los 6 jsonl de DESARROLLO de la batería v2 (solo numpy).

    python3 -m model.alt.w1_filas --dev DIR_DEV --salida DIR [--n-por-tarea 300]

Escribe en DIR:
  soporte.jsonl         el soporte de oro de cada tarea, 8 por clase como `muestrear(ys, 8, 0)` de R11 (mismo índice y orden), con tarea, id, etiqueta, pregunta, tipo, opciones y texto:
                        es la entrada de server/cmd/chispa-banco-qwen, que escribe el banco y banco-<tarea>.ejemplos.jsonl (las 8 demostraciones fs8 de L3).
  soporte/<tarea>.jsonl el MISMO soporte por tarea, en el formato de `llm_logprobs.py --ejemplos` (el Python elige de él las mismas 8 demostraciones: misma semilla y mismo orden).
  test/<tarea>.jsonl    hasta N filas de test por tarea (las mismas para el servidor y para Python), elegidas por el hash de su id (determinista, sin depender del orden del fichero).
  consultas.jsonl       esas filas, con tarea, para el cliente r10_t5_cliente.py (servidor).
  tareas.json           {tarea: {preguntas, n_soporte, n_test, tipo}}
"""
import argparse
import hashlib
import json
from pathlib import Path

from model.alt.fusion_knn_l2 import muestrear
from model.alt.lineas_base_v2 import cargar_dev

ORO_POR_CLASE = 8


def escribir(ruta, filas):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def elegir_test(ev, n):
    """Hasta n filas por el hash de su id: la muestra no depende del orden del fichero ni de nada que cambie entre ejecuciones."""
    if n is None or len(ev) <= n:
        return list(ev)
    return sorted(ev, key=lambda r: hashlib.sha256(r["id"].encode("utf-8")).hexdigest())[:n]


def preparar(dev, salida, n_por_tarea=300):
    sal = Path(salida)
    sop_all, cons, tareas = [], [], {}
    for t, (sop, ev) in cargar_dev(dev).items():
        ys = [r["etiqueta"] for r in sop]
        oro = [sop[i] for i in muestrear(ys, ORO_POR_CLASE, 0)]
        test = elegir_test(ev, n_por_tarea)
        escribir(sal / "soporte" / (t + ".jsonl"), oro)
        escribir(sal / "test" / (t + ".jsonl"), test)
        tipo = sorted({r.get("tipo", "choice") for r in sop + ev})
        assert len(tipo) == 1, (t, tipo)
        for r in oro:
            sop_all.append({"tarea": t, "id": r["id"], "etiqueta": r["etiqueta"], "pregunta": r["pregunta"], "tipo": r.get("tipo", "choice"),
                            "opciones": [str(o) for o in r["opciones"]], "texto": r["texto"]})
        for r in test:
            cons.append({"tarea": t, "id": r["id"], "tipo": r.get("tipo", "choice"), "pregunta": r["pregunta"], "texto": r["texto"], "opciones": [str(o) for o in r["opciones"]],
                         "etiqueta": r["etiqueta"], "descripciones": r.get("descripciones") or {}})
        tareas[t] = {"preguntas": sorted({r["pregunta"] for r in sop + ev}), "n_soporte": len(oro), "n_soporte_total": len(sop), "n_test": len(test), "n_test_total": len(ev), "tipo": tipo[0]}
    escribir(sal / "soporte.jsonl", sop_all)
    escribir(sal / "consultas.jsonl", cons)
    (sal / "tareas.json").write_text(json.dumps(tareas, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return "tareas=%d soporte=%d test=%d" % (len(tareas), len(sop_all), len(cons))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--n-por-tarea", type=int, default=300)
    a = ap.parse_args(argv)
    print(preparar(a.dev, a.salida, a.n_por_tarea))


if __name__ == "__main__":
    main()
