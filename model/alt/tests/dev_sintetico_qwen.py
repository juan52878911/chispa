"""Un «dev» inventado con los 6 nombres de tareas de la batería v2 (datos sintéticos: palabras sueltas por clase, nada real) para probar el ud, el empaquetador y
las herramientas de L2-qwen3emb sin las baterías de verdad.   python model/alt/tests/dev_sintetico_qwen.py DIR [--test N] [--por-clase K]"""
import argparse
import json
import random
from pathlib import Path

TAREAS = {"cfpb-quejas": ("choice", ["a", "b", "c"]), "congreso-eeuu-partido": ("choice", ["x", "y"]), "finanzas-documentos": ("choice", ["p", "q", "r", "s"]),
          "horeca-resenas": ("score", ["1", "2", "3", "4", "5"]), "jql-valor-educativo": ("score", ["1", "2", "3"]), "usajobs-nivel": ("choice", ["u", "v", "w"])}
PALABRAS = ["alfa", "beta", "gamma", "delta", "épsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "ómicron", "pi", "rho", "sigma", "tau", "ípsilon",
            "phi", "chi", "psi", "omega", "ñandú", "café", "año", "río", "luz", "mar", "sol", "tierra"]


def crear(d, n_test=40, por_clase=12, semilla=3):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(semilla)
    for t, (tipo, ops) in TAREAS.items():
        with open(d / (t + ".jsonl"), "w", encoding="utf-8") as fh:
            filas = [("train", ops[i % len(ops)]) for i in range(por_clase * len(ops))] + [("test", rnd.choice(ops)) for _ in range(n_test)]
            for k, (sp, y) in enumerate(filas):
                base = PALABRAS[(ops.index(y) * 5) % len(PALABRAS):][:6]
                txt = " ".join(rnd.choice(base) if rnd.random() < 0.7 else rnd.choice(PALABRAS) for _ in range(rnd.randint(4, 30)))
                fh.write(json.dumps({"id": "%s:%s:%d" % (t, sp, k), "split": sp, "tipo": tipo, "pregunta": "¿Cuál es la clase de este texto sobre %s?" % t, "texto": txt, "opciones": ops,
                                     "etiqueta": y, "descripciones": {o: "clase %s" % o for o in ops}}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--por-clase", type=int, default=12)
    a = ap.parse_args()
    crear(a.dir, a.test, a.por_clase)
