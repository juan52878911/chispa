"""W1d: resultados.tgz de la fase B y faseA.tar INVENTADOS, con la estructura de los de verdad, para probar el ud B2 en seco y sus herramientas (datos sintéticos: nada real).

    python3 -m model.alt.w1_fixture_b2 DIR [--n-por-tarea 25]

Escribe en DIR: resultados.tgz (res/ con Go 200 por L3 en todas las filas, demos-log.jsonl, demos/, test-demos/, seleccion.json y py/ con cfpb y congreso válidos, finanzas con 5 ids ausentes y una
con aviso en `error` pero válida, y horeca, jql y usajobs enteras con «Connection refused»), faseA.tar (ent/ real de w1_filas y un onnx/ de relleno que B2 no debe extraer), faseA.json y dev/.
Devuelve {"res_sha256", "fasea_sha256", "faltan": {tarea: n}}.
"""
import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path

from model.alt import w1_demos, w1_filas
from model.alt.tests import dev_sintetico_qwen

FALTAN_FINANZAS = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p, filas):
    w1_demos.escribir(p, filas)


def crear(d, n_por_tarea=25):
    d = Path(d)
    dev, ent, res = d / "dev", d / "ent", d / "res"
    dev_sintetico_qwen.crear(dev, n_test=40, por_clase=12)
    w1_filas.preparar(str(dev), str(ent), n_por_tarea)
    cons = w1_demos.leer(ent / "consultas.jsonl")
    sop = {}
    for r in w1_demos.leer(ent / "soporte.jsonl"):
        sop.setdefault(r["tarea"], []).append(r["id"])
    rnd = random.Random(7)
    log, go, py = [], [], {}
    for r in cons:
        t = r["tarea"]
        log.append({"id": r["id"], "estado_sha": w1_demos.sha_estado(r["texto"]), "tarea": t, "demos": rnd.sample(sop[t], 8)})
        probs = {o: rnd.random() + (1.5 if o == r["etiqueta"] and rnd.random() < 0.6 else 0) for o in r["opciones"]}
        z = sum(probs.values())
        probs = {o: v / z for o, v in probs.items()}
        go.append({"tarea": t, "id": r["id"], "http": 200, "capa": "L3", "probs": probs, "answer": max(probs, key=probs.get)})
        mal = t in ("horeca-resenas", "jql-valor-educativo", "usajobs-nivel")
        py.setdefault(t, []).append({"id": r["id"], "probs": None if mal else probs, "valida": not mal, "raw": "", "intentos": 2 if mal else 1, "ms": 0 if mal else 900,
                                     "error": "<urlopen error [Errno 111] Connection refused>" if mal else ""})
    py["finanzas-documentos"] = py["finanzas-documentos"][:-FALTAN_FINANZAS]
    py["finanzas-documentos"][0]["error"] = "prefijo: aviso con probs"
    jl(res / "demos-log.jsonl", log + log[:12])   # + la autoprueba: el registro repite filas
    jl(res / "e2e-w1-filas.jsonl", go)
    for t in sorted(py):
        jl(res / "py" / (t + ".jsonl"), py[t])
        jl(res / ("e2e-w1-%s-filas.jsonl" % t), [g for g in go if g["tarea"] == t])
    w1_demos.main(["escribir", "--log", str(res / "demos-log.jsonl"), "--test", str(ent / "test"), "--salida", str(res / "demos"), "--salida-test", str(res / "test-demos")])
    (res / "seleccion.json").write_text(json.dumps({"k": 8, "n": 60, "iguales": 60, "sin_registro": 0, "acuerdo": 1.0, "por_tarea": {t: {"n": 10, "iguales": 10, "sin_registro": 0} for t in py}}))
    (res / "estado.tsv").write_text("e2e-servidor\tfixture\t+1s\n")
    with tarfile.open(d / "resultados.tgz", "w:gz") as t:
        t.add(res, arcname="res")
    fa = d / "fa"
    (fa / "onnx").mkdir(parents=True)
    (fa / "onnx" / "relleno.bin").write_bytes(b"x" * 1000)
    for p in ent.rglob("*"):
        if p.is_file():
            (fa / "ent" / p.relative_to(ent)).parent.mkdir(parents=True, exist_ok=True)
            (fa / "ent" / p.relative_to(ent)).write_bytes(p.read_bytes())
    with tarfile.open(d / "faseA.tar", "w") as t:
        t.add(fa, arcname=".")
    ficheros = {str(p.relative_to(fa)): sha(p) for p in sorted(fa.rglob("*")) if p.is_file()}
    (d / "faseA.json").write_text(json.dumps({"fase": "A", "tar_sha256": sha(d / "faseA.tar"), "n_por_tarea": n_por_tarea, "ficheros": ficheros}, indent=1))
    n = n_por_tarea
    return {"res_sha256": sha(d / "resultados.tgz"), "fasea_sha256": sha(d / "faseA.tar"),
            "faltan": {"cfpb-quejas": 0, "congreso-eeuu-partido": 0, "finanzas-documentos": FALTAN_FINANZAS, "horeca-resenas": n, "jql-valor-educativo": n, "usajobs-nivel": n}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--n-por-tarea", type=int, default=25)
    a = ap.parse_args(argv)
    print(json.dumps(crear(a.dir, a.n_por_tarea)))


if __name__ == "__main__":
    main()
