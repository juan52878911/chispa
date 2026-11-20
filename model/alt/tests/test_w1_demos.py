"""W1b: demostraciones fs8 por consulta (kNN): fixture de paridad con Go, el banco falso en formato pool y la comparación de la selección (datos inventados)."""
import json
import subprocess
from pathlib import Path

import numpy as np

from model.alt import w1_banco_falso, w1_demos, w1_filas, w1_fixture_demos
from model.alt.lineas_base_v2_pred import demos_fila
from model.alt.tests import dev_sintetico_qwen


def test_fixture_demos_al_dia():
    w1_fixture_demos.main(["--comprobar"])


def test_demos_fila_regla():
    sim = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    assert demos_fila(sim, ["a", "b", "c", "a"], 2) == [3, 1]                 # más clases que k: los 2 más cercanos, del más lejano al más cercano
    assert demos_fila(np.array([0.9, 0.8, 0.7, 0.1], np.float32), ["a", "a", "a", "b"], 3) == [3, 1, 0]   # k clases o menos: una de cada y el resto por cercanía


def _preparar(tmp_path):
    dev = tmp_path / "dev"
    dev_sintetico_qwen.crear(dev, n_test=30, por_clase=12)
    ent = tmp_path / "ent"
    w1_filas.preparar(str(dev), str(ent), 12)
    w1_banco_falso.main(["--soporte", str(ent / "soporte.jsonl"), "--salida", str(tmp_path / "bancos")])
    return ent, tmp_path / "bancos"


def test_banco_falso_es_pool_con_las_fijas_marcadas(tmp_path):
    ent, ban = _preparar(tmp_path)
    for p in sorted(ban.glob("banco-*.ejemplos.jsonl")):
        filas = [json.loads(l) for l in open(p, encoding="utf-8")]
        t = p.name[len("banco-"):-len(".ejemplos.jsonl")]
        sop = [json.loads(l) for l in open(ent / "soporte" / (t + ".jsonl"), encoding="utf-8")]
        assert [f["id"] for f in filas] == [r["id"] for r in sop] and [f["fila"] for f in filas] == list(range(len(sop)))
        fs = sorted(f["fs"] for f in filas if "fs" in f)
        assert fs == list(range(1, min(8, len(sop)) + 1))
        assert np.load(ban / ("banco-%s.npy" % t)).shape[0] == len(sop)


def test_comparar_iguales_y_distintas(tmp_path):
    ent, ban = _preparar(tmp_path)
    muestra = tmp_path / "muestra.jsonl"
    w1_demos.main(["muestra", "--consultas", str(ent / "consultas.jsonl"), "--por-tarea", "5", "--salida", str(muestra)])
    filas = w1_demos.leer(muestra)
    assert len(filas) == 6 * 5
    # el registro "del servidor" = la elección de Python: acuerdo 1; si se cambia una fila, baja
    reg = []
    for t in sorted({r["tarea"] for r in filas}):
        pool = w1_demos.leer(ban / ("banco-%s.ejemplos.jsonl" % t))
        X = np.load(ban / ("banco-%s.npy" % t))
        ys = [e["etiqueta"] for e in pool]
        for r in [r for r in filas if r["tarea"] == t]:
            S = (w1_banco_falso.vector(r["texto"]) @ X.T)
            reg.append({"id": "q", "estado_sha": w1_demos.sha_estado(r["texto"]), "tarea": t, "demos": [pool[i]["id"] for i in demos_fila(S, ys, 8)]})
    log = tmp_path / "log.jsonl"
    w1_demos.escribir(log, reg)
    sal = tmp_path / "sel.json"
    w1_demos.main(["comparar", "--log", str(log), "--muestra", str(muestra), "--bancos", str(ban), "--hash-dim", "8", "--salida", str(sal)])
    assert json.load(open(sal))["acuerdo"] == 1.0
    reg[0]["demos"] = list(reversed(reg[0]["demos"]))
    w1_demos.escribir(log, reg)
    w1_demos.main(["comparar", "--log", str(log), "--muestra", str(muestra), "--bancos", str(ban), "--hash-dim", "8", "--salida", str(sal)])
    out = json.load(open(sal))
    assert out["iguales"] == out["n"] - 1 and out["acuerdo"] < 1.0
    # escribir: demos por tarea y solo las filas con registro
    w1_demos.main(["escribir", "--log", str(log), "--test", str(ent / "test"), "--salida", str(tmp_path / "demos"), "--salida-test", str(tmp_path / "td")])
    t = reg[0]["tarea"]
    d = {r["id"]: r["ejemplos"] for r in w1_demos.leer(tmp_path / "demos" / (t + ".jsonl"))}
    assert list(d.values())[0] == reg[0]["demos"] and w1_demos.sha_estado(w1_demos.leer(tmp_path / "td" / (t + ".jsonl"))[0]["texto"]) == reg[0]["estado_sha"] and set(d) == {r["id"] for r in w1_demos.leer(tmp_path / "td" / (t + ".jsonl"))}
