"""Pruebas de model.alt.bateria_v2 sin red: lectores sobre un hub falso en miniatura, reglas de `armar`, partición y registro de aperturas."""
import json

import pytest

from model.alt import bateria_v2 as bv
from model.datos import comun


def _snap(hub, repo):
    d = hub / ("datasets--" + repo.replace("/", "--")) / "snapshots" / "abc"
    d.mkdir(parents=True)
    return d


def test_armar_reglas():
    items = [dict(texto="texto %d de clase %s" % (i, c), etiqueta=c, split=None) for c in "ab" for i in range(40)]
    items += [dict(texto="repetido", etiqueta="a", split=None)] * 3                      # duplicado exacto
    items += [dict(texto="ambiguo", etiqueta="a", split=None), dict(texto="  AMBIGUO ", etiqueta="b", split=None)]   # dos etiquetas
    items += [dict(texto="largo " + "x" * 4000, etiqueta="b", split=None)]
    spec = dict(tarea="t", tipo="choice", pregunta="¿?", idioma="es", opciones=["a", "b"], descripciones={"a": "A", "b": "B"})
    filas, cif = bv.armar("x", items, spec, "repo", "mit", k=8, tope=30)
    sop = [r for r in filas if r["split"] == "train"]
    ev = [r for r in filas if r["split"] == "test"]
    assert len(sop) == 16 and Counter_(sop) == {"a": 8, "b": 8}
    assert len(ev) == 30 and abs(Counter_(ev)["a"] - Counter_(ev)["b"]) <= 2             # estratificado
    textos = [bv.norm(r["texto"]) for r in ev]
    assert len(set(textos)) == len(textos) and "ambiguo" not in textos
    assert not set(textos) & {bv.norm(r["texto"]) for r in sop}
    assert all(len(r["texto"]) <= bv.TOPE_CAR for r in filas)
    assert cif["conflicto_test"] == 2 and cif["n_test"] == 30
    assert all(r["uso_comercial"] and r["opciones"] == ["a", "b"] for r in filas)


def Counter_(filas):
    from collections import Counter
    return dict(Counter(r["etiqueta"] for r in filas))


def test_armar_score_y_split_oficial():
    items = [dict(texto="t%d" % i, etiqueta=str(1 + i % 3), split="train" if i < 30 else "test") for i in range(90)]
    spec = dict(tarea="s", tipo="score", pregunta="¿?", idioma="en", opciones=["1", "2", "3", "4"], descripciones={})
    filas, cif = bv.armar("s", items, spec, "repo", "mit", k=2, tope=500)
    assert filas[0]["opciones"] == ["1", "2", "3"]                                      # el nivel 4 sin test se recorta
    assert comun.score_valido(filas[0]["opciones"])
    assert {r["split"] for r in filas} == {"train", "test"} and cif["n_soporte"] == 6
    ids = [r["id"] for r in filas]
    assert len(set(ids)) == len(ids)


def test_lectores_miniatura(tmp_path):
    hub = tmp_path
    d = _snap(hub, "mrcsgh/horeca-spanish-reviews")
    (d / "dataset.txt").write_text("negocio\ttexto\testrellas\testilo\nBar\tmuy rico\t5\tentusiasta\nPub\tmalo\t1\tcrítico\n", encoding="utf-8")
    items, spec = bv.t_horeca(bv.snapshot(hub, "mrcsgh/horeca-spanish-reviews"))
    assert [i["etiqueta"] for i in items] == ["5", "1"] and spec["tipo"] == "score" and spec["opciones"] == ["1", "2", "3", "4", "5"]
    d = _snap(hub, "MariaIsabel/PROMISE_NFR_translated")
    (d / "PROMISE Dataset in Spanish.txt").write_text("﻿PROJECT,REQUIREMENT,LABEL\n1,El sistema actualizará la pantalla,PE\n1,\"Debe ser seguro, siempre\",SE\n", encoding="utf-8")
    items, spec = bv.t_promise(bv.snapshot(hub, "MariaIsabel/PROMISE_NFR_translated"))
    assert [i["etiqueta"] for i in items] == ["PE", "SE"] and spec["descripciones"]["SE"] == "seguridad"
    d = _snap(hub, "jumafernandez/consultas-unlu")
    (d / "data").mkdir()
    for f in ("train", "test"):
        (d / "data" / (f + ".csv")).write_text("consulta,clase\nhola boleto,Boleto Universitario\n", encoding="utf-8")
    items, spec = bv.t_unlu(bv.snapshot(hub, "jumafernandez/consultas-unlu"))
    assert [i["split"] for i in items] == ["train", "test"] and spec["opciones"] == ["Boleto Universitario"]


def test_particion_determinista_y_equilibrada():
    p = bv.particionar()
    assert p == bv.particionar()
    dev = [n for n, v in p.items() if v == "dev"]
    assert len(dev) == bv.N_DEV and len(p) - len(dev) >= 8
    meta = {t["nombre"]: t for t in bv.TAREAS}
    assert len({meta[n]["dominio"] for n in dev}) == len(dev)                           # dominios distintos en desarrollo
    assert {bv.idioma_de(meta[n]) for n in dev} == {"es", "en"}
    assert {bv.tipo_de(meta[n]) for n in dev} == {"score", "choice"}
    assert len({t["familia"] for t in bv.TAREAS}) == len(bv.TAREAS)                    # una tarea por familia


def test_manifiesto_requisitos():
    assert len(bv.TAREAS) >= 14 and sum(1 for t in bv.TAREAS if bv.tipo_de(t) == "score") >= 3
    assert len({t["dominio"] for t in bv.TAREAS}) >= 6
    for t in bv.TAREAS:
        assert t["url_tarjeta"].startswith("https://huggingface.co/datasets/") and t["licencia_fuente"] and t["repo"] in bv.DESCARGAS
    assert all(d["motivo"] for d in bv.DESCARTADOS)


def test_abrir_sellada(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps({"id": 1}) + "\n", encoding="utf-8")
    reg = tmp_path / "aperturas.jsonl"
    with pytest.raises(PermissionError):
        bv.abrir_sellada(f, "", registro=reg)
    with pytest.raises(PermissionError):
        bv.abrir_sellada(f, "  ", registro=reg)
    assert not reg.exists()                                                              # sin motivo no se anota ni se lee
    assert bv.abrir_sellada(f, "medición final del modelo X", quien="prueba", registro=reg) == [{"id": 1}]
    linea = json.loads(reg.read_text(encoding="utf-8"))
    assert linea["quien"] == "prueba" and linea["motivo"] == "medición final del modelo X" and linea["sha256"] == bv.sha256_fichero(f)
    bv.abrir_sellada(f, "segunda lectura de comprobación", quien="prueba", registro=reg)
    assert len(reg.read_text(encoding="utf-8").splitlines()) == 2
