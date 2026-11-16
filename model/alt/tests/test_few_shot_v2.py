"""Pruebas de model.alt.few_shot_v2 sin red ni torch: preparar con un hub falso, orden de la g5, dev32, vectores falsos y la medición con el criterio."""
import json

import pytest

from model.alt import bateria_v2 as bv
from model.alt import few_shot_v2 as fs


def _fila(corto, split, n, texto, et, ops, tipo="choice"):
    return {"id": "%s:%s:%d" % (corto, split, n), "tarea": "t_" + corto, "tipo": tipo, "pregunta": "¿?", "texto": texto, "opciones": ops, "descripciones": {o: o for o in ops},
            "etiqueta": et, "idioma": "es", "split": split, "fuente": "r", "licencia": "mit", "uso_comercial": True}


def _mundo(tmp_path, monkeypatch):
    """Dos tareas: «ab» (choice, con split oficial, 60 por clase) y «sc» (score sin split, 3 niveles)."""
    dev = tmp_path / "dev"
    dev.mkdir()
    hub = tmp_path / "hub"
    for repo in ("x/ab", "x/sc"):
        (hub / ("datasets--" + repo.replace("/", "--")) / "snapshots" / "abc").mkdir(parents=True)
    items_ab = [dict(texto="ab %s %d" % (c, i), etiqueta=c, split="train" if i < 50 else "test") for c in "ab" for i in range(60)]
    items_sc = [dict(texto="sc %d" % i, etiqueta=str(1 + i % 3), split=None) for i in range(400)]
    monkeypatch.setattr(bv, "TAREAS", [dict(nombre="ab", repo="x/ab", loader=lambda r: (items_ab, None)), dict(nombre="sc", repo="x/sc", loader=lambda r: (items_sc, None))])
    f_ab = [_fila("ab", "train", n, "ab %s %d" % (c, i), c, ["a", "b"]) for n, (c, i) in enumerate([(c, i) for c in "ab" for i in range(8)])]
    f_ab += [_fila("ab", "test", n, "ab %s %d" % (c, 50 + i), c, ["a", "b"]) for n, (c, i) in enumerate([(c, i) for c in "ab" for i in range(5)])]
    f_sc = [_fila("sc", "train", n, "sc %d" % i, str(1 + i % 3), ["1", "2", "3"], "score") for n, i in enumerate(range(24))]
    f_sc += [_fila("sc", "test", n, "sc %d" % (24 + i), str(1 + (24 + i) % 3), ["1", "2", "3"], "score") for n, i in enumerate(range(30))]
    for corto, fl in (("ab", f_ab), ("sc", f_sc)):
        fs.escribir(dev / (corto + ".jsonl"), fl)
    return hub, dev


def test_preparar_reglas(tmp_path, monkeypatch):
    hub, dev = _mundo(tmp_path, monkeypatch)
    cu = fs.preparar(hub, dev, tmp_path / "sal", k_oro=32, max_fondo=10)
    oro, fondo = fs.leer(tmp_path / "sal" / "oro32.jsonl"), fs.leer(tmp_path / "sal" / "fondo.jsonl")
    fuente = fs.leer(tmp_path / "sal" / "fondo-fuente.jsonl")
    test = {fs.norm(r["texto"]) for t in ("ab", "sc") for r in fs.leer(dev / (t + ".jsonl")) if r["split"] == "test"}
    assert not {fs.norm(r["texto"]) for r in oro + fondo} & test                              # nada del test
    for t in ("ab", "sc"):
        ids8 = {r["id"] for r in fs.leer(dev / (t + ".jsonl")) if r["split"] == "train"}
        mios = [r for r in oro if r["id"].split(":")[0] == t]
        assert ids8 <= {r["id"] for r in mios}                                                 # oro32 contiene a los 8 de dev
        assert max(__import__("collections").Counter(r["etiqueta"] for r in mios).values()) <= 32
        assert len({fs.norm(r["texto"]) for r in mios}) == len(mios)
    assert not {fs.norm(r["texto"]) for r in oro} & {fs.norm(r["texto"]) for r in fondo}      # fondo distinto de oro32
    assert all(r["id"].startswith("fondo-v2:") and "etiqueta" not in r and r["conjunto"] in ("ab", "sc") for r in fondo)
    assert [r["conjunto"] for r in fondo[:4]] == ["ab", "sc", "ab", "sc"]                    # intercalado por tarea
    assert len(fuente) == len(fondo) and {r["id"] for r in fuente} == {r["id"] for r in fondo}
    assert cu["ab"]["oro32"] == 64 and cu["sc"]["fondo"] == 10 and cu["ab"]["sintetico"] is False
    # determinista
    fs.preparar(hub, dev, tmp_path / "sal2", k_oro=32, max_fondo=10)
    assert (tmp_path / "sal2" / "fondo.jsonl").read_bytes() == (tmp_path / "sal" / "fondo.jsonl").read_bytes()


def test_sin_train_suficiente(tmp_path, monkeypatch):
    hub, dev = _mundo(tmp_path, monkeypatch)
    items = [dict(texto="sc %d" % i, etiqueta=str(1 + i % 3), split=None) for i in range(54)]   # todo cae en dev
    monkeypatch.setattr(bv, "TAREAS", [dict(nombre="ab", repo="x/ab", loader=lambda r: ([dict(texto="ab %s %d" % (c, i), etiqueta=c, split="train") for c in "ab" for i in range(13)], None)),
                                       dict(nombre="sc", repo="x/sc", loader=lambda r: (items, None))])
    cu = fs.preparar(hub, dev, tmp_path / "sal")
    assert cu["sc"]["fondo"] == 0 and cu["sc"]["oro32"] == 24 and cu["sc"]["aviso"]


def test_orden_g5():
    def f(c, i, n):
        return {"id": "fondo-v2:%s:%d" % (c, i), "conjunto": c, "opciones": [str(k) for k in range(n)], "texto": "t"}
    filas = [f("caro", i, 59) for i in range(500)] + [f(c, i, 5) for c in ("a", "b") for i in range(800)]
    out, cost = fs.ordenar(filas)
    assert len(out) == len(filas) and len({r["id"] for r in out}) == len(filas)
    assert cost["caro"] > 1.0 and cost["a"] == 0.05
    assert {r["conjunto"] for r in out[:200]} == {"a", "b"}                                   # primero las baratas a 100 por clase = 500 (≤ tope)
    hechos, total, t_total, t_usado = fs.estimar(out, cost, 300)
    assert t_usado <= 300 and hechos["caro"] > 0 and hechos["a"] > 0 and total["caro"] == 500
    assert t_total == pytest.approx(500 * cost["caro"] + 1600 * 0.05)


def test_dev32_y_medir(tmp_path, monkeypatch):
    hub, dev = _mundo(tmp_path, monkeypatch)
    fs.preparar(hub, dev, tmp_path / "sal", max_fondo=20)
    d32 = tmp_path / "dev32"
    n = fs.armar_dev32(dev, tmp_path / "sal" / "oro32.jsonl", d32)
    assert n["ab"][1] == 10 and n["sc"][1] == 30
    fondo = fs.leer(tmp_path / "sal" / "fondo.jsonl")
    vec = tmp_path / "vec"
    fs.main(["vectores-falsos", "--bateria", str(d32), "--fondo", str(tmp_path / "sal" / "fondo.jsonl"), "--salida", str(vec)])
    # lp: etiqueta correcta con prob 0,9 en todos salvo uno inválido
    fuente = {r["id"]: r["etiqueta"] for r in fs.leer(tmp_path / "sal" / "fondo-fuente.jsonl")}
    lp = []
    for i, r in enumerate(fondo):
        ops = r["opciones"]
        lp.append({"id": r["id"], "valida": i != 0, "probs": {o: (0.9 if o == fuente[r["id"]] else 0.1 / (len(ops) - 1)) for o in ops}})
    fs.escribir(tmp_path / "lp.jsonl", lp)
    crit = tmp_path / "crit.json"
    sha = fs.escribir_criterio(crit)
    assert sha == fs.sha256_fichero(crit) and json.loads(crit.read_text()) == fs.CRIT
    out = tmp_path / "med.json"
    fs.main(["medir", "--bateria", str(d32), "--vec", str(vec), "--fondo", str(tmp_path / "sal" / "fondo.jsonl"), "--fondo-fuente", str(tmp_path / "sal" / "fondo-fuente.jsonl"),
             "--lp", str(tmp_path / "lp.jsonl"), "--crit", str(crit), "--crit-sha", sha, "--salida", str(out), "--permitir-falso"])
    m = json.loads(out.read_text())
    assert m["veredicto"] in ("PASA", "REFUTADO", "NO PASA") and m["tareas"] == ["ab", "sc"]
    assert m["etiquetador"]["ab"]["acierto_vs_fuente"] == 1.0 and m["etiquetador"]["sc"]["etiquetados"] <= 20
    assert set(m["macro"]) == {s + "/" + k for s in ("knn", "proto") for k in fs.CRIT["soportes"]}
    assert "mas_menos_1" in m["por_tarea"]["sc"]["knn/oro32"] and "mae" in m["por_tarea"]["sc"]["knn/oro8"]
    with pytest.raises(SystemExit):                                                             # criterio con otro sha: se niega
        fs.main(["medir", "--bateria", str(d32), "--vec", str(vec), "--fondo", str(tmp_path / "sal" / "fondo.jsonl"), "--fondo-fuente", str(tmp_path / "sal" / "fondo-fuente.jsonl"),
                 "--lp", str(tmp_path / "lp.jsonl"), "--crit", str(crit), "--crit-sha", "0" * 64, "--salida", str(out), "--permitir-falso"])
    with pytest.raises(SystemExit):                                                             # vectores falsos sin permiso: se niega
        fs.main(["medir", "--bateria", str(d32), "--vec", str(vec), "--fondo", str(tmp_path / "sal" / "fondo.jsonl"), "--fondo-fuente", str(tmp_path / "sal" / "fondo-fuente.jsonl"),
                 "--lp", str(tmp_path / "lp.jsonl"), "--crit", str(crit), "--crit-sha", sha, "--salida", str(out)])


def test_veredicto():
    c = fs.CRIT
    assert fs.veredicto(4.0, {"a": 5, "b": -2.0}, c) == "PASA"
    assert fs.veredicto(6.0, {"a": 9, "b": -2.5}, c) == "NO PASA"                              # una tarea cae más de 2
    assert fs.veredicto(0.9, {"a": 1}, c) == "REFUTADO"
    assert fs.veredicto(2.0, {"a": 2}, c) == "NO PASA"
    assert fs.veredicto(1.0, {"a": 1}, c) == "NO PASA"


def test_criterio_congelado_en_el_repo():
    assert fs.CRIT_RUTA.exists() and json.loads(fs.CRIT_RUTA.read_text(encoding="utf-8")) == fs.CRIT
