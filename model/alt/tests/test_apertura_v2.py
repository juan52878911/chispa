"""Pruebas de apertura_v2: rúbrica mecánica, apertura única de las selladas (solo con ficheros y registro temporales) y `medir` con datos sintéticos diminutos."""
import hashlib
import json
from pathlib import Path

import pytest

from model.alt import apertura_v2 as ap
from model.alt import lineas_base_v2 as lb

CRIT = json.loads((Path(ap.__file__).parent / "crit-r10-apertura.json").read_text(encoding="utf-8"))


def escribir(p, filas):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas), encoding="utf-8")


def fila(t, tipo, i, split, y, ops, texto=None, desc=None):
    r = {"id": "%s:%s:%d" % (t, split, i), "tarea": t, "tipo": tipo, "pregunta": "¿?", "texto": texto or "texto %s %d" % (split, i), "opciones": list(ops),
         "etiqueta": y, "idioma": "es", "split": split}
    if desc:
        r["descripciones"] = dict(desc)
    return r


def tarea_score(t, desc=None, por_nivel=8, n_test=24, orden_inverso=False):
    ops = ["1", "2", "3"]
    sop = [fila(t, "score", i, "train", ops[i % 3], ops, texto="ejemplo %d " % i + "x" * 300, desc=desc) for i in range(3 * por_nivel)]
    if orden_inverso:
        sop = sop[::-1]
    ev = [fila(t, "score", i, "test", ops[i % 3], ops, desc=desc) for i in range(n_test)]
    return sop + ev


def tarea_choice(t, n_test=24):
    ops = ["a", "b", "c"]
    sop = [fila(t, "choice", i, "train", ops[i % 3], ops) for i in range(24)]
    ev = [fila(t, "choice", i, "test", ops[i % 3], ops) for i in range(n_test)]
    return sop + ev


# ---------------------------------------------------------------- rúbrica mecánica
def test_rubrica_mecanica(tmp_path):
    escribir(tmp_path / "con.jsonl", tarea_score("con", desc={"1": "malo", "2": "regular", "3": "bueno"}, orden_inverso=True))
    escribir(tmp_path / "sin.jsonl", tarea_score("sin"))
    escribir(tmp_path / "cho.jsonl", tarea_choice("cho"))
    rub = ap.rubrica_mecanica(lb.cargar_dev(tmp_path))
    assert sorted(rub) == ["con", "sin"]   # solo score
    n = rub["con"]["niveles"]
    assert n["1"]["descripcion"] == "malo" and n["1"]["descripcion_origen"] == "opcion"
    assert rub["sin"]["niveles"]["2"]["descripcion"] == "Etiqueta 2 = nivel 2 de 3." and rub["sin"]["niveles"]["2"]["descripcion_origen"] == "plantilla"
    # ejemplo = primera fila del soporte del nivel por id en orden natural (aunque el fichero venga al revés): niveles 1,2,3 -> filas 0,1,2
    assert [n[o]["fuente_ejemplo"] for o in "123"] == ["con:train:0", "con:train:1", "con:train:2"]
    assert n["1"]["ejemplo"].endswith("…") and len(n["1"]["ejemplo"]) == ap.EJ_MAX and n["1"]["ejemplo"].startswith("ejemplo 0 xxx")
    assert ap.recortar("corto  y   limpio") == "corto y limpio"


def test_rubrica_orden_natural_con_ids_de_dos_cifras(tmp_path):
    ops = ["1", "2"]
    sop = [fila("t", "score", i, "train", ops[i % 2], ops, texto="f%d" % i) for i in (10, 2, 3, 1, 0, 9)]
    ev = [fila("t", "score", 0, "test", "1", ops)]
    escribir(tmp_path / "t.jsonl", sop + ev)
    rub = ap.rubrica_mecanica(lb.cargar_dev(tmp_path))
    assert rub["t"]["niveles"]["1"]["fuente_ejemplo"] == "t:train:0" and rub["t"]["niveles"]["2"]["fuente_ejemplo"] == "t:train:1"


# ---------------------------------------------------------------- apertura única
def montar_selladas(tmp_path):
    dir_sel = tmp_path / "selladas"
    dir_sel.mkdir()
    tareas = []
    for t, filas in (("a-choice", tarea_choice("a-choice")), ("b-score", tarea_score("b-score"))):
        f = dir_sel / (t + ".jsonl")
        escribir(f, filas)
        tareas.append({"nombre": t, "particion": "sellada", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    man = tmp_path / "manifiesto.json"
    man.write_text(json.dumps({"tareas": tareas + [{"nombre": "dev-x", "particion": "dev", "sha256": "0"}]}), encoding="utf-8")
    return dir_sel, man


def args_sellar(tmp_path, dir_sel, man, previo, nuevo):
    return ap.argparse.Namespace(manifiesto=str(man), selladas=str(dir_sel), salida=str(tmp_path / "datos"), motivo="apertura de prueba con motivo suficiente",
                                 registro_previo=str(previo), registro_nuevo=str(nuevo))


def test_apertura_unica(tmp_path):
    dir_sel, man = montar_selladas(tmp_path)
    previo, nuevo = tmp_path / "previo.jsonl", tmp_path / "nuevo.jsonl"
    previo.write_text("", encoding="utf-8")
    ap.cmd_sellar(args_sellar(tmp_path, dir_sel, man, previo, nuevo))
    lineas = [json.loads(l) for l in nuevo.read_text(encoding="utf-8").splitlines()]
    assert sorted(l["fichero"] for l in lineas) == ["a-choice.jsonl", "b-score.jsonl"] and all(l["coincide_con_manifiesto"] for l in lineas)
    assert (tmp_path / "datos" / "a-choice.jsonl").read_bytes() == (dir_sel / "a-choice.jsonl").read_bytes()   # reescritas byte a byte
    assert (tmp_path / "datos" / "conteo.txt").read_text().split() == ["2", "48", "48"]
    # segunda apertura: el registro nuevo ya las trae -> se niega y no añade nada
    with pytest.raises(SystemExit, match="apertura única"):
        ap.cmd_sellar(args_sellar(tmp_path, dir_sel, man, previo, nuevo))
    assert len(nuevo.read_text(encoding="utf-8").splitlines()) == 2
    # lo mismo si la apertura figura en el registro previo (el del repo)
    previo2, nuevo2 = tmp_path / "previo2.jsonl", tmp_path / "nuevo2.jsonl"
    previo2.write_text(json.dumps({"fichero": "b-score.jsonl"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="apertura única"):
        ap.cmd_sellar(args_sellar(tmp_path, dir_sel, man, previo2, nuevo2))
    assert not nuevo2.exists()


def test_no_abre_si_el_sha_no_coincide(tmp_path):
    dir_sel, man = montar_selladas(tmp_path)
    (dir_sel / "b-score.jsonl").write_text("{}\n", encoding="utf-8")
    nuevo = tmp_path / "nuevo.jsonl"
    with pytest.raises(SystemExit, match="no coincide"):
        ap.cmd_sellar(args_sellar(tmp_path, dir_sel, man, tmp_path / "no-existe.jsonl", nuevo))
    assert not nuevo.exists()   # nada se abrió ni se registró


# ---------------------------------------------------------------- medir (dev como selladas) con datos sintéticos
def pred(filas, f, ms=1.0):
    return [{"id": r["id"], "probs": f(r), "valida": True, "ms": ms} for r in filas if r["split"] == "test"]


def montar_medir(tmp_path, sin_zs=False):
    datos, P = tmp_path / "datos", tmp_path / "pred"
    datos.mkdir()
    cho, sco = tarea_choice("cho"), tarea_score("sco")
    escribir(datos / "cho.jsonl", cho)
    escribir(datos / "sco.jsonl", sco)
    ok = lambda r: {o: float(o == r["etiqueta"]) for o in r["opciones"]}  # noqa: E731
    ko = lambda r: {o: float(o == r["opciones"][0]) for o in r["opciones"]}  # noqa: E731
    sal = {"knn-e5-k8": {"cho": pred(cho, ko), "sco": pred(sco, ko)}, "fs8-azar": {"cho": pred(cho, ko), "sco": pred(sco, ko)}, "fs8-knn": {"cho": pred(cho, ok)},
           "rub": {"sco": pred(sco, ok)}, "rub-sop": {"sco": [{"id": r["id"], "probs": ok(r), "valida": True, "ms": 1.0} for r in sco if r["split"] == "train"]}}
    if not sin_zs:
        sal["zs"] = {"cho": pred(cho, ko), "sco": pred(sco, ko)}
    for s, por in sal.items():
        for t, filas in por.items():
            (P / s).mkdir(parents=True, exist_ok=True)
            escribir(P / s / (t + ".jsonl"), filas)
    c = dict(CRIT)
    c["dev_ensayo_sha256"] = {"cho.jsonl": "x", "sco.jsonl": "x"}
    (tmp_path / "crit.json").write_text(json.dumps(c), encoding="utf-8")
    return datos, P


def medir(tmp_path, datos, P):
    a = ap.argparse.Namespace(datos=str(datos), pred=[str(P)], crit=str(tmp_path / "crit.json"), salida=str(tmp_path / "cifras.json"), modo="ensayo")
    ap.cmd_medir(a)
    return json.loads((tmp_path / "cifras.json").read_text(encoding="utf-8"))


def test_preparar_y_medir_dev_como_selladas(tmp_path, capsys):
    datos, P = montar_medir(tmp_path)
    ap.cmd_preparar(ap.argparse.Namespace(datos=str(datos), salida=str(tmp_path / "t"), crit=str(tmp_path / "crit.json"), modo="ensayo"))
    assert "tareas=2 test=48 soporte=48" in capsys.readouterr().out
    assert (tmp_path / "t" / "score.txt").read_text() == "sco" and (tmp_path / "t" / "choice.txt").read_text() == "cho" and (tmp_path / "t" / "topk.txt").read_text() == ""
    f = (tmp_path / "t" / "rub" / "test" / "sco.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert "Etiqueta 1 = nivel 1 de 3. Ejemplo: «" in json.loads(f)["descripciones"]["1"]
    out = medir(tmp_path, datos, P)
    assert out["veredicto"]["estado"] == "medido" and out["modo"] == "ensayo" and out["cascada_en_final"] is False
    s = out["sistemas"]
    assert s["final"]["macro"]["valor"] == 1.0 and s["final"]["tareas"]["sco"]["qwk"] == 1.0 and s["final"]["cobertura_total"] == 1.0
    assert s["final"]["macro_tipo"]["choice"]["valor"] == 1.0 and s["final"]["macro_tipo"]["score"]["valor"] == 1.0
    assert s["zs"]["macro"]["valor"] == pytest.approx(1 / 3, abs=1e-3) and s["mayoritaria"]["tareas"]["cho"]["exactitud"] == pytest.approx(1 / 3, abs=1e-3)
    d = out["diferencias_final_menos"]["zs"]
    assert d["global"]["dif"] == pytest.approx(2 / 3, abs=1e-3) and len(d["global"]["ic95_tareas"]) == 2 and len(d["global"]["ic95_filas"]) == 3 and d["choice"]["n_tareas"] == 1
    assert set(out["veredicto"]["lectura"]) == {"choice", "score", "global"} and out["cobertura"]["final"] == {"cho": 1.0, "sco": 1.0}
    assert len(out["umbrales"]["sco"]) == 2 and out["cascada_informativa"]["u"] == 0.039217


def test_medir_no_medido_si_falta_el_zs(tmp_path):
    datos, P = montar_medir(tmp_path, sin_zs=True)
    out = medir(tmp_path, datos, P)
    assert out["veredicto"]["estado"] == "no medido" and "zs/cho" in out["veredicto"]["motivo"] and "lectura" not in out["veredicto"]


def test_medir_no_medido_si_el_final_no_cubre(tmp_path):
    datos, P = montar_medir(tmp_path)
    f = P / "fs8-knn" / "cho.jsonl"
    f.write_text("".join(f.read_text(encoding="utf-8").splitlines(True)[:-1]), encoding="utf-8")   # una fila menos
    out = medir(tmp_path, datos, P)
    assert out["veredicto"]["estado"] == "no medido" and "final/cho" in out["veredicto"]["motivo"]
    assert out["sistemas"]["final"]["tareas"]["cho"]["cobertura"] < 1.0


def test_criterio_congelable():
    assert CRIT["config"]["cascada_en_final"] is False and CRIT["config"]["cascada"]["u"] == 0.039217
    assert sum(t["n_test"] for t in CRIT["tareas_selladas"].values()) == 2955 and len(CRIT["selladas_sha256"]) == 8
    assert "__SHA_COMMIT_AL_CONGELAR__" in json.dumps(CRIT)   # el marcador que debe rellenar la sesión principal
