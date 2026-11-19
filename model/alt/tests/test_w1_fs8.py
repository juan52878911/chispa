"""W1: fixtures de paridad con el servidor Go, entradas de la medida fs8 y su lectura (solo datos inventados)."""
import json
from pathlib import Path

import pytest

from model.alt import lineas_base_v2 as lb
from model.alt import w1_filas, w1_fixture_ejemplos, w1_fixture_score, w1_medir
from model.alt.tests import dev_sintetico_qwen

CRIT = Path(__file__).resolve().parents[1] / "crit-w1-fs8-e2e.json"


def test_fixture_score_al_dia():
    """El fixture que lee server/decide/score_regla_test.go es lo que da hoy lineas_base_v2.top1 (si cambia la regla en Python, hay que regenerarlo)."""
    w1_fixture_score.main(["--comprobar"])


def test_fixture_score_cubre_los_casos_dificiles():
    filas = w1_fixture_score.construir()
    assert any(f["tipo"] == "score" and f["esperada"] != f["opciones"][max(range(len(f["probs"])), key=f["probs"].__getitem__)] for f in filas), "la mediana debe diferir del argmax en alguna"
    assert any(f["esperada"] is None for f in filas)
    # una cola exactamente en 0,5: [0,5; 0,25; 0,25] no sube del nivel 1
    assert set(f["esperada"] for f in filas if f["probs"] == [0.5, 0.25, 0.25] and f["tipo"] == "score" and f["opciones"] == ["1", "2", "3"]) == {"1"}


def test_fixture_ejemplos_al_dia():
    w1_fixture_ejemplos.main(["--comprobar"])


def test_regla_top1_es_la_mediana_coral():
    r = {"tipo": "score", "opciones": ["1", "2", "3"]}
    assert lb.top1({"1": 0.4, "2": 0.2, "3": 0.4}, r) == "2"     # el argmax sería 1
    assert lb.top1({"1": 0.4, "2": 0.2, "3": 0.4}, r, esp="argmax") == "1"


@pytest.fixture()
def entradas(tmp_path):
    dev = tmp_path / "dev"
    dev_sintetico_qwen.crear(dev, n_test=40, por_clase=12)
    sal = tmp_path / "ent"
    w1_filas.preparar(str(dev), str(sal), 25)
    return dev, sal


def test_filas_soporte_8_por_clase_y_misma_muestra_para_los_dos(entradas):
    dev, sal = entradas
    sop = [json.loads(l) for l in open(sal / "soporte.jsonl", encoding="utf-8")]
    cons = [json.loads(l) for l in open(sal / "consultas.jsonl", encoding="utf-8")]
    tareas = json.load(open(sal / "tareas.json", encoding="utf-8"))
    assert len(tareas) == 6 and len(cons) == 6 * 25
    for t in tareas:
        ys = [r["etiqueta"] for r in sop if r["tarea"] == t]
        assert all(ys.count(y) <= 8 for y in set(ys))
        assert all({"id", "tipo", "opciones", "pregunta", "texto"} <= set(r) for r in sop if r["tarea"] == t)   # lo que pide chispa-banco-qwen para escribir los ejemplos
        py_test = [json.loads(l)["id"] for l in open(sal / "test" / (t + ".jsonl"), encoding="utf-8")]
        assert py_test == [r["id"] for r in cons if r["tarea"] == t]
        py_sop = [json.loads(l)["id"] for l in open(sal / "soporte" / (t + ".jsonl"), encoding="utf-8")]
        assert py_sop == [r["id"] for r in sop if r["tarea"] == t]
        assert not set(py_sop) & set(py_test)


def test_muestra_determinista_por_hash():
    ev = [{"id": "x%d" % i} for i in range(100)]
    a = w1_filas.elegir_test(ev, 10)
    b = w1_filas.elegir_test(list(reversed(ev)), 10)
    assert [r["id"] for r in a] == [r["id"] for r in b] and len(a) == 10


def _escribir(p, filas):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in filas), encoding="utf-8")


def test_medir_lee_las_dos_con_top1_y_aplica_el_criterio(tmp_path):
    crit = json.load(open(CRIT, encoding="utf-8"))
    ev = [{"tarea": "t", "id": "r%d" % i, "tipo": "score", "opciones": ["1", "2", "3"], "etiqueta": "2"} for i in range(4)]
    probs = {"1": 0.4, "2": 0.2, "3": 0.4}   # mediana 2 (acierta), argmax 1 (falla)
    go = [{"tarea": "t", "id": r["id"], "http": 200, "probs": probs, "answer": "1"} for r in ev]
    _escribir(tmp_path / "py" / "t.jsonl", [{"id": r["id"], "probs": probs, "valida": True} for r in ev])
    sel = {"acuerdo": 1.0, "n": 60}
    out = w1_medir.cifras(ev, go, str(tmp_path / "py"), crit, sel)
    assert out["macro_go"] == out["macro_python"] == 1.0 and out["dif"] == 0 and out["pasa_todo"]
    # sin selección medida, o con acuerdo < 0,97, no pasa aunque la exactitud coincida
    assert not w1_medir.cifras(ev, go, str(tmp_path / "py"), crit)["pasa_todo"]
    assert not w1_medir.cifras(ev, go, str(tmp_path / "py"), crit, {"acuerdo": 0.95, "n": 60})["condiciones"]["seleccion_demos"]["pasa"]
    assert out["informativo"]["macro_answer_del_servidor"] == 0.0   # la answer tal cual (argmax) no cuenta
    # un fallo de cobertura del servidor lo hace no pasar
    go[0]["http"] = 503
    out = w1_medir.cifras(ev, go, str(tmp_path / "py"), crit, sel)
    assert not out["condiciones"]["cobertura"]["pasa"] and not out["pasa_todo"]
    # una diferencia de macro > 0,005 también
    go[0]["http"] = 200
    go[1]["probs"] = {"1": 0.9, "2": 0.05, "3": 0.05}
    out = w1_medir.cifras(ev, go, str(tmp_path / "py"), crit, sel)
    assert out["dif"] == -0.25 and not out["condiciones"]["dif_macro"]["pasa"]


def test_criterio_propuesto_no_esta_congelado():
    crit = json.load(open(CRIT, encoding="utf-8"))
    assert crit["congelado_antes_de_medir"] is False and crit["criterio"]["tolerancia_macro"] == 0.005 and crit["criterio"]["cobertura_minima"] == 1.0
