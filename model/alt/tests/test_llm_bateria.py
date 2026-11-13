"""Pruebas de model.alt.llm_bateria sin red ni torch."""
import json
from collections import Counter

from model.alt import llm_bateria as lb


def filas(n_a, n_b, c="t"):
    return [{"id": "%s%d" % (e, i), "conjunto": c, "etiqueta": e} for e, n in (("a", n_a), ("b", n_b)) for i in range(n)]


def test_estratificada_mantiene_proporcion_y_es_fija():
    f = filas(75, 25)
    m = lb.estratificada(f, 20, "s")
    assert Counter(r["etiqueta"] for r in m) == {"a": 15, "b": 5}
    assert [r["id"] for r in m] == [r["id"] for r in lb.estratificada(f, 20, "s")]
    assert len(lb.estratificada(f, 500, "s")) == 100


def test_estratificada_anidada_es_subconjunto():
    m = lb.estratificada(filas(60, 40), 30, "x")
    s = lb.estratificada(m, 10, "y")
    assert {r["id"] for r in s} <= {r["id"] for r in m}


def test_intercalar_reparte_tareas():
    f = filas(3, 0, "x") + filas(1, 0, "y")
    assert [r["conjunto"] for r in lb.intercalar(f)] == ["x", "y", "x", "x"]


def test_boot_ponderada_y_por_pares():
    ok = {"x": [1.0] * 10, "y": [0.0] * 10}
    assert lb.boot(ok, {"x": 3, "y": 1})[0] == 0.75
    d = lb.boot(ok, {"x": 1, "y": 1}, ref={"x": [0.0] * 10, "y": [0.0] * 10})
    assert d == [0.5, 0.5, 0.5]


def test_llm_pred_sin_respuesta_es_invalida(tmp_path):
    p = tmp_path / "l3.jsonl"
    p.write_text(json.dumps({"id": "1", "probs": {"a": 0.8, "b": 0.2}, "valida": True, "raw": ""}) + "\n" +
                 json.dumps({"id": "2", "probs": None, "valida": False, "raw": '{"probs": {"a": 0.3, "b": 0.9}}'}) + "\n")
    f = [{"id": i, "opciones": ["a", "b"], "tipo": "choice"} for i in ("1", "2", "3")]
    est = lb.llm_pred(p, f)
    assert est["1"] == ("a", 0.8, True) and est["2"] == (None, 0.0, False) and est["3"] == (None, 0.0, False)
    assert lb.llm_pred(p, f, permisivo=True)["2"][0] == "b"


def test_veredicto():
    out = {"agrupada_ponderada": {"estricta": [0.70, 0.68, 0.72], "fusion": 0.61},
           "por_tarea": {"hatecheck-es": {"estricta": [0.80, 0, 0], "knn": 0.69, "knn_recalculado": 0.70}},
           "q4": {"q4_menos_bf16": [-0.01, -0.03, 0.01]}}
    v = lb.veredicto(out)
    assert v["estado"] == "PASA" and v["hatecheck_llm_menos_knn"] == 0.1
    out["agrupada_ponderada"]["estricta"] = [0.60, 0, 0]
    assert lb.veredicto(out)["estado"] == "REFUTADO"
    out["agrupada_ponderada"]["estricta"] = [0.63, 0, 0]
    assert lb.veredicto(out)["estado"] == "NO PASA"
