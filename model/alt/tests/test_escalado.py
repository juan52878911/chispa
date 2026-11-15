"""Pruebas de model.alt.escalado sin red ni torch, con la batería sintética de test_knn_soporte_llm."""
import json

import numpy as np

from model.alt import escalado as es
from model.alt.tests.test_knn_soporte_llm import _bateria


def test_medir_de_punta_a_punta(tmp_path):
    bat, vec = _bateria(tmp_path)
    for f in vec.glob("*.npz"):   # la versión de transformers va en cada .npz
        z = dict(np.load(f))
        z["transformers"] = np.array("x")
        np.savez(f, **z)
    filas = [json.loads(l) for c in ("hatecheck-es", "tuits-es") for l in open(bat / (c + ".jsonl"))]
    l3 = tmp_path / "l3t.jsonl"   # L3 acierta siempre
    l3.write_text("".join(json.dumps({"id": r["id"], "valida": True, "probs": {r["etiqueta"]: 1.0, ("b" if r["etiqueta"] == "a" else "a"): 0.0}}) + "\n"
                          for r in filas if r["split"] == "test"))
    sal = tmp_path / "e.json"
    es.main(["medir", "--bateria", str(bat), "--vec", str(vec), "--fondo", str(tmp_path / "fondo.jsonl"), "--l3-fondo", str(tmp_path / "l3.jsonl"),
             "--l3", str(l3), "--salida", str(sal), "--semillas", "4"])
    out = json.loads(sal.read_text())
    v = out["veredicto"]
    assert v["criterio_sha256"] == es.CRIT8E_SHA and v["senal"] in es.SENALES and v["fraccion"] in es.FRACS
    assert v["l3"] == 1.0 and v["cascada"] >= v["l2"] - 1e-9 and 0.0 <= v["escalado"] <= 0.35
    assert len(out["validacion"]) == 2 and out["tareas"]["tuits-es"]["n"] == 40


def test_regla_umbral_vota_solo_con_oro():
    Xte = np.array([[1.0, 0.0]])
    P, sen = es.l2(Xte, [["a", "b"]], np.array([[1.0, 0.0]]), ["a"], np.array([[1.0, 0.0]] * 5), ["b"] * 5)
    assert P[0, 0] == 1.0 and sen["sim_oro"][0] == 1.0     # el oro está a 1,0 ≥ 0,90: los 5 del LLM no votan
