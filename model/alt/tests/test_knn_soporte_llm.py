"""Pruebas de model.alt.knn_soporte_llm sin red ni torch: batería y fondo sintéticos en 2-D."""
import json

import numpy as np

from model.alt import knn_soporte_llm as ks


def _bateria(tmp_path):
    """Dos tareas binarias: la clase es el signo de la primera coordenada."""
    bat, vec = tmp_path / "bat", tmp_path / "vec"
    bat.mkdir()
    vec.mkdir()
    rng = np.random.RandomState(0)
    fondo, l3, fuente, Xf = [], [], [], []
    for c in ("hatecheck-es", "tuits-es"):
        filas, Xtr, Xte = [], [], []
        for split, n, X in (("train", 4, Xtr), ("test", 40, Xte)):
            for i in range(n):
                y = "a" if i % 2 else "b"
                v = np.array([1.0 if y == "a" else -1.0, rng.randn() * 0.3])
                X.append(v / np.linalg.norm(v))
                filas.append({"id": "%s:%s:%d" % (c, split, i), "split": split, "etiqueta": y, "opciones": ["a", "b"], "texto": "t"})
        (bat / (c + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in filas))
        np.savez(vec / (c + ".npz"), Xtr=np.array(Xtr), Xte=np.array(Xte),
                 idtr=np.array([r["id"] for r in filas if r["split"] == "train"]), idte=np.array([r["id"] for r in filas if r["split"] == "test"]))
        for i in range(30):
            y = "a" if i % 2 else "b"
            v = np.array([1.0 if y == "a" else -1.0, rng.randn() * 0.3])
            Xf.append(v / np.linalg.norm(v))
            fid = "fondo:%s:%d" % (c, i)
            fondo.append({"id": fid, "conjunto": c, "opciones": ["a", "b"], "tipo": "noul", "texto": "t"})
            pred = y if i % 10 else ("b" if y == "a" else "a")          # el LLM se equivoca en 1 de cada 10
            l3.append({"id": fid, "valida": i != 7, "probs": {pred: 0.95, ("b" if pred == "a" else "a"): 0.05}})
            fuente.append({"id": fid, "conjunto": c, "etiqueta": y})
    np.savez(vec / "fondo.npz", X=np.array(Xf), ids=np.array([r["id"] for r in fondo]))
    for nombre, filas in (("fondo.jsonl", fondo), ("l3.jsonl", l3), ("fuente.jsonl", fuente)):
        (tmp_path / nombre).write_text("".join(json.dumps(r) + "\n" for r in filas))
    return bat, vec


def test_soportes_quita_invalidas_y_mide_acuerdo(tmp_path):
    _bateria(tmp_path)
    sop = ks.soportes(ks.leer(tmp_path / "fondo.jsonl"), ks.leer(tmp_path / "l3.jsonl"), ks.leer(tmp_path / "fuente.jsonl"))
    assert len(sop["hatecheck-es"]["llm"]) == 29 and len(sop["hatecheck-es"]["fuente"]) == 30
    assert ks.acuerdo(sop["hatecheck-es"]) == round(26 / 29, 4)


def test_medir_de_punta_a_punta(tmp_path):
    bat, vec = _bateria(tmp_path)
    sal = tmp_path / "inf.json"
    ks.main(["medir", "--bateria", str(bat), "--vec", str(vec), "--fondo", str(tmp_path / "fondo.jsonl"), "--l3", str(tmp_path / "l3.jsonl"),
             "--fuente", str(tmp_path / "fuente.jsonl"), "--vec-oro", str(vec), "--salida", str(sal), "--semillas", "2"])
    out = json.loads(sal.read_text())
    assert out["criterio_sha256"] == ks.CRIT_SHA
    assert set(out["curvas"]) == {"oro", "llm", "llm_p09", "oro+llm", "fuente"}
    assert out["curvas"]["oro"]["todo"]["agrupada"][0] > 0.9          # problema separable
    assert out["vectores_frente_a_paso1"]["tuits-es"] == 0.0
    assert out["veredicto"]["estado"] in ("PASA", "NO PASA", "REFUTADO")
    assert out["curvas"]["llm"]["1"]["soporte_medio_por_clase"]["tuits-es"] == 1.0


def test_criterio_congelado():
    assert ks.CRIT == {"variante": "llm/todo", "referencia": "oro/todo", "perdida_max": 0.03, "agrupada_min": 0.55, "hatecheck_min": 0.60,
                       "refutado_si_agrupada_le": 0.516}


def test_criterio_r4_y_varios_fondos(tmp_path):
    bat, vec = _bateria(tmp_path)
    f = str(tmp_path / "fondo.jsonl")
    sal = tmp_path / "inf.json"
    import numpy as np
    z = np.load(vec / "fondo.npz")
    np.savez(vec / "fondo.npz", X=np.vstack([z["X"], z["X"]]), ids=np.concatenate([z["ids"], z["ids"]]))   # el mismo fondo dos veces
    l3 = str(tmp_path / "l3.jsonl")
    ks.main(["medir", "--bateria", str(bat), "--vec", str(vec), "--fondo", f + "," + f, "--l3", l3 + "," + l3, "--salida", str(sal), "--semillas", "1", "--criterio", "r4"])
    out = json.loads(sal.read_text())
    assert out["criterio_r4_sha256"] == ks.CRIT4_SHA and out["veredicto_r4"]["estado"] in ("PASA", "NO PASA", "REFUTADO")
    assert out["soporte"]["tuits-es"]["en_fondo"] == 60


def test_r6b_etiquetas_alternativas(tmp_path):
    bat, vec = _bateria(tmp_path)
    alt = tmp_path / "alt.jsonl"   # las alternativas corrigen todos los errores del LLM
    fu = [json.loads(l) for l in open(tmp_path / "fuente.jsonl")]
    alt.write_text("".join(json.dumps({"id": r["id"], "valida": True, "probs": {r["etiqueta"]: 1.0, ("b" if r["etiqueta"] == "a" else "a"): 0.0}}) + "\n" for r in fu))
    sal = tmp_path / "inf.json"
    ks.main(["medir", "--bateria", str(bat), "--vec", str(vec), "--fondo", str(tmp_path / "fondo.jsonl"), "--l3", str(tmp_path / "l3.jsonl"),
             "--l3-alt", str(alt), "--salida", str(sal), "--semillas", "1", "--criterio", "r6b"])
    out = json.loads(sal.read_text())
    assert out["criterio_r6b_sha256"] == ks.CRIT6B_SHA and "llm_alt" in out["curvas"]
    assert out["reetiquetados_por_tarea"]["tuits-es"] == 30 and out["veredicto_r6b"]["estado"] in ("PASA", "NO PASA", "REFUTADO")
