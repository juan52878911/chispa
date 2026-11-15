"""Pruebas de model.alt.prequential sin red ni torch: batería y fondo sintéticos en 2-D."""
import json

import numpy as np

from model.alt import fusion_knn_l2 as fk
from model.alt import prequential as pq


def _vec(y, rng, ruido=0.3):
    v = np.array([1.0 if y == "a" else -1.0, rng.randn() * ruido])
    return v / np.linalg.norm(v)


def _escenario(tmp_path, n_test=60, falla_llm=0):
    """hatecheck-es (oro 2 por clase), fakenews-es (sin oro) y fondo solo de hatecheck-es. El LLM acierta salvo `falla_llm` filas."""
    bat, vec = tmp_path / "bat", tmp_path / "vec"
    bat.mkdir()
    vec.mkdir()
    rng = np.random.RandomState(0)
    l3, fondo, l3f, Xf = [], [], [], []
    for c, n_oro in (("hatecheck-es", 4), ("fakenews-es", 0)):
        filas, Xtr, Xte = [], [], []
        for split, n, X in (("train", n_oro, Xtr), ("test", n_test, Xte)):
            for i in range(n):
                y = "a" if i % 2 else "b"
                X.append(_vec(y, rng))
                filas.append({"id": "%s:%s:%d" % (c, split, i), "split": split, "etiqueta": y, "opciones": ["a", "b"], "texto": "t"})
                if split == "test":
                    pred = y if i >= falla_llm else ("b" if y == "a" else "a")
                    l3.append({"id": filas[-1]["id"], "valida": True, "probs": {pred: 0.9, ("b" if pred == "a" else "a"): 0.1}})
        (bat / (c + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in filas))
        np.savez(vec / (c + ".npz"), Xtr=np.array(Xtr).reshape(len(Xtr), 2), Xte=np.array(Xte),
                 idtr=np.array([r["id"] for r in filas if r["split"] == "train"], dtype=str),
                 idte=np.array([r["id"] for r in filas if r["split"] == "test"]))
    for i in range(30):
        y = "a" if i % 2 else "b"
        Xf.append(_vec(y, rng))
        fid = "fondo:%d" % i
        fondo.append({"id": fid, "conjunto": "hatecheck-es", "opciones": ["a", "b"], "tipo": "noul", "texto": "t"})
        l3f.append({"id": fid, "valida": i != 3, "probs": {y: 0.95, ("b" if y == "a" else "a"): 0.05}})
    np.savez(vec / "fondo.npz", X=np.array(Xf), ids=np.array([r["id"] for r in fondo]))
    for nombre, filas in (("fondo.jsonl", fondo), ("l3.jsonl", l3), ("l3_fondo.jsonl", l3f)):
        (tmp_path / nombre).write_text("".join(json.dumps(r) + "\n" for r in filas))
    return bat, vec


def _tareas(tmp_path, **kw):
    bat, vec = _escenario(tmp_path, **kw)
    z = np.load(vec / "fondo.npz")
    return pq.cargar_tareas([str(bat)], vec, [str(tmp_path / "l3.jsonl")], pq.leer(tmp_path / "fondo.jsonl"),
                            pq.leer(tmp_path / "l3_fondo.jsonl"), z)


def test_estatico_igual_que_fusion_knn(tmp_path):
    t = _tareas(tmp_path)["hatecheck-es"]
    assert (t.G, t.F, t.n) == (4, 29, 60)           # una fila del fondo no es válida
    ok = t.correr(np.arange(t.n), True, False, "igual", 50)
    Xs = t.S.shape  # el pozo se reconstruye con los vectores para comparar con el kNN del paso 1
    assert Xs == (60, 4 + 29 + 60)
    bat = tmp_path / "vec" / "hatecheck-es.npz"
    z = np.load(bat)
    zf = np.load(tmp_path / "vec" / "fondo.npz")
    fondo = pq.leer(tmp_path / "fondo.jsonl")
    keep = [i for i, f in enumerate(fondo) if i != 3]
    X = np.vstack([z["Xtr"], zf["X"][keep]])
    ys = ["b", "a", "b", "a"] + [("a" if i % 2 else "b") for i in keep]
    ops = [["a", "b"]] * 60
    P, _, _ = fk.knn(z["Xte"], X, ys, ops)
    ref = (np.argmax(P, 1) == t.y).astype(float)
    assert np.array_equal(ok, ref)


def test_retraso_se_respeta(tmp_path):
    """Con retraso N la etiqueta de la fila en el paso p entra tras el paso p+N: el banco del paso t tiene max(0, t−N) filas del flujo."""
    t = _tareas(tmp_path)["fakenews-es"]
    visto = []
    original = pq.votar

    def espia(s, y, w, n_oro, lut, n_ops, regla):
        visto.append(len(s))
        return original(s, y, w, n_oro, lut, n_ops, regla)
    pq.votar = espia
    try:
        orden = np.random.RandomState(1).permutation(t.n)
        t.correr(orden, False, True, "peso", 7)
    finally:
        pq.votar = original
    assert visto[:8] == [0] * 8                     # pasos 0..7: nada (la fila 0 entra tras el paso 7)
    assert visto[8] == 1 and visto[20] == 13        # el paso p tiene p−7 elementos (todas válidas)
    assert visto[-1] == t.n - 1 - 7


def test_retraso_no_usa_la_etiqueta_antes(tmp_path):
    """Una fila marcada con una etiqueta absurda solo influye desde el paso r+N+1."""
    t = _tareas(tmp_path)["fakenews-es"]
    orden = np.arange(t.n)
    N = 5
    r = 10
    # etiqueta de la fila r envenenada: ocupa el código de «a» en una fila de «b»; sin banco lo demás no influye
    t.ypozo[t.G + t.F + r] = t.cod["a"]
    t.S[:] = 0.0
    t.S[:, t.G + t.F + r] = 1.0                      # todas las filas son vecinas únicas (sim 1) del elemento r; el resto sim 0
    ok = t.correr(orden, False, True, "peso", N)
    # filas ≤ r+N: banco sin r. El único voto posible desde r+N+1 es «a» (los demás tienen sim 0 pero votan igual)
    t.S[:, t.G + t.F + r] = 5.0                       # r domina por la temperatura
    ok2 = t.correr(orden, False, True, "peso", N)
    assert np.array_equal(ok[:r + N + 1], ok2[:r + N + 1])


def test_oro_manda_con_peso():
    """Con un vecino de oro y uno del LLM igual de cercanos y etiquetas distintas, gana el oro (peso 1 frente a ALFA)."""
    s = np.array([0.8, 0.8])
    y = np.array([0, 1])
    lut = np.array([0, 1])
    w = np.array([1.0, pq.ALFA])
    P = pq.votar(s, y, w, 1, lut, 2, "peso")
    assert P[0] > P[1] and abs(P[0] - 1 / (1 + pq.ALFA)) < 1e-9
    P = pq.votar(s, y, np.ones(2), 1, lut, 2, "igual")
    assert abs(P[0] - 0.5) < 1e-9


def test_regla_umbral():
    lut = np.array([0, 1])
    y = np.array([0, 1, 1, 1])
    w = np.ones(4)
    cerca = np.array([pq.S_ORO + 0.05, 0.99, 0.99, 0.99])   # el LLM está más cerca, pero el oro supera s*: solo oro
    assert pq.votar(cerca, y, w, 1, lut, 2, "umbral")[0] == 1.0
    lejos = np.array([pq.S_ORO - 0.2, 0.99, 0.99, 0.99])   # oro lejos: vota todo el banco
    assert pq.votar(lejos, y, w, 1, lut, 2, "umbral")[1] > 0.99


def test_banco_vacio_sin_oro(tmp_path):
    t = _tareas(tmp_path)["fakenews-es"]
    assert t.G == 0 and t.F == 0
    ok = t.correr(np.arange(t.n), False, False, "igual", 50)   # banco vacío: uniforme, siempre la primera opción
    assert np.array_equal(ok, (t.y == 0).astype(float))
    assert abs(ok.mean() - 0.5) < 0.1
    P = pq.votar(np.zeros(0), np.zeros(0, dtype=int), np.zeros(0), 0, np.array([0, 1]), 2, "peso")
    assert np.allclose(P, 0.5)


def test_medir_de_punta_a_punta_y_criterio(tmp_path):
    bat, vec = _escenario(tmp_path)
    sal = tmp_path / "inf.json"
    pq.main(["medir", "--bateria", str(bat), "--vec", str(vec), "--l3", str(tmp_path / "l3.jsonl"), "--fondo", str(tmp_path / "fondo.jsonl"),
             "--l3-fondo", str(tmp_path / "l3_fondo.jsonl"), "--vec-fondo", str(vec / "fondo.npz"), "--salida", str(sal), "--semillas", "3",
             "--retraso", "10"])
    out = json.loads(sal.read_text())
    assert out["criterio_sha256"] == pq.CRIT_R6_SHA
    assert out["tareas"]["fakenews-es"]["oro"] == 0 and out["tareas"]["hatecheck-es"]["fondo_llm"] == 29
    V = out["variantes"]
    assert set(V) == set(pq.VARIANTES)
    assert V["oro_estatico"]["por_tarea"]["hatecheck-es"]["exactitud"][0] > 0.9          # problema separable
    assert V["preq_oro"]["por_tarea"]["fakenews-es"]["exactitud"][0] > V["oro_estatico"]["por_tarea"]["fakenews-es"]["exactitud"][0] + 0.2
    assert len(V["preq_oro_fondo"]["curva"]["todas"]) == pq.TRAMOS
    assert out["veredicto"]["estado"] in ("PASA", "NO PASA", "REFUTADO")
    assert out["veredicto"]["parametros_coinciden"] is False                                # retraso 10 != 50
    d = out["diferencias"]["preq_oro_fondo"]["frente_a_oro_estatico"]
    assert len(d["nuevas"]) == 3 and d["nuevas"][1] <= d["nuevas"][0] <= d["nuevas"][2]
    assert "texto" not in json.dumps(out)


def test_criterio_congelado(capsys):
    assert pq.CRIT_R6 == {"variante": "preq_oro_fondo", "ref_nuevas": "oro_estatico", "ref_originales": "oro_llm_estatico",
                          "nuevas": ["hatecheck-es", "lexboe", "fakenews-es"], "sube_min": 0.05, "perdida_max_original": 0.02,
                          "refutado_si_dif_le": 0.0, "alfa": 0.5, "retraso": 50, "vecinos": 32, "temp": 0.05, "regla": "peso"}
    pq.main(["criterio"])
    sha, js = capsys.readouterr().out.split(" ", 1)
    assert sha == pq.CRIT_R6_SHA and json.loads(js) == pq.CRIT_R6
