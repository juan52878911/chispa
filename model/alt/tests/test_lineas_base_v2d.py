"""R10-D: demostraciones por kNN, mezcla de posteriores con w elegido en la mitad A, medida en B y el user-data. Sin red."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb
from model.alt import lineas_base_v2_pred as pr
from model.alt import lineas_base_v2d as vd
from model.alt.auditoria import mitad
from model.alt.tests.test_lineas_base_v2b import dev_falso, escribir_pred

AQUI = Path(__file__).resolve().parents[1]
CRIT = json.loads((AQUI / "crit-r10-d.json").read_text(encoding="utf-8"))
TODAS = set(CRIT["tareas_dev"])
FIN = "finanzas-documentos"


def test_demos_fila_cubre_clases_si_caben_y_ordena_por_cercania():
    ys = ["a", "b", "c"] * 6
    sim = np.array([0.9, 0.8, 0.7] * 3 + [0.1, 0.2, 0.3] * 3)   # los 9 primeros son los cercanos
    sim[2] = 0.01   # la clase c tiene a su más cercano lejos
    d = pr.demos_fila(sim, ys, 4)
    assert len(d) == 4 and {ys[i] for i in d} == {"a", "b", "c"}                 # una de cada clase aunque la c esté lejos
    assert [sim[i] for i in d] == sorted(sim[i] for i in d)                      # del más lejano al más cercano
    muchas = [str(i % 12) for i in range(48)]
    s2 = np.linspace(0, 1, 48)
    d2 = pr.demos_fila(s2, muchas, 8)
    assert d2 == list(range(40, 48))                                              # 12 clases no caben en 8: los 8 más cercanos


def test_cmd_demos_escribe_el_formato_de_llm_logprobs(tmp_path):
    rng = np.random.RandomState(1)
    ops = ["x", "y"]
    mk = lambda i, y, sp: {"id": "t-%s%d" % (sp[:2], i), "tarea": "t", "tipo": "choice", "pregunta": "?", "texto": "t", "opciones": ops, "descripciones": {}, "etiqueta": y, "split": sp}  # noqa: E731
    rows = [mk(i, ops[i % 2], "train") for i in range(10)] + [mk(i, ops[i % 2], "test") for i in range(6)]
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    v = rng.randn(16, 8)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    (tmp_path / "vec").mkdir()
    np.savez(tmp_path / "vec" / "t.npz", Xtr=v[:10], Xte=v[10:], idtr=np.array([r["id"] for r in rows[:10]]), idte=np.array([r["id"] for r in rows[10:]]), transformers=np.array("5.22.2"))
    pr.main(["demos", "--dev", str(dev), "--vec", str(tmp_path / "vec"), "--salida", str(tmp_path / "d"), "--k", "4"])
    d = [json.loads(l) for l in (tmp_path / "d/t.jsonl").read_text().splitlines()]
    train = {r["id"]: r for r in rows[:10]}
    assert [x["id"] for x in d] == [r["id"] for r in rows[10:]]
    assert all(len(x["ejemplos"]) == 4 and all(e in train for e in x["ejemplos"]) and {train[e]["etiqueta"] for e in x["ejemplos"]} == {"x", "y"} for x in d)   # las 2 clases caben en 4


def test_mezclar():
    r = {"opciones": ["a", "b"]}
    assert vd.mezclar({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, r, 0.25) == {"a": 0.25, "b": 0.75}
    assert vd.mezclar({"a": 1.0}, None, r, 0.5) is None and vd.mezclar({"a": 1.0, "b": 0.0}, None, r, 1.0) == {"a": 1.0, "b": 0.0}
    assert vd.mezclar({"a": 0.5, "b": 0.5}, {"a": 1.0}, r, 0.0) == {"a": 1.0, "b": 0.0}   # con las 10 candidatas de finanzas: lo que falta vale 0


def montar(tmp_path, acc_knn=0.9, acc_demos=0.7, cubrir=True):
    dev = dev_falso(tmp_path)
    t3, pred = tmp_path / "t3", tmp_path / "pred"
    escribir_pred(dev, t3, "4b-zs", TODAS, 0.4)
    escribir_pred(dev, t3, "4b-fs8", TODAS, 0.5)
    escribir_pred(dev, pred, "zs-topk", {FIN}, 0.4)
    escribir_pred(dev, pred, "fs8-topk", {FIN}, 0.5)
    escribir_pred(dev, pred, "fs8-knn", TODAS if cubrir else {FIN}, acc_demos)
    escribir_pred(dev, pred, "knn-e5-k8", TODAS if cubrir else {FIN}, acc_knn)   # sin cobertura del kNN las mezclas no cubren todo
    return dev, t3, pred


def test_medir_elige_w_en_A_mide_en_B_y_decide(tmp_path):
    dev, t3, pred = montar(tmp_path)
    out = tmp_path / "c.json"
    vd.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["criterio_sha256"] == hashlib.sha256((AQUI / "crit-r10-d.json").read_bytes()).hexdigest()
    assert r["n_A"] + r["n_B"] == 240 and r["n_A"] > 60 and r["n_B"] > 60
    assert r["A"]["mejor_variante_demos"] == "fs8-knn" and r["sistema_final"] == "fs8-knn+knn"
    assert set(r["A"]["w"]) == {"zs", "fs8-azar", "fs8-knn"} and all(w in CRIT["mezcla"]["w_grid"] for w in r["A"]["w"].values())
    assert r["sistemas_B"]["fs8-knn+knn"]["w"] == r["A"]["w"]["fs8-knn"] and r["sistemas_B"]["fs8-knn"]["w"] is None
    v = r["veredicto"]
    assert v["veredicto"] == "PASA" and v["macro_dif"] >= 0.03 and v["peor_tarea"] >= -0.02 and v["cobertura"] == 1.0
    ms = v["score_aparte"]["final"]
    assert ms["n_tareas"] == 3 and {"exactitud", "qwk", "mae", "exactitud_sin_sinteticas"} <= set(ms) and "macro_sin_sinteticas" in r["sistemas_B"]["knn"]
    # B es solo B: n_test de cada tarea = filas con mitad 1
    esperado = sum(1 for _, ev in lb.cargar_dev(dev).values() for x in ev if mitad(x["id"]) == 1)
    assert sum(t["n_test"] for t in r["sistemas_B"]["fs8-azar"]["tareas"].values()) == esperado
    assert "texto" not in out.read_text(encoding="utf-8")


def test_no_pasa_si_el_final_no_gana_y_no_medido_sin_cobertura(tmp_path):
    dev, t3, pred = montar(tmp_path, acc_knn=0.1, acc_demos=0.3)
    out = tmp_path / "c.json"
    vd.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["A"]["mejor_variante_demos"] == "fs8-azar" and r["veredicto"]["veredicto"] == "NO PASA"   # las demostraciones por kNN no ganan: la mezcla va sobre fs8 al azar
    dev2, t32, pred2 = montar(tmp_path / "b", cubrir=False) if (tmp_path / "b").mkdir() is None else (None, None, None)
    out2 = tmp_path / "c2.json"
    vd.main(["medir", "--dev", str(dev2), "--pred", str(pred2), "--t3", str(t32), "--salida", str(out2)])
    assert json.loads(out2.read_text())["veredicto"]["veredicto"] == "no medido"


def test_criterio_congelado_dice_lo_pedido():
    assert CRIT["pasa"]["macro_min"] == 0.03 and CRIT["pasa"]["tarea_max_perdida"] == 0.02 and CRIT["referencia"] == "fs8-azar"
    assert CRIT["mezcla"]["w_grid"] == [round(0.1 * i, 1) for i in range(11)] and CRIT["demos"]["k"] == 8 and CRIT["topk"]["k"] == 10
    assert "candidato_final" in CRIT and "mitad A" in CRIT["candidato_final"] and "auditoria.mitad" in CRIT["particion"] and "selladas" in CRIT["datos"]
    assert CRIT["sinteticas"] == ["finanzas-documentos", "horeca-resenas"] and len(CRIT["entradas_t3_sha256"]) == 11


def test_ud_d():
    ud = (AQUI / "aws" / "r10_d.sh").read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(AQUI / "aws" / "r10_d.sh")]).returncode == 0
    assert set(re.findall(r"__[A-Z0-9_]+__", ud)) == {"__CHISPA_S3__", "__CRIT_SHA__", "__N_TEST__", "__N_SOP__"}
    sust = ud.replace("__CHISPA_S3__", "s3://chispa-pruebas-879381269751/x-" + "0" * 40).replace("__CRIT_SHA__", "0" * 64).replace("__N_TEST__", "2930").replace("__N_SOP__", "728")
    assert len(sust.encode()) <= 16384
    for s in ("MIN=${CHISPA_MIN:-110}", "dev.tgz", "t3.tgz", "sellad", "CRIT_SHA", "shutdown -h +$MIN", "--max-logprobs 20", "cdbee75f", "crit-r10-d.json", "--demos", "TORCH=2.7.1", "TRANSFORMERS=5.22.2",
              "lineas_base_v2d medir", "$P demos", "$P knn"):
        assert s in ud, s
    assert "14b" not in ud.lower() and "s3 ls" not in ud and ud.index("lp zs-topk") < ud.index("lp fs8-topk") < ud.index("lp fs8-knn")
