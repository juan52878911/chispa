"""R10-T3a/T3b: variantes del 4B (medida compuesta, esperanza ordinal, rúbrica, preselección top-k) y los dos user-data. Sin red."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb
from model.alt import lineas_base_v2_pred as pr
from model.alt import lineas_base_v2b as vb

AQUI = Path(__file__).resolve().parents[1]
CRIT = json.loads((AQUI / "crit-r10-t3b.json").read_text(encoding="utf-8"))
SCORE = ["horeca-resenas", "jql-valor-educativo", "usajobs-nivel"]


def tarea(nombre, tipo, n=40):
    ops = [str(i + 1) for i in range(5)] if tipo == "score" else ["a", "b", "c"]
    mk = lambda i, sp: {"id": "%s-%s%d" % (nombre, sp[0], i), "tarea": nombre, "tipo": tipo, "pregunta": "?", "texto": "t", "opciones": ops,  # noqa: E731
                        "descripciones": {o: "d" + o for o in ops}, "etiqueta": ops[i % len(ops)], "split": sp}
    return [mk(i, "train") for i in range(10)] + [mk(i, "test") for i in range(n)]


def dev_falso(tmp_path):
    d = tmp_path / "dev"
    d.mkdir()
    for t in CRIT["tareas_dev"]:
        (d / (t + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in tarea(t, "score" if t in SCORE else "choice")), encoding="utf-8")
    return d


def escribir_pred(dev, dest, sistema, tareas, acierto, sesgo_alto=False):
    for t, (_, ev) in lb.cargar_dev(dev).items():
        if t not in tareas:
            continue
        filas = []
        for i, r in enumerate(ev):
            ops = lb.orden(r)
            gold = ops.index(r["etiqueta"])
            idx = gold if (i % 10) < acierto * 10 else (gold + 1) % len(ops)
            probs = {o: (0.9 if j == idx else 0.1 / (len(ops) - 1)) for j, o in enumerate(ops)}
            if sesgo_alto and r["tipo"] == "score":   # masa repartida de forma bimodal: la esperanza cae en el centro, la mediana no
                probs = {o: (0.45 if o in ("1", "5") else 0.0333) for o in ops}
            filas.append({"id": r["id"], "probs": probs, "valida": True, "ms": 5.0})
        (Path(dest) / sistema).mkdir(parents=True, exist_ok=True)
        (Path(dest) / sistema / (t + ".jsonl")).write_text("".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")


def test_esperanza_ordinal_frente_a_mediana():
    r = {"tipo": "score", "opciones": ["1", "2", "3", "4", "5"]}
    p = {"1": 0.3, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.7}
    assert lb.top1(p, r) == "5"                  # mediana CORAL: las 4 colas valen 0,7 > 0,5
    assert lb.top1(p, r, esp=True) == "4"        # esperanza = 3,8 -> nivel 4
    assert lb.top1({"1": 0.0, "2": 0.5, "3": 0.5, "4": 0.0, "5": 0.0}, r, esp=True) == "2"   # empate exacto: el menor
    assert lb.top1({"a": 1.0}, {"tipo": "choice", "opciones": ["a", "b"]}, esp=True) == "a"   # en choice no cambia nada
    assert lb.top1(None, r, esp=True) is None


def test_rubrica_en_descripciones_solo_score(tmp_path):
    dev = dev_falso(tmp_path)
    rub = {"tareas": {t: {"niveles": {str(i): {"descripcion": "desc%d." % i, "ejemplo": "ej%d" % i} for i in range(1, 6)}} for t in SCORE}}
    (tmp_path / "r.json").write_text(json.dumps(rub), encoding="utf-8")
    vb.main(["rubrica", "--dev", str(dev), "--rubricas", str(tmp_path / "r.json"), "--salida", str(tmp_path / "t")])
    assert sorted(p.stem for p in (tmp_path / "t/rub/test").glob("*.jsonl")) == sorted(SCORE)
    f = json.loads((tmp_path / "t/rub/test/horeca-resenas.jsonl").read_text().splitlines()[0])
    assert f["descripciones"]["3"] == "desc3. Ejemplo: «ej3»" and f["etiqueta"] and f["texto"] == "t"


def test_rubrica_real_cubre_los_niveles_del_soporte():
    rub = json.loads((AQUI / "rubricas-r10-t3b.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((AQUI / "rubricas-r10-t3b.json").read_bytes()).hexdigest() == CRIT["rubricas_sha256"]
    assert sorted(rub["tareas"]) == sorted(SCORE)
    assert {t: len(v["niveles"]) for t, v in rub["tareas"].items()} == {"horeca-resenas": 5, "jql-valor-educativo": 5, "usajobs-nivel": 11}
    assert all(n["descripcion"] and n["ejemplo"] and ":train:" in n["fuente_ejemplo"] for v in rub["tareas"].values() for n in v["niveles"].values())


def test_medir_compuesto_veredictos(tmp_path):
    dev = dev_falso(tmp_path)
    t3, pred = tmp_path / "t3", tmp_path / "pred"
    todas = set(CRIT["tareas_dev"])
    escribir_pred(dev, t3, "4b-zs", todas, 0.5)
    escribir_pred(dev, t3, "4b-fs8", todas, 0.6)
    escribir_pred(dev, pred, "zs-topk", {"finanzas-documentos"}, 0.5)
    escribir_pred(dev, pred, "fs8-topk", {"finanzas-documentos"}, 0.6)
    escribir_pred(dev, pred, "rub", set(SCORE), 0.9)         # la rúbrica sube las score de 50 a 90: macro +0,2
    escribir_pred(dev, pred, "fs8rub", set(SCORE), 0.5)
    out = tmp_path / "c.json"
    vb.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["criterio_sha256"] == hashlib.sha256((AQUI / "crit-r10-t3b.json").read_bytes()).hexdigest()
    v = r["veredictos"]
    assert v["rub"]["veredicto"] == "PASA" and v["rub"]["macro_dif"] > 0.15 and v["rub"]["peor_choice"] == 0.0
    assert all(v[c]["veredicto"] in ("PASA", "NO PASA") for c in CRIT["candidatos"])
    assert v["base+esp"]["veredicto"] == "NO PASA"   # la esperanza no cambia nada con predicciones nítidas
    assert r["sistemas"]["base"]["tareas"]["finanzas-documentos"]["sintetica"] and r["sistemas"]["base"]["macro_sin_sinteticas"]["n_tareas"] == 4
    assert r["sistemas"]["rub"]["macro_score"]["n_tareas"] == 3 and "mae" in r["sistemas"]["rub"]["macro_score"]
    assert set(r["pareadas_extra"]) == {"%s_menos_%s" % tuple(p) for p in CRIT["pares_extra"]}
    assert r["v4_topk"]["veredicto"] == "PASA" and r["v4_topk"]["filas_comunes"] == 40
    assert "texto" not in out.read_text(encoding="utf-8")


def test_medir_sin_cobertura_es_no_medido(tmp_path):
    dev = dev_falso(tmp_path)
    t3, pred = tmp_path / "t3", tmp_path / "pred"
    escribir_pred(dev, t3, "4b-zs", set(CRIT["tareas_dev"]), 0.5)
    escribir_pred(dev, t3, "4b-fs8", set(CRIT["tareas_dev"]), 0.5)
    escribir_pred(dev, pred, "zs-topk", {"finanzas-documentos"}, 0.5)
    escribir_pred(dev, pred, "rub", {"horeca-resenas"}, 0.9)   # faltan jql y usajobs
    out = tmp_path / "c.json"
    vb.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["veredictos"]["rub"]["veredicto"] == "no medido" and r["veredictos"]["fs8rub"]["veredicto"] == "no medido"
    assert r["sistemas"]["rub"]["tareas"]["jql-valor-educativo"]["cobertura"] == 0.0
    assert r["sistemas"]["rub"]["tareas"]["horeca-resenas"]["exactitud_cubiertas"] > 0.8


def test_topk_preselecciona_con_oro_forzado_en_soporte(tmp_path):
    rng = np.random.RandomState(0)
    cls = ["c%02d" % i for i in range(15)]
    centro = rng.randn(15, 16)
    mk = lambda n: [(i % 15, centro[i % 15] + 0.05 * rng.randn(16)) for i in range(n)]  # noqa: E731
    tr, te = mk(60), mk(30)
    norm = lambda v: v / np.linalg.norm(v)  # noqa: E731
    fila = lambda i, y, sp: {"id": "f-%s%d" % (sp[0], i), "tarea": "f", "tipo": "choice", "pregunta": "?", "texto": "t", "opciones": cls,  # noqa: E731
                             "descripciones": {c: "d" + c for c in cls}, "etiqueta": cls[y], "split": sp}
    dev = tmp_path / "dev"
    dev.mkdir()
    rows = [fila(i, y, "train") for i, (y, _) in enumerate(tr)] + [fila(i, y, "test") for i, (y, _) in enumerate(te)]
    (dev / "f.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    vec = tmp_path / "vec"
    vec.mkdir()
    np.savez(vec / "f.npz", Xtr=np.array([norm(x) for _, x in tr]), Xte=np.array([norm(x) for _, x in te]), idtr=np.array([r["id"] for r in rows[:60]]),
             idte=np.array([r["id"] for r in rows[60:]]), transformers=np.array("5.22.2"))
    pr.main(["topk", "--dev", str(dev), "--vec", str(vec), "--tarea", "f", "--salida", str(tmp_path / "t")])
    c = json.loads((tmp_path / "t/preseleccion.json").read_text())
    assert c["k"] == 10 and c["recall_at_k"] == 1.0 and c["knn_top1"] > 0.9 and c["transformers"] == "5.22.2"
    test = [json.loads(l) for l in (tmp_path / "t/test/f.jsonl").read_text().splitlines()]
    sop = [json.loads(l) for l in (tmp_path / "t/soporte/f.jsonl").read_text().splitlines()]
    assert len(test) == 30 and len(sop) == 60
    assert all(len(r["opciones"]) == 10 and r["opciones"] == sorted(r["opciones"]) and set(r["descripciones"]) == set(r["opciones"]) for r in test + sop)
    assert all(r["etiqueta"] in r["opciones"] for r in sop)       # oro forzado dentro de las 10 en el soporte
    assert [r["id"] for r in test] == [r["id"] for r in rows[60:]]


def ud_ok(nombre, placeholders, extra):
    ud = (AQUI / "aws" / nombre).read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(AQUI / "aws" / nombre)]).returncode == 0
    assert set(re.findall(r"__[A-Z0-9_]+__", ud)) == placeholders
    sust = ud
    for k in placeholders:
        sust = sust.replace(k, "0" * 64 if k == "__CRIT_SHA__" else "s3://chispa-pruebas-879381269751/x-" + "0" * 40 if k == "__CHISPA_S3__" else "2930")
    assert len(sust.encode()) <= 16384
    for s in ("dev.tgz", "sellad", "CRIT_SHA", "shutdown -h +$MIN", "torch==$TORCH", "https://download.pytorch.org/whl/cpu", *extra):
        assert s in ud, s
    assert "s3 ls" not in ud and "--dryrun" not in ud and "TORCH=2.7.1" in ud and "TRANSFORMERS=5.22.2" in ud
    return ud


def test_ud_t3a():
    ud = ud_ok("r10_t3a.sh", {"__CHISPA_S3__", "__CRIT_SHA__", "__N_TEST__", "__N_SOP__", "__NLI_FILAS__"}, ["MIN=${CHISPA_MIN:-40}", "mejor.pt", "pred knn-e5-k8", "mdeberta-nli"])
    assert ud.index("pred knn-vectores") < ud.index("pred l2-zs") < ud.index("pred mdeberta-nli")   # el más caro, el último; kNN no depende de L2 ni de NLI
    assert ud.index("pred knn-e5-k8") < ud.index("pred l2-zs")


def test_ud_t3b():
    ud = ud_ok("r10_t3b.sh", {"__CHISPA_S3__", "__CRIT_SHA__", "__N_TEST__", "__N_SOP__"}, ["MIN=${CHISPA_MIN:-45}", "t3.tgz", "entradas_t3_sha256", "rubricas_sha256", "--max-logprobs 20", "cdbee75f"])
    assert "14b" not in ud.lower() and ud.index("lp zs-topk") < ud.index("lp rub") < ud.index("lp fs8rub") < ud.index("lp fs8-topk")
