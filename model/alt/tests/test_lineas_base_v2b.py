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


def escribir_pred(dev, dest, sistema, tareas, acierto, sesgo_alto=False, salto=1):
    for t, (_, ev) in lb.cargar_dev(dev).items():
        if t not in tareas:
            continue
        filas = []
        for i, r in enumerate(ev):
            ops = lb.orden(r)
            gold = ops.index(r["etiqueta"])
            idx = gold if (i % 10) < acierto * 10 else (gold + salto) % len(ops)
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


def test_qwk_pav_y_umbrales():
    g = np.array([1, 2, 3, 4, 5, 1, 2, 3])
    assert lb.qwk(g, g, list("12345")) == 1.0
    assert lb.qwk(g, np.full(8, 3), list("12345")) == 0.0          # constante: nada mejor que el azar
    assert lb.qwk(np.array([1, 5]), np.array([5, 1]), list("12345")) < 0
    assert vb.pav([1, 3, 2, 4]) == [1, 2.5, 2.5, 4]
    sop = [{"id": "s%d%d" % (n, i), "tipo": "score", "opciones": ["1", "2", "3"], "etiqueta": str(n)} for n in (1, 2, 3) for i in range(4)]
    preds = {r["id"]: {"valida": True, "probs": {o: (0.8 if o == str(int(r["etiqueta"])) else 0.1) for o in "123"}} for r in sop}
    u = vb.umbrales(sop, preds)
    assert len(u) == 2 and u[0] < u[1] and 1 < u[0] < 2 and 2 < u[1] < 3
    # un modelo sesgado hacia el 3: sus esperanzas suben, los umbrales se desplazan y la lectura calibrada recupera los niveles
    sesgo = {r["id"]: {"valida": True, "probs": {o: (0.2 + 0.2 * int(o) * (1 + int(r["etiqueta"]) * 0.1)) for o in "123"}} for r in sop}
    u2 = vb.umbrales(sop, sesgo)
    assert u2 != u
    assert vb.umbrales(sop, preds, excluir={"s11", "s12", "s13"}) is not None
    assert lb.top1(preds[sop[0]["id"]]["probs"], sop[0], esp=u) == "1" and lb.top1(preds[sop[-1]["id"]]["probs"], sop[-1], esp=u) == "3"


def test_mayoritaria_y_qwk_en_cifras(tmp_path):
    dev = dev_falso(tmp_path)
    pred = tmp_path / "pred"
    escribir_pred(dev, pred, "x", set(CRIT["tareas_dev"]), 1.0)
    c, _ = lb.cifras_sistema(lb.cargar_dev(dev), pred / "x", set(CRIT["sinteticas"]))
    j = c["tareas"]["jql-valor-educativo"]
    assert j["qwk"] == 1.0 and j["mayoritaria"]["clase"] == "1" and 0 < j["mayoritaria"]["exactitud_test"] <= 0.2 and j["mayoritaria"]["oraculo_test_exactitud"] >= j["mayoritaria"]["exactitud_test"]
    assert c["tareas"]["cfpb-quejas"]["mayoritaria"]["clase"] == "a" and "qwk" not in c["tareas"]["cfpb-quejas"]


def montar(tmp_path, rub_acc, sop_ok=True):
    dev = dev_falso(tmp_path)
    t3, pred = tmp_path / "t3", tmp_path / "pred"
    todas = set(CRIT["tareas_dev"])
    escribir_pred(dev, t3, "4b-zs", todas, 0.5, salto=2)
    escribir_pred(dev, t3, "4b-fs8", todas, 0.6)
    escribir_pred(dev, pred, "zs-topk", {"finanzas-documentos"}, 0.5, salto=2)
    escribir_pred(dev, pred, "fs8-topk", {"finanzas-documentos"}, 0.6)
    escribir_pred(dev, pred, "rub", set(SCORE), rub_acc)
    escribir_pred(dev, pred, "fs8rub", set(SCORE), 0.5)
    if sop_ok:   # predicciones del soporte (split train): nítidas, para ajustar los umbrales
        for sistema in ("zs-sop", "rub-sop"):
            for t in SCORE:
                _, ev = lb.cargar_dev(dev)[t]
                sop = [json.loads(l) for l in (dev / (t + ".jsonl")).read_text().splitlines() if json.loads(l)["split"] == "train"]
                d = pred / sistema
                d.mkdir(parents=True, exist_ok=True)
                (d / (t + ".jsonl")).write_text("".join(json.dumps({"id": r["id"], "valida": True, "ms": 1.0, "probs": {o: (0.8 if o == r["etiqueta"] else 0.05) for o in lb.orden(r)}}) + "\n" for r in sop), encoding="utf-8")
    return dev, t3, pred


def test_medir_compuesto_veredictos(tmp_path):
    dev, t3, pred = montar(tmp_path, 0.9)
    out = tmp_path / "c.json"
    vb.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["criterio_sha256"] == hashlib.sha256((AQUI / "crit-r10-t3b.json").read_bytes()).hexdigest()
    v = r["veredictos"]
    assert set(v) == set(CRIT["candidatos"])
    assert v["rub"]["macro_acc_dif"] > 0.3 and v["rub"]["macro_qwk_dif"] > 0.15 and v["rub"]["cobertura_score"] == 1.0
    assert v["base+esp"]["veredicto"] == "NO PASA" and v["rub"]["veredicto"] in ("PASA", "NO PASA")
    assert r["sistemas"]["base"]["tareas"]["finanzas-documentos"]["sintetica"] and r["sistemas"]["base"]["macro_sin_sinteticas"]["n_tareas"] == 4
    ms = r["sistemas"]["rub"]["macro_score"]
    assert ms["n_tareas"] == 3 and {"exactitud", "qwk", "mae", "exactitud_pm1", "exactitud_sin_sinteticas"} <= set(ms)
    assert r["sistemas"]["base+cal"]["tareas"]["jql-valor-educativo"]["modo_score"] == "umbrales" and set(r["umbrales"]["base+cal"]) == set(SCORE)
    assert len(r["umbrales"]["rub+cal"]["usajobs-nivel"]) == 4 and r["sistemas"]["mayoritaria"]["tareas"]["horeca-resenas"]["modo_score"] == "mediana"
    assert r["sistemas"]["base-argmax"]["tareas"]["horeca-resenas"]["modo_score"] == "argmax"
    assert set(r["pareadas_extra"]) == {"%s_menos_%s" % tuple(p) for p in CRIT["pares_extra"]}
    assert r["v4_topk"]["veredicto"] == "informativo" and r["v4_topk"]["filas_comunes"] == 40
    assert "texto" not in out.read_text(encoding="utf-8")


def test_pasa_exige_no_quedar_bajo_la_mayoritaria(tmp_path):
    dev, t3, pred = montar(tmp_path, 0.9)
    crit = dict(CRIT, pasa=dict(CRIT["pasa"]))
    out = tmp_path / "c.json"
    vb.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    # con tareas equilibradas la mayoritaria es ≈ 0,2: una rúbrica de 0,9 la supera en las 3 y PASA; con 0,1 de acierto queda bajo la mayoritaria
    assert r["veredictos"]["rub"]["bajo_mayoritaria"] == [] and r["veredictos"]["rub"]["veredicto"] == "PASA"
    dev2, t32, pred2 = montar(tmp_path / "b", 0.1) if (tmp_path / "b").mkdir() is None else (None, None, None)
    out2 = tmp_path / "c2.json"
    vb.main(["medir", "--dev", str(dev2), "--pred", str(pred2), "--t3", str(t32), "--salida", str(out2)])
    r2 = json.loads(out2.read_text(encoding="utf-8"))
    assert r2["veredictos"]["rub"]["veredicto"] == "NO PASA" and r2["veredictos"]["rub"]["bajo_mayoritaria"]
    assert crit["pasa"]["acc_min"] == 0.05 and crit["pasa"]["qwk_min"] == 0.15


def test_medir_sin_cobertura_es_no_medido(tmp_path):
    dev = dev_falso(tmp_path)
    t3, pred = tmp_path / "t3", tmp_path / "pred"
    escribir_pred(dev, t3, "4b-zs", set(CRIT["tareas_dev"]), 0.5)
    escribir_pred(dev, t3, "4b-fs8", set(CRIT["tareas_dev"]), 0.5)
    escribir_pred(dev, pred, "zs-topk", {"finanzas-documentos"}, 0.5)
    escribir_pred(dev, pred, "rub", {"horeca-resenas"}, 0.9)   # faltan jql y usajobs y todo el soporte
    out = tmp_path / "c.json"
    vb.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["veredictos"]["rub"]["veredicto"] == "no medido" and r["veredictos"]["base+cal"]["veredicto"] in ("PASA", "NO PASA", "no medido")
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
    assert "14b" not in ud.lower() and ud.index("lp zs-sop") < ud.index("lp rub-sop") < ud.index("lp rub \"") < ud.index("lp zs-topk") < ud.index("lp fs8-topk") < ud.index("lp fs8rub")


def test_calibracion_sin_soporte_completo_es_no_medido(tmp_path):
    dev, t3, pred = montar(tmp_path, 0.9, sop_ok=False)
    out = tmp_path / "c.json"
    vb.main(["medir", "--dev", str(dev), "--pred", str(pred), "--t3", str(t3), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["veredictos"]["base+cal"]["veredicto"] == "no medido" and r["veredictos"]["rub+cal"]["veredicto"] == "no medido"
    assert r["veredictos"]["rub"]["veredicto"] in ("PASA", "NO PASA")   # lo que no necesita el soporte se mide igual
