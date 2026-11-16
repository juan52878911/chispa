"""R10-T3: la medida de líneas base (numpy), la lectura congelada y el user-data. Sin red."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from model.alt import lineas_base_v2 as lb

AQUI = Path(__file__).resolve().parents[1]
CRIT = json.loads((AQUI / "crit-r10-t3.json").read_text(encoding="utf-8"))


def tarea(nombre, tipo, n=40, k=5):
    ops = [str(i + 1) for i in range(k)] if tipo == "score" else ["a", "b", "c"]
    rs = [{"id": "%s-s%d" % (nombre, i), "tarea": nombre, "tipo": tipo, "pregunta": "?", "texto": "t", "opciones": ops, "descripciones": {},
           "etiqueta": ops[i % len(ops)], "split": "train"} for i in range(6)]
    rs += [{"id": "%s-t%d" % (nombre, i), "tarea": nombre, "tipo": tipo, "pregunta": "?", "texto": "t", "opciones": ops, "descripciones": {},
            "etiqueta": ops[i % len(ops)], "split": "test"} for i in range(n)]
    return rs


def dev_falso(tmp_path):
    d = tmp_path / "dev"
    d.mkdir()
    for t in CRIT["tareas_dev"]:
        tipo = "score" if t in ("horeca-resenas", "jql-valor-educativo", "usajobs-nivel") else "choice"
        (d / (t + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in tarea(t, tipo)), encoding="utf-8")
    return d


def pred_falsa(dev, dest, acierto, cubrir=1.0):
    """Predicciones con acierto fijo: los casos según i mod 10 aciertan; el resto pone la masa en otra opción."""
    for t, (_, ev) in lb.cargar_dev(dev).items():
        filas = []
        for i, r in enumerate(ev[:int(len(ev) * cubrir)]):
            ops = lb.orden(r)
            gold = ops.index(r["etiqueta"])
            idx = gold if (i % 10) < acierto * 10 else (gold + 1) % len(ops)
            filas.append({"id": r["id"], "probs": {o: (0.9 if j == idx else 0.1 / (len(ops) - 1)) for j, o in enumerate(ops)}, "valida": True, "ms": 10 + i})
        Path(dest / (t + ".jsonl")).parent.mkdir(parents=True, exist_ok=True)
        (dest / (t + ".jsonl")).write_text("".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")


def test_top1_score_es_la_mediana_coral():
    r = {"tipo": "score", "opciones": ["1", "2", "3", "4", "5"]}
    assert lb.top1({"1": 0.1, "2": 0.1, "3": 0.2, "4": 0.3, "5": 0.3}, r) == "4"     # P(y>3)=0,6>0,5 pero P(y>4)=0,3
    assert lb.top1({"1": 0.5, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.2}, r) == "1"
    assert lb.top1(None, r) is None and lb.top1({"1": 0.0}, r) is None
    assert lb.top1({"a": 0.2, "b": 0.7}, {"tipo": "choice", "opciones": ["a", "b"]}) == "b"


def test_macro_ic_por_tareas_y_sinteticas(tmp_path, capsys):
    dev = dev_falso(tmp_path)
    pred = tmp_path / "pred"
    pred_falsa(dev, pred / "4b-zs", 0.5)
    pred_falsa(dev, pred / "14b-awq-zs", 0.8)
    pred_falsa(dev, pred / "mdeberta-nli", 0.3, cubrir=0.5)
    out = tmp_path / "c.json"
    lb.main(["medir", "--dev", str(dev), "--pred", str(pred), "--salida", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["criterio_sha256"] == __import__("hashlib").sha256((AQUI / "crit-r10-t3.json").read_bytes()).hexdigest()
    c = r["sistemas"]["4b-zs"]
    assert c["tareas"]["finanzas-documentos"]["sintetica"] and c["tareas"]["horeca-resenas"]["sintetica"] and not c["tareas"]["cfpb-quejas"]["sintetica"]
    assert c["macro"]["n_tareas"] == 6 and c["macro_sin_sinteticas"]["n_tareas"] == 4
    lo, hi = c["macro"]["ic95_tareas"]
    assert lo <= c["macro"]["valor"] <= hi
    assert "exactitud_pm1" in c["tareas"]["usajobs-nivel"] and "mae" in c["tareas"]["usajobs-nivel"] and "mae" not in c["tareas"]["cfpb-quejas"]
    assert r["sistemas"]["mdeberta-nli"]["cobertura_total"] == 0.5 and r["sistemas"]["mdeberta-nli"]["tareas"]["cfpb-quejas"]["invalidas"] == 20
    assert set(r["pareadas"]) == {"14b-awq-zs", "mdeberta-nli"}
    p = r["pareadas"]["14b-awq-zs"]
    assert p["macro"]["dif"] > 0.2 and set(p["tareas"]["cfpb-quejas"]) >= {"dif", "ic95", "mcnemar_p", "solo_sistema", "solo_referencia"}
    v = r["veredicto"]
    assert v["delta"] == p["macro"]["dif"] and v["lectura"] == CRIT["lectura"]["si_ge"] and "delta_sin_sinteticas" in v
    json.dumps(r)   # sin textos: solo cifras e ids de sistemas/tareas
    assert "texto" not in out.read_text(encoding="utf-8")


def test_lectura_congelada_umbrales_y_cobertura():
    crit = CRIT
    def v(delta, cob=1.0):
        out = {"referencia": "4b-zs", "sistemas": {"4b-zs": {"cobertura_total": 1.0}, "14b-awq-zs": {"cobertura_total": cob}},
               "pareadas": {"14b-awq-zs": {"macro": {"dif": delta, "ic95_tareas": [0, 0]}, "macro_sin_sinteticas": {"dif": 0.0, "ic95_tareas": [0, 0]}}}}
        return lb.veredicto(out, crit)["lectura"]
    assert v(0.08) == "el cuello es L3" and v(0.12) == "el cuello es L3"
    assert v(0.0799) == "indeterminado" and v(0.03) == "indeterminado"
    assert v(0.0299) == "el cuello es la especificación de la tarea" and v(-0.05) == "el cuello es la especificación de la tarea"
    assert v(0.2, cob=0.9) == "no medido"
    assert lb.veredicto({"referencia": "4b-zs", "sistemas": {"4b-zs": {}}, "pareadas": {}}, crit)["lectura"] == "no medido"


def test_criterio_dice_lo_pedido():
    assert CRIT["sinteticas"] == ["finanzas-documentos", "horeca-resenas"] and len(CRIT["tareas_dev"]) == 6
    assert "TAREAS" in CRIT["ic95"]["macro"] and "selladas" in CRIT["datos"] and "dev.tgz" in CRIT["datos"]
    assert CRIT["lectura"]["delta_ge"] == 0.08 and CRIT["lectura"]["delta_lt"] == 0.03


def test_mcnemar_exacto():
    a, b = np.array([1, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 1])
    assert lb.mcnemar(a, b) == (3, 1, 0.625)


def test_filas_separa_test_y_soporte(tmp_path, capsys):
    dev = dev_falso(tmp_path)
    lb.main(["filas", "--dev", str(dev), "--salida", str(tmp_path / "t")])
    assert capsys.readouterr().out.strip() == "tareas=6 test=240 soporte=36"
    assert len((tmp_path / "t/soporte/cfpb-quejas.jsonl").read_text().splitlines()) == 6


def test_ud_sintaxis_placeholders_y_tamano():
    ud = (AQUI / "aws" / "r10_t3.sh").read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(AQUI / "aws" / "r10_t3.sh")]).returncode == 0
    assert set(re.findall(r"__[A-Z0-9_]+__", ud)) == {"__CHISPA_S3__", "__CRIT_SHA__", "__N_TEST__", "__N_SOP__"}
    sust = ud.replace("__CHISPA_S3__", "s3://chispa-pruebas-cuenta-aws/r10-t3-" + "0" * 40).replace("__CRIT_SHA__", "0" * 64).replace("__N_TEST__", "2930").replace("__N_SOP__", "700")
    assert len(sust.encode()) <= 16384
    assert "dev.tgz" in ud and "sellad" in ud and "--max-logprobs 20" in ud and "shutdown -h +$MIN" in ud and "CRIT_SHA" in ud
    assert "s3 ls" not in ud and "--dryrun" not in ud
