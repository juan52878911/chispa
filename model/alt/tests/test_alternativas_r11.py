"""R11: la medida (`medir`) sobre predicciones sintéticas, el bootstrap pareado, la plantilla del reranker y el ud."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from model.alt import alternativas_r11 as A

RAIZ = Path(__file__).resolve().parents[1]
TAREAS = {"t-choice": ("choice", ["a", "b", "c"]), "t-score": ("score", ["1", "2", "3", "4", "5"])}


def _dev(d, n=40):
    d.mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(0)
    for t, (tipo, ops) in TAREAS.items():
        with open(d / (t + ".jsonl"), "w", encoding="utf-8") as fh:
            for sp, m in (("train", 2 * len(ops)), ("test", n)):
                for i in range(m):
                    y = ops[i % len(ops)] if sp == "train" else ops[rs.randint(len(ops))]
                    fh.write(json.dumps({"id": "%s:%s:%d" % (t, sp, i), "tipo": tipo, "pregunta": "¿?", "texto": "x%d" % i, "opciones": ops,
                                         "etiqueta": y, "split": sp}) + "\n")


def _pred(d, t, ev, acierta, extra=False):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / (t + ".jsonl"), "w", encoding="utf-8") as fh:
        for r, ok in zip(ev, acierta):
            y = r["etiqueta"] if ok else next(o for o in r["opciones"] if o != r["etiqueta"])
            p = {o: (0.9 if o == y else 0.1 / (len(r["opciones"]) - 1)) for o in r["opciones"]}
            x = {"id": r["id"], "probs": p, "valida": True}
            if extra:
                x["yes"] = {o: float(np.log(v)) for o, v in p.items()}
                x["lsm"] = x["yes"]
            fh.write(json.dumps(x) + "\n")


def test_medir_veredictos(tmp_path):
    from model.alt.lineas_base_v2 import cargar_dev
    _dev(tmp_path / "dev")
    dev = cargar_dev(tmp_path / "dev")
    P, R = tmp_path / "pred", tmp_path / "ref"
    rs = np.random.RandomState(1)
    for t, (_, ev) in dev.items():
        n = len(ev)
        base = rs.rand(n) < 0.5
        mejor = base | (rs.rand(n) < 0.3)            # Qwen ≥ e5 fila a fila: diferencia clara
        _pred(P / "knn-e5-v32", t, ev, base)
        _pred(P / "knn-qwen-v32", t, ev, mejor)
        _pred(R / "knn-e5-k8-t3a", t, ev, base)
        _pred(R / "4b-zs", t, ev, base)
        _pred(R / "l2-zs", t, ev, rs.rand(n) < 0.2)
        _pred(P / "rerank", t, ev, base, extra=True)  # igual que el 4B, así que pasa solo por latencia
    (P / "rerank" / "lat-rerank.json").write_text(json.dumps({"t-choice": [10.0, 12.0], "t-score": [11.0]}))
    crit = json.loads((RAIZ / "crit-r11.json").read_text(encoding="utf-8"))
    (tmp_path / "c.json").write_text(json.dumps(crit))
    A.main(["medir", "--dev", str(tmp_path / "dev"), "--pred", str(P), "--ref", str(R), "--crit", str(tmp_path / "c.json"), "--salida", str(tmp_path / "o.json")])
    o = json.loads((tmp_path / "o.json").read_text())
    v = o["veredicto"]
    assert v["A"]["estado"] == "PASA" and v["A"]["ic95"][0] > 0
    assert v["B"]["estado"] == "PASA" and v["B"]["veces_mas_rapido"] == round(104.5 / 11.0, 2) and abs(v["B"]["dif_macro_vs_4b"]) < 1e-9
    assert v["C"]["estado"] == "no medido: licencia"
    assert o["reproduce_t3a"] == {"t-choice": 1.0, "t-score": 1.0}
    assert o["sistemas"]["knn-e5-v8"]["cobertura_total"] is False   # lo que no se midió queda sin cobertura, no como cero silencioso


def test_b_sin_latencia_y_peor_no_pasa(tmp_path):
    from model.alt.lineas_base_v2 import cargar_dev
    _dev(tmp_path / "dev")
    dev = cargar_dev(tmp_path / "dev")
    P, R = tmp_path / "pred", tmp_path / "ref"
    for t, (_, ev) in dev.items():
        _pred(R / "4b-zs", t, ev, [True] * len(ev))
        _pred(R / "l2-zs", t, ev, [False] * len(ev))
        _pred(P / "rerank", t, ev, [i % 2 == 0 for i in range(len(ev))], extra=True)
    (P / "rerank" / "lat-rerank.json").write_text(json.dumps({"t-choice": [1.0]}))
    A.main(["medir", "--dev", str(tmp_path / "dev"), "--pred", str(P), "--ref", str(R), "--crit", str(RAIZ / "crit-r11.json"), "--salida", str(tmp_path / "o.json")])
    v = json.loads((tmp_path / "o.json").read_text())["veredicto"]
    assert v["B"]["estado"] == "NO PASA"            # 100 veces más rápido, pero −50 puntos
    assert v["A"]["estado"] == "no medido"


def test_ic_pareado_es_cero_si_son_iguales():
    o = {"t": np.array([1.0, 0, 1, 1, 0]), "u": np.array([0.0, 1, 1])}
    assert A.ic_dif_macro(o, o) == [0.0, 0.0]


def test_restringir_a_las_candidatas():
    p = {"i": {"yes": {"a": 0.0, "b": 5.0, "c": 1.0}}}
    r = A.restringir(p, {"i": ["a", "c"]})
    assert set(r["i"]["probs"]) == {"a", "c"} and r["i"]["probs"]["c"] > r["i"]["probs"]["a"]


def test_plantillas():
    r = {"pregunta": "¿Q?", "texto": "T", "opciones": ["x"], "descripciones": {"x": "equis"}}
    assert A.cuerpo_rr(r, "T", "x") == "<Instruct>: ¿Q?\n<Query>: T\n<Document>: equis"
    assert A.consulta_emb(r) == "Instruct: Classify the text to answer this question: ¿Q?\nQuery:T"
    assert A.PREFIJO_RR.startswith("<|im_start|>system\nJudge whether the Document") and A.SUFIJO_RR.endswith("<think>\n\n</think>\n\n")


def test_criterio_y_ud():
    crit = json.loads((RAIZ / "crit-r11.json").read_text(encoding="utf-8"))
    assert crit["C"]["estado"] == "no medido: licencia" and crit["B"]["lat_ref_4b_ms_p50"] == 104.5
    ud = (RAIZ / "aws" / "r11_alternativas.sh").read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(RAIZ / "aws" / "r11_alternativas.sh")]).returncode == 0
    assert "shutdown -h +$MIN" in ud and "MIN=${CHISPA_MIN:-90}" in ud and "verificado" in ud and "listo" in ud
    for rev in ("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "e61197ed45024b0ed8a2d74b80b4d909f1255473", "614241f622f53c4eeff9890bdc4f31cfecc418b3"):
        assert rev in ud
    assert "Horizon-Labs" not in ud.split("# --- 1. entorno", 1)[1]   # C no se baja
    # los imports se comprueban antes de bajar los modelos
    assert ud.index('import torch, transformers, numpy, huggingface_hub, sentencepiece') < ud.index("# --- 3. modelos")
    assert ud.index("import model.alt.alternativas_r11") < ud.index("# --- 3. modelos")
    assert len(re.findall(r"__[A-Z0-9_]+__", ud)) >= 5
    assert hashlib.sha256((RAIZ / "crit-r11.json").read_bytes()).hexdigest()   # el ud compara este sha con el congelado
