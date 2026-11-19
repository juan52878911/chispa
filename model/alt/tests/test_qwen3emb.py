"""L2-qwen3emb: exportador de ONNX y de tokenizador sobre el Qwen3 en miniatura, filas/config del servidor y la medida (veredictos del criterio)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from model.alt import qwen3emb_medir as M
from model.alt.aws import qwen3emb_filas as F
from model.alt.tests.dev_sintetico_qwen import crear as crear_dev

RAIZ = Path(__file__).resolve().parents[3]
FIXTURE = RAIZ / "runtime/tok/testdata/qwen/fixture_paridad.jsonl"
CRIT = RAIZ / "model/alt/crit-l2-qwen3emb.json"


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from model.alt.qwen3emb_mini import crear
    return crear(tmp_path_factory.mktemp("qmini"), FIXTURE)


def test_exportar_onnx_mini_y_metadatos(mini, tmp_path):
    import onnx
    from model.alt.exportar_qwen3emb_onnx import main
    main(["--modelo", str(mini), "--salida", str(tmp_path), "--comprobar", "--permitir-versiones"])
    info = json.load(open(tmp_path / "qwen3emb-onnx.json"))
    assert info["revision"] == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" and info["max_len"] == 320 and info["pooling"] == "last_token"
    assert info["comprobacion"]["qwen3emb-fp32.onnx"]["coseno_min_ultimo_token"] > 0.99999
    assert info["comprobacion"]["qwen3emb-int8.onnx"]["coseno_min_ultimo_token"] > 0.99
    m = onnx.load(str(tmp_path / "qwen3emb-fp32.onnx"), load_external_data=False)
    props = {p.key: p.value for p in m.metadata_props}
    assert props["chispa.pooling"] == "last_token" and props["chispa.max_len"] == "320" and props["chispa.normalizacion"] == "l2"
    assert (tmp_path / "qwen3emb-fp32.onnx.data").stat().st_size > 1000   # pesos externos siempre


def test_exportar_tokenizador_determinista_y_rechaza_otro(mini, tmp_path):
    sys.path.insert(0, str(RAIZ / "runtime/tok"))
    import exportar_bpe
    a, b = tmp_path / "a.bpe", tmp_path / "b.bpe"
    ia, ib = exportar_bpe.exportar(str(mini), str(a)), exportar_bpe.exportar(str(mini), str(b))
    assert a.read_bytes() == b.read_bytes() and ia["sha256"] == ib["sha256"] and ia["plantilla"][-1]["id"] == ia["pad"]
    # un post-procesador que no sabe exportar no pasa en silencio (el normalizador y el regex de Qwen2Tokenizer los fija la clase de transformers)
    d = tmp_path / "otro"
    d.mkdir()
    for f in mini.iterdir():
        (d / f.name).write_bytes(f.read_bytes())
    j = json.load(open(d / "tokenizer.json"))
    j["post_processor"] = {"type": "BertProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0]}
    json.dump(j, open(d / "tokenizer.json", "w"))
    with pytest.raises(SystemExit, match="BertProcessing"):
        exportar_bpe.exportar(str(d), str(tmp_path / "c.bpe"))


def test_filas_ocho_por_clase_y_config(tmp_path):
    crear_dev(tmp_path / "dev", n_test=10, por_clase=12)
    r = F.filas(tmp_path / "dev", tmp_path / "ent")
    sop = [json.loads(l) for l in open(tmp_path / "ent/soporte.jsonl")]
    assert r == "tareas=6 soporte=%d test=60" % len(sop)
    por = {}
    for s in sop:
        por[(s["tarea"], s["etiqueta"])] = por.get((s["tarea"], s["etiqueta"]), 0) + 1
    assert set(por.values()) == {8}
    # los mismos índices que muestrear(ys, 8, 0) de R11
    from model.alt.fusion_knn_l2 import muestrear
    from model.alt.lineas_base_v2 import cargar_dev
    for t, (s, _) in cargar_dev(tmp_path / "dev").items():
        idx = muestrear([x["etiqueta"] for x in s], 8, 0)
        assert [x["id"] for x in sop if x["tarea"] == t] == [s[i]["id"] for i in idx]
    F.config(tmp_path / "ent/tareas.json", tmp_path / "b", "o.onnx", "t.bpe", 8, tmp_path / "cfg.json")
    k = json.load(open(tmp_path / "cfg.json"))["knn"]
    assert k["codificador"] == "qwen3-emb" and k["vecinos"] == 8 and k["umbral"] == 0.0 and k["siempre"] == [] and len(k["tareas"]) == 6 and "l3" not in k


def _e2e(res, dev, modo, v, acierta):
    from model.alt.lineas_base_v2 import cargar_dev
    with open(res / ("e2e-%s-v%d-filas.jsonl" % (modo, v)), "w") as fh:
        for t, (_, ev) in cargar_dev(dev).items():
            for r in ev:
                y = r["etiqueta"] if acierta else next(o for o in r["opciones"] if o != r["etiqueta"])
                fh.write(json.dumps({"id": r["id"], "tarea": t, "http": 200, "probs": {o: (0.9 if o == y else 0.1 / (len(r["opciones"]) - 1)) for o in r["opciones"]}}) + "\n")
    lat = lambda p50: {"p50": p50, "p95": p50 * 2, "n": 10}
    json.dump({"latencia_ms": {"total_cliente": lat(180 if modo == "fp32" else 90), "cod_ms": lat(100), "knn_ms": lat(1)}}, open(res / ("e2e-%s-v%d-resumen.json" % (modo, v)), "w"))


def test_medir_veredictos(tmp_path):
    dev, res = tmp_path / "dev", tmp_path / "res"
    crear_dev(dev, n_test=20)
    res.mkdir()
    json.dump({"n": 200, "iguales": 200, "pasa": True}, open(res / "paridad-tokens.json", "w"))
    json.dump({"coseno": {"min": 0.9999, "n_bajo_0.999": 0}}, open(res / "paridad-fp32.json", "w"))
    for modo in ("fp32", "int8"):
        for v in (32, 8):
            _e2e(res, dev, modo, v, True)
    o = M.cifras(res, dev, CRIT)
    c = o["condiciones"]
    assert c["tokens"]["pasa"] and c["coseno"]["pasa"]
    assert c["e2e-fp32-v32"]["macro"] == 1.0 and c["e2e-fp32-v32"]["cobertura"] == 1.0 and c["e2e-fp32-v32"]["pasa"] is False   # lejos de 0,4675
    assert o["pasa_todo"] is False and o["completo"] is True
    assert c["int8_vs_fp32"] == {"v32": True, "v8": True, "regla": c["int8_vs_fp32"]["regla"]}
    assert o["eleccion"]["modo"] == "int8" and o["eleccion"]["vecinos"] == 32 and o["eleccion"]["nota_producto"] is None   # p50 90 ms ≤ 150
    # con las referencias del dev inventado el e2e pasa; y un int8 peor que el fp32 no se elige y la p50 de 180 ms lleva la nota
    crit = json.load(open(CRIT))
    crit["referencia_python"]["macro_dev_knn_qwen"] = {"v32": 1.0, "v8": 1.0}
    json.dump(crit, open(tmp_path / "c.json", "w"))
    _e2e(res, dev, "int8", 32, False)
    o = M.cifras(res, dev, tmp_path / "c.json")
    assert o["condiciones"]["e2e-fp32-v32"]["pasa"] is True and o["condiciones"]["int8_vs_fp32"]["v32"] is False
    assert o["eleccion"]["modo"] == "fp32" and o["eleccion"]["nota_producto"] == "recomendado con GPU o cuando la latencia no importe"
    # una fila sin respuesta baja la cobertura
    lines = open(res / "e2e-fp32-v32-filas.jsonl").read().splitlines()
    r0 = json.loads(lines[0]); r0["http"] = 503; r0["probs"] = None
    open(res / "e2e-fp32-v32-filas.jsonl", "w").write("\n".join([json.dumps(r0)] + lines[1:]) + "\n")
    o = M.cifras(res, dev, tmp_path / "c.json")
    assert o["condiciones"]["e2e-fp32-v32"]["cobertura"] < 1 and o["condiciones"]["e2e-fp32-v32"]["pasa"] is False


def test_int8_no_viable_solo_fp32(tmp_path):
    dev, res = tmp_path / "dev", tmp_path / "res"
    crear_dev(dev, n_test=20)
    res.mkdir()
    json.dump({"n": 200, "iguales": 200, "pasa": True}, open(res / "paridad-tokens.json", "w"))
    json.dump({"coseno": {"min": 0.9999, "n_bajo_0.999": 0}}, open(res / "paridad-fp32.json", "w"))
    json.dump({"viable": False, "motivo": "x"}, open(res / "int8-no-viable.json", "w"))
    for v in (32, 8):
        _e2e(res, dev, "fp32", v, True)
    o = M.cifras(res, dev, CRIT)
    assert o["completo"] is True and o["eleccion"]["modo"] == "fp32" and o["informativo"]["int8"]["viable"] is False
    assert o["condiciones"]["e2e-int8-v32"] == {"pasa": None, "motivo": "int8 no viable"}


def test_criterio_congelado_no_cambio():
    import hashlib
    assert hashlib.sha256(CRIT.read_bytes()).hexdigest() == "58a62d618a4f917c1c4c232934a31394874a59f4cb5b589e945efd91511ade0b"
