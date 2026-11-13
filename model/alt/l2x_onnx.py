"""P3: exporta L2x a ONNX fp32 e int8 dinámico y lo usa para predecir (misma entrada que `L2x`).

    python -m model.alt.l2x_onnx exportar --ckpt DIR --salida DIR        # DIR/l2x_fp32.onnx y DIR/l2x_int8.onnx

El grafo recibe ids (B,T), máscara (B,T), tramos de opción (B,K,T) y de texto (B,T) y devuelve las puntuaciones (B,K). La latencia se
mide con `model/alt/bench_l2x_i7.py` (solo numpy + onnxruntime, para correrlo donde no hay torch).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def exportar(ckpt, salida, opset=17):
    import torch
    import torch.nn as nn
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from .l2x import L2x

    m = L2x.cargar(ckpt, "cpu").eval()
    try:
        m.enc.config._attn_implementation = "eager"
    except Exception:  # noqa: BLE001
        pass

    class Grafo(nn.Module):
        def __init__(self, x):
            super().__init__()
            self.x = x

        def forward(self, input_ids, attention_mask, opcion_mask, texto_mask):
            return self.x.puntuar(input_ids, attention_mask, opcion_mask, texto_mask)

    g = Grafo(m).eval()
    sal = Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    fila = {"id": "x", "pregunta": "que es esto", "texto": "mensaje de ejemplo " * 4, "opciones": ["a", "b", "c"], "descripciones": {"a": "uno", "b": "dos", "c": "tres"}, "etiqueta": "a"}
    ent = m.tensores([(fila, ["a", "b", "c"]), (fila, ["a", "b"])])
    f32 = sal / "l2x_fp32.onnx"
    with torch.no_grad():
        ref = g(*ent)
        torch.onnx.export(g, ent, str(f32), input_names=["input_ids", "attention_mask", "opcion_mask", "texto_mask"], output_names=["puntuaciones"],
                          dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "opcion_mask": {0: "b", 1: "k", 2: "t"},
                                        "texto_mask": {0: "b", 1: "t"}, "puntuaciones": {0: "b", 1: "k"}}, opset_version=opset, dynamo=False)
    i8 = sal / "l2x_int8.onnx"
    quantize_dynamic(str(f32), str(i8), weight_type=QuantType.QInt8)
    out = {"fp32": str(f32), "int8": str(i8), "tam_fp32_mb": round(f32.stat().st_size / 1e6, 1), "tam_int8_mb": round(i8.stat().st_size / 1e6, 1)}
    for nombre in ("fp32", "int8"):
        s = sesion(out[nombre])
        got = s.run(None, dict(zip(("input_ids", "attention_mask", "opcion_mask", "texto_mask"), (x.numpy() for x in ent))))[0]
        out["max_dif_" + nombre] = float(np.abs(got[0, :3] - ref[0, :3].numpy()).max())
    (sal / "exportacion.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def sesion(ruta, hilos=4):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = hilos
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ruta), o, providers=["CPUExecutionProvider"])


class OrtL2x:
    """`probs(filas)`, `largo` y `partir` como `L2x`, con ORT. Más de `max_opciones` opciones: el mismo torneo que `L2x`."""

    def __init__(self, ruta_onnx, ckpt, hilos=4):
        import torch.nn as nn
        from transformers import AutoTokenizer

        from .l2x import L2x
        from .modelo import PREFIJOS

        cfg = json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8"))
        self.s = sesion(ruta_onnx, hilos)
        outer = self

        class Aux(L2x):   # solo la parte de entrada y el torneo; `_pase` usa ORT
            def __init__(s):
                nn.Module.__init__(s)
                s.tok = AutoTokenizer.from_pretrained(Path(ckpt) / "tokenizer")
                s.codificador, s.pooling, s.max_tok = cfg["codificador"], cfg["pooling"], cfg["max_tok"]
                s._largos, s.prefijo, s.max_opciones = {}, PREFIJOS.get(cfg["codificador"], ""), cfg.get("max_opciones", 8)
                s.marca_id = s.tok.convert_tokens_to_ids("[OPC]")

            def _pase(s, items):
                ids, mask, om, tm = (x.numpy() for x in s.tensores(items))
                sc = outer.s.run(None, {"input_ids": ids, "attention_mask": mask, "opcion_mask": om, "texto_mask": tm})[0]
                import torch
                return [torch.tensor(sc[i, :len(ops)].astype(np.float64)) for i, (_, ops) in enumerate(items)]

        self.aux = Aux()

    def largo(self, f):
        return self.aux.largo(f)

    def partir(self, filas, max_tokens):
        return self.aux.partir(filas, max_tokens)

    def probs(self, filas):
        return self.aux.probs(filas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exportar")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--salida", required=True)
    a = ap.parse_args(argv)
    exportar(a.ckpt, a.salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
