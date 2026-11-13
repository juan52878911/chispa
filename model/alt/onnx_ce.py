"""P2: exporta el cross-encoder del hito 4 (`model/alt/modelo.py`) a ONNX fp32 y a int8 dinámico, y lo mide.

    python -m model.alt.onnx_ce exportar --ckpt DIR --salida DIR            # DIR/ce_fp32.onnx y DIR/ce_int8.onnx
    python -m model.alt.onnx_ce latencia --onnx F.onnx [--hilos 4] [--tokens 128] [--opciones 2 4 8 16] [--n 200] --salida J.json

El grafo es codificador + media con máscara + cabeza lineal: entra (ids, máscara) de N secuencias (una por opción) y sale una
puntuación por secuencia. Una petición de N opciones es UNA llamada a ORT con lote N (una pasada por opción, todas en el mismo lote);
el softmax por pregunta se hace fuera. `latencia` usa ids sintéticos (el coste no depende del contenido) y secuencias llenas de
`--tokens` tokens, sin relleno: es el peor caso para ese presupuesto. Pensado para correr igual en el PC de pruebas y en el i7.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- exportar
def exportar(ckpt, salida, opset=17):
    import torch
    import torch.nn as nn
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from .modelo import CrossEncoder

    m = CrossEncoder.cargar(ckpt, "cpu").eval()
    if m.pooling != "media":
        raise ValueError("solo se exporta el pooling por media")
    try:
        m.enc.config._attn_implementation = "eager"   # el grafo con atención sdpa trae ramas que ORT no necesita
    except Exception:  # noqa: BLE001
        pass

    class Grafo(nn.Module):
        def __init__(self, enc, cabeza):
            super().__init__()
            self.enc, self.cabeza = enc, cabeza

        def forward(self, input_ids, attention_mask):
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mk = attention_mask.unsqueeze(-1).to(h.dtype)
            v = (h * mk).sum(1) / mk.sum(1).clamp(min=1)
            return self.cabeza(v).squeeze(-1)

    g = Grafo(m.enc, m.cabeza).eval()
    sal = Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    ids = torch.randint(5, 1000, (3, 20))
    mask = torch.ones_like(ids)
    mask[0, 15:] = 0
    f32 = sal / "ce_fp32.onnx"
    with torch.no_grad():
        ref = g(ids, mask)
        torch.onnx.export(g, (ids, mask), str(f32), input_names=["input_ids", "attention_mask"], output_names=["puntuacion"],
                          dynamic_axes={"input_ids": {0: "n", 1: "t"}, "attention_mask": {0: "n", 1: "t"}, "puntuacion": {0: "n"}},
                          opset_version=opset, dynamo=False)
    i8 = sal / "ce_int8.onnx"
    quantize_dynamic(str(f32), str(i8), weight_type=QuantType.QInt8)
    out = {"fp32": str(f32), "int8": str(i8), "tam_fp32_mb": round(f32.stat().st_size / 1e6, 1), "tam_int8_mb": round(i8.stat().st_size / 1e6, 1)}
    for nombre in ("fp32", "int8"):
        s = sesion(out[nombre])
        got = s.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
        out["max_dif_" + nombre] = float(np.abs(got - ref.numpy()).max())
    print(json.dumps(out, indent=1))
    return out


def sesion(ruta, hilos=4):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = hilos
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ruta), o, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- predecir con ORT (misma entrada que CrossEncoder)
class OrtCE:
    """`probs(filas)` y `largo(fila)` como `CrossEncoder`, con el grafo ONNX en lugar de PyTorch."""

    def __init__(self, ruta_onnx, ckpt, hilos=4):
        import torch.nn as nn
        from transformers import AutoTokenizer

        from .modelo import PREFIJOS, CrossEncoder

        cfg = json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8"))

        class Aux(CrossEncoder):   # solo la parte de entrada: sin codificador ni cabeza
            def __init__(s):
                nn.Module.__init__(s)
                s.tok, s.codificador, s.pooling, s.max_tok = AutoTokenizer.from_pretrained(Path(ckpt) / "tokenizer"), cfg["codificador"], cfg["pooling"], cfg["max_tok"]
                s._largos, s.prefijo = {}, PREFIJOS.get(cfg["codificador"], "")

        self.aux = Aux()
        self.s = sesion(ruta_onnx, hilos)

    def largo(self, f):
        return self.aux.largo(f)

    def probs(self, filas):
        from .modelo import opciones_ordenadas
        ids, mask, tams = self.aux.codificar(filas)
        sc = self.s.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0].astype(np.float64)
        out, i = [], 0
        for f, n in zip(filas, tams):
            x = sc[i:i + n]
            i += n
            e = np.exp(x - x.max())
            out.append(dict(zip(opciones_ordenadas(f), (e / e.sum()).tolist())))
        return out


# ---------------------------------------------------------------- latencia
def percentil(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def latencia(onnx, hilos=4, tokens=128, opciones=(2, 4, 8, 16), n=200, calentar=10, vocab=250000, semilla=0):
    s = sesion(onnx, hilos)
    rng = np.random.RandomState(semilla)
    res = {"onnx": str(onnx), "hilos": hilos, "tokens": tokens, "n": n, "por_opciones": {}}
    for k in opciones:
        ms = []
        for i in range(calentar + n):
            ids = rng.randint(5, vocab, (k, tokens)).astype(np.int64)
            mask = np.ones_like(ids)
            t0 = time.perf_counter()
            s.run(None, {"input_ids": ids, "attention_mask": mask})
            if i >= calentar:
                ms.append((time.perf_counter() - t0) * 1000)
        res["por_opciones"][str(k)] = {"p50_ms": round(statistics.median(ms), 2), "p99_ms": round(percentil(ms, 99), 2), "media_ms": round(statistics.mean(ms), 2)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exportar")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--salida", required=True)
    la = sub.add_parser("latencia")
    la.add_argument("--onnx", required=True)
    la.add_argument("--hilos", type=int, default=4)
    la.add_argument("--tokens", type=int, default=128)
    la.add_argument("--opciones", type=int, nargs="*", default=[2, 4, 8, 16])
    la.add_argument("--n", type=int, default=200)
    la.add_argument("--salida")
    a = ap.parse_args(argv)
    if a.cmd == "exportar":
        exportar(a.ckpt, a.salida)
    else:
        r = latencia(a.onnx, a.hilos, a.tokens, a.opciones, a.n)
        print(json.dumps(r, indent=1))
        if a.salida:
            Path(a.salida).write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
