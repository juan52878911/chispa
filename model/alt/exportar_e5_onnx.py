"""T5b: exporta intfloat/multilingual-e5-small (e5 puro, sin ajustar) a ONNX con la salida `last_hidden_state`, para el codificador del kNN del servidor Go.

    python model/alt/exportar_e5_onnx.py --salida DIR [--cache HF_HOME] [--revision SHA] [--opset 17] [--int8] [--comprobar]

Versiones con las que se calcularon los vectores de los bancos del i7 (R6/R8): torch 2.5.1 CPU y transformers 4.46.3; el script se niega a
exportar con otras (los vectores de e5 cambian con la versión de transformers, ver `knn_soporte_llm.py`). El modelo se lee de la caché de HF
por revisión fijada (614241f6…, licencia MIT: se comprueba en la ficha del modelo) y sin red.

Salida en DIR: `e5.onnx` (fp32, ejes dinámicos de lote y secuencia, entradas `input_ids` y `attention_mask` int64, salida `last_hidden_state`
float32 (lote, T, 384)), `e5.onnx.sha256`, y con --int8 `e5-int8.onnx` (cuantización dinámica de ORT, pesos int8, para móvil) y su sha256.
Con --comprobar compara el ONNX con PyTorch en unas frases (error máximo de last_hidden_state) si onnxruntime está disponible.
El pooling (media con máscara) y la normalización L2 NO van en el grafo: los hace el codificador (server/decide/e5_ort.go), igual que
`model/l2/linea_base_knn_e5.py:vectores`.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
NOMBRE = "intfloat/multilingual-e5-small"
TORCH, TRANSFORMERS = "2.5.1", "4.46.3"
FRASES = ["query: hola, ¿cómo estás?", "query: " + "texto largo de prueba con ñ y acentos áéíóú " * 40, "query: ESTO ES UN MENSAJE EN MAYÚSCULAS", "query: a"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salida", required=True)
    ap.add_argument("--cache", default=os.environ.get("HF_HOME", ""), help="HF_HOME con models--intfloat--multilingual-e5-small")
    ap.add_argument("--revision", default=REV)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--comprobar", action="store_true")
    a = ap.parse_args(argv)
    import torch
    import transformers
    if not torch.__version__.startswith(TORCH) or transformers.__version__ != TRANSFORMERS:
        sys.exit("torch %s y transformers %s van fijados (los vectores de los bancos): hay %s y %s" % (TORCH, TRANSFORMERS, torch.__version__, transformers.__version__))
    snap = Path(a.cache) / "models--intfloat--multilingual-e5-small" / "snapshots" / a.revision
    if not snap.is_dir():
        sys.exit("falta %s" % snap)
    ficha = (snap / "README.md").read_text(encoding="utf-8")
    if "license: mit" not in ficha.split("---")[1].lower():
        sys.exit("la ficha del modelo no declara licencia MIT")
    from transformers import AutoModel
    m = AutoModel.from_pretrained(str(snap), local_files_only=True).eval()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)

    class E5(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    b = tok(FRASES[:2], padding=True, truncation=True, max_length=256, return_tensors="pt")
    ruta = sal / "e5.onnx"
    with torch.no_grad():
        torch.onnx.export(E5(m), (b["input_ids"], b["attention_mask"]), str(ruta), opset_version=a.opset, do_constant_folding=True,
                          input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                          dynamic_axes={"input_ids": {0: "lote", 1: "T"}, "attention_mask": {0: "lote", 1: "T"}, "last_hidden_state": {0: "lote", 1: "T"}})
    info = {"modelo": NOMBRE, "revision": a.revision, "licencia": "MIT", "opset": a.opset, "torch": torch.__version__, "transformers": transformers.__version__,
            "e5.onnx": {"bytes": ruta.stat().st_size, "sha256": sha256(ruta)}}
    (sal / "e5.onnx.sha256").write_text("%s  e5.onnx\n" % info["e5.onnx"]["sha256"])
    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        r8 = sal / "e5-int8.onnx"
        quantize_dynamic(str(ruta), str(r8), weight_type=QuantType.QInt8)
        info["e5-int8.onnx"] = {"bytes": r8.stat().st_size, "sha256": sha256(r8), "cuantizacion": "onnxruntime.quantization.quantize_dynamic, QInt8, pesos int8 y activaciones dinámicas"}
        (sal / "e5-int8.onnx.sha256").write_text("%s  e5-int8.onnx\n" % info["e5-int8.onnx"]["sha256"])
    if a.comprobar:
        import numpy as np
        import onnxruntime as ort
        info["onnxruntime_python"] = ort.__version__
        for nombre in [n for n in ("e5.onnx", "e5-int8.onnx") if (sal / n).exists()]:
            s = ort.InferenceSession(str(sal / nombre), providers=["CPUExecutionProvider"])
            peor = 0.0
            for f in FRASES:
                bb = tok([f], truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    ref = m(**bb).last_hidden_state.numpy()
                got = s.run(["last_hidden_state"], {"input_ids": bb["input_ids"].numpy(), "attention_mask": bb["attention_mask"].numpy()})[0]
                peor = max(peor, float(np.abs(ref - got).max()))
            info[nombre]["max_abs_dif_last_hidden_state_vs_torch"] = peor
    (sal / "e5-onnx.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
