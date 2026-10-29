"""L2-qwen3emb: exporta Qwen3-Embedding-0.6B a ONNX con la salida `last_hidden_state`, para el codificador del kNN del servidor Go (decide/qwen_ort.go).

    python model/alt/exportar_qwen3emb_onnx.py --modelo DIR --salida DIR_SALIDA [--revision SHA] [--opset 17] [--comprobar] [--sin-int8] [--permitir-versiones]

DIR es una copia local del modelo en la revisión fijada (97b0c614…, Apache-2.0: se comprueba en la ficha si hay README.md); se lee sin red. Versiones fijadas: torch,
transformers, onnx y onnxruntime de PINS (la referencia de paridad usa transformers 5.22.2); con --permitir-versiones solo avisa (pruebas con el modelo en miniatura).

Salida en DIR_SALIDA:
  qwen3emb-fp32.onnx (+ qwen3emb-fp32.onnx.data): fp32, ejes dinámicos de lote y secuencia, entradas `input_ids` y `attention_mask` int64, salida
      `last_hidden_state` float32 (lote, T, 1024). Los pesos van SIEMPRE en un único fichero de datos externos: el modelo pasa de 2 GB y el
      protobuf no lo admite en uno. Quien abra el ONNX debe hacerlo por ruta (no por bytes), con el .data a su lado.
  qwen3emb-int8.onnx: int8 dinámico de ORT (quantize_dynamic, QInt8) del anterior (salvo --sin-int8).
  qwen3emb-onnx.json: versiones, tamaños y sha256 de cada fichero (y de la comprobación contra torch con --comprobar).
Los metadatos del ONNX (metadata_props) llevan chispa.max_len=320 y chispa.pooling=last_token (más la normalización L2 y la instrucción): el servidor los comprueba al
abrir. El pooling (última posición) y la normalización L2 NO van en el grafo: los hace el codificador Go; con un lote de 1 no hay relleno y la última
posición es el último token. La atención se exporta «eager» (el trazado clásico de torch no admite la de sdpa con las máscaras de transformers 5).
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
NOMBRE = "Qwen/Qwen3-Embedding-0.6B"
PINS = {"torch": "2.15.0", "transformers": "5.22.2", "onnx": "1.23.2", "onnxruntime": "1.31.1"}
MAX_LEN = 320
INSTR = "Classify the text to answer this question: %s"
FRASES = ["Instruct: " + INSTR % "¿Qué emoción expresa el mensaje?" + "\nQuery:hola, ¿cómo estás? 😀", "Instruct: " + INSTR % "Is the review positive?" + "\nQuery:" + "texto largo de prueba con ñ y acentos áéíóú " * 30,
          "Instruct: " + INSTR % "x" + "\nQuery:ESTO ES UN MENSAJE EN MAYÚSCULAS", "a"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def versiones(permitir):
    import onnx
    import onnxruntime
    import torch
    import transformers
    hay = {"torch": torch.__version__.split("+")[0], "transformers": transformers.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__}
    mal = {k: (v, PINS[k]) for k, v in hay.items() if v != PINS[k]}
    if mal and not permitir:
        sys.exit("versiones distintas de las fijadas (hay, fijada): %s (con --permitir-versiones solo se avisa)" % mal)
    if mal:
        print("AVISO: versiones distintas de las fijadas:", mal, file=sys.stderr)
    return hay


def poner_metadatos(ruta, revision, opset):
    """Añade metadata_props al ONNX SIN cargar los datos externos y lo reescribe en su sitio."""
    import onnx
    m = onnx.load(str(ruta), load_external_data=False)
    del m.metadata_props[:]
    for k, v in (("chispa.codificador", "qwen3-embedding-0.6b"), ("chispa.revision", revision), ("chispa.max_len", str(MAX_LEN)), ("chispa.pooling", "last_token"),
                 ("chispa.normalizacion", "l2"), ("chispa.instruccion", INSTR), ("chispa.opset", str(opset))):
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(ruta))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--revision", default=REV)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--comprobar", action="store_true")
    ap.add_argument("--sin-int8", action="store_true")
    ap.add_argument("--permitir-versiones", action="store_true")
    a = ap.parse_args(argv)
    hay = versiones(a.permitir_versiones)
    import onnx
    import torch
    from transformers import AutoModel, AutoTokenizer
    mod = Path(a.modelo)
    if len(mod.name) == 40 and mod.name != a.revision and all(c in "0123456789abcdef" for c in mod.name):
        sys.exit("el directorio %s es de otra revisión que %s" % (mod.name, a.revision))
    if (mod / "README.md").exists() and "license: apache-2.0" not in (mod / "README.md").read_text(encoding="utf-8").split("---")[1].lower():
        sys.exit("la ficha del modelo no declara Apache-2.0")
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(mod), padding_side="left", local_files_only=True)
    m = AutoModel.from_pretrained(str(mod), dtype=torch.float32, attn_implementation="eager", local_files_only=True).eval()

    class Q(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    b = tok(FRASES[:2], padding=True, truncation=True, max_length=48, return_tensors="pt")
    fp32 = sal / "qwen3emb-fp32.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "m.onnx"
        with torch.no_grad():
            torch.onnx.export(Q(m), (b["input_ids"], b["attention_mask"]), str(t), opset_version=a.opset, dynamo=False, do_constant_folding=True,
                              input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                              dynamic_axes={"input_ids": {0: "lote", 1: "T"}, "attention_mask": {0: "lote", 1: "T"}, "last_hidden_state": {0: "lote", 1: "T"}})
        mo = onnx.load(str(t))   # con los datos externos que haya dejado el exportador (a partir de 2 GB, un fichero por peso)
        for x in (fp32, Path(str(fp32) + ".data")):
            x.unlink(missing_ok=True)
        onnx.save(mo, str(fp32), save_as_external_data=True, all_tensors_to_one_file=True, location=fp32.name + ".data", size_threshold=1024)
        del mo
    poner_metadatos(fp32, a.revision, a.opset)
    info = {"modelo": NOMBRE, "revision": a.revision, "licencia": "Apache-2.0", "opset": a.opset, "max_len": MAX_LEN, "pooling": "last_token", "versiones": hay, "ficheros": {}}
    info["ficheros"][fp32.name] = {"bytes": fp32.stat().st_size, "sha256": sha256(fp32)}
    info["ficheros"][fp32.name + ".data"] = {"bytes": Path(str(fp32) + ".data").stat().st_size, "sha256": sha256(str(fp32) + ".data")}
    refs = []   # último token de torch para FRASES (se calcula antes de soltar el modelo: la cuantización necesita la memoria)
    if a.comprobar:
        import numpy as np
        for f in FRASES:
            bb = tok([f], truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                refs.append((bb["input_ids"].numpy(), bb["attention_mask"].numpy(), m(**bb).last_hidden_state.numpy()[0, -1]))
    del m
    import gc
    gc.collect()
    if not a.sin_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        r8 = sal / "qwen3emb-int8.onnx"
        quantize_dynamic(str(fp32), str(r8), weight_type=QuantType.QInt8, use_external_data_format=True)
        poner_metadatos(r8, a.revision, a.opset)
        info["ficheros"][r8.name] = {"bytes": r8.stat().st_size, "sha256": sha256(r8), "cuantizacion": "onnxruntime.quantization.quantize_dynamic, QInt8, pesos int8 y activaciones dinámicas"}
        if Path(str(r8) + ".data").exists():
            info["ficheros"][r8.name + ".data"] = {"bytes": Path(str(r8) + ".data").stat().st_size, "sha256": sha256(str(r8) + ".data")}
    if a.comprobar:
        import onnxruntime as ort
        info["comprobacion"] = {}
        for nombre in [n for n in ("qwen3emb-fp32.onnx", "qwen3emb-int8.onnx") if (sal / n).exists()]:
            s = ort.InferenceSession(str(sal / nombre), providers=["CPUExecutionProvider"])
            peor, cmin = 0.0, 1.0
            for ids, mask, ref in refs:
                got = s.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0][0, -1]
                peor = max(peor, float(np.abs(ref - got).max()))
                cmin = min(cmin, float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got))))
            info["comprobacion"][nombre] = {"max_abs_dif_ultimo_token_vs_torch": peor, "coseno_min_ultimo_token": cmin}
    (sal / "qwen3emb-onnx.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
