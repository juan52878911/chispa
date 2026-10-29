"""Referencia de transformers para la paridad del codificador Qwen3-Embedding en Go: vectores de la última posición normalizados (fp32, CPU) del fixture.

    python model/alt/qwen3emb_referencia.py --modelo DIR --fixture runtime/tok/testdata/qwen/fixture_paridad.jsonl --salida DIR_SALIDA [--lote 16] [--max-len 320]

Hace lo que `EmbQwen` de alternativas_r11.py: AutoTokenizer(padding_side="left"), truncation, max_length 320, lotes con relleno a la izquierda,
AutoModel en fp32 y `last_hidden_state[:, -1]` normalizado. Escribe en DIR_SALIDA:
  vectores.npy      (n, d) float32, en el orden del fixture
  ids.jsonl         {id, entrada, ids, largo_sin_truncar} (los de runtime/tok/referencia_bpe.py, con el mismo AutoTokenizer)
  referencia.json   versiones de transformers y torch, revisión, n, d y nº de textos truncados
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "tok"))
from referencia_bpe import entrada  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--revision", default="")
    a = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.modelo, padding_side="left", local_files_only=True)
    m = AutoModel.from_pretrained(a.modelo, dtype=torch.float32, local_files_only=True).eval()
    filas = [json.loads(l) for l in open(a.fixture, encoding="utf-8") if l.strip()]
    textos = [entrada(r) for r in filas]
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    out, t0 = [], time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(textos), a.lote):
            b = tk(textos[i:i + a.lote], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
            h = m(**b).last_hidden_state[:, -1]
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).numpy())
    X = np.concatenate(out).astype(np.float32)
    np.save(sal / "vectores.npy", X)
    n_trunc = 0
    with open(sal / "ids.jsonl", "w", encoding="utf-8") as fh:
        for r, t in zip(filas, textos):
            ids = tk(t, truncation=True, max_length=a.max_len)["input_ids"]
            sin = len(tk(t)["input_ids"])
            n_trunc += sin > a.max_len
            fh.write(json.dumps({"id": r["id"], "entrada": t, "ids": ids, "largo_sin_truncar": sin}, ensure_ascii=False) + "\n")
    info = {"transformers": transformers.__version__, "torch": torch.__version__, "revision": a.revision, "n": len(textos), "d": int(X.shape[1]), "truncados": int(n_trunc),
            "segundos": round(time.perf_counter() - t0, 1), "lote": a.lote, "max_len": a.max_len}
    (sal / "referencia.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
