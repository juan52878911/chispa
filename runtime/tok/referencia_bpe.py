#!/usr/bin/env python3
"""Ids de referencia del tokenizador de Qwen3-Embedding con transformers: AutoTokenizer(padding_side="left")(entrada, truncation=True, max_length=320),
que es lo que hace model/alt/alternativas_r11.py (EmbQwen). Sin torch.

  python runtime/tok/referencia_bpe.py DIR_MODELO FIXTURE.jsonl SALIDA.jsonl [--max-len 320] [--fuzz N]

FIXTURE.jsonl: filas {id, tipo, pregunta, texto} (testdata/qwen/fixture_paridad.jsonl). La entrada de «consulta» es
`Instruct: Classify the text to answer this question: <pregunta>\\nQuery:<texto>` y la de «descripcion» el texto a secas.
SALIDA: {id, entrada, ids, largo_sin_truncar}. Con --fuzz N añade N cadenas al azar (semilla fija) de caracteres de borde del regex.
"""
import argparse
import json
import random

INSTR = "Classify the text to answer this question: %s"


def entrada(r):
    return "Instruct: %s\nQuery:%s" % (INSTR % r["pregunta"], r["texto"]) if r["tipo"] == "consulta" else r["texto"]


ALFABETO = list("aAbZ09 \t\n\r'sStTrReEvVmMlLdD.,;:!?-_/\\()[]{}<>|\"@#$%&*+=~^`") + ["é", "ñ", "Ü", "é", "́", "ß", "ǅ", "٣", "²", "½", "Ⅻ", "中", "日", "한", " ", "　", " ",
                                                                                 "​", "‍", "😀", "👍🏽", "️", "\u0085", "\x0b", "\x0c", "\x00", "<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|", "|>", "�"]


def fuzz(n, semilla=11):
    rnd = random.Random(semilla)
    return [{"id": "f%04d" % i, "entrada": "".join(rnd.choice(ALFABETO) for _ in range(rnd.randint(0, 24)))} for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("modelo")
    ap.add_argument("fixture")
    ap.add_argument("salida")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--fuzz", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.modelo, padding_side="left", local_files_only=True)
    filas = [{"id": r["id"], "entrada": entrada(r)} for r in map(json.loads, open(a.fixture, encoding="utf-8")) if r]
    filas += fuzz(a.fuzz)
    with open(a.salida, "w", encoding="utf-8") as fh:
        for f in filas:
            ids = tk(f["entrada"], truncation=True, max_length=a.max_len)["input_ids"]
            sin = len(tk(f["entrada"])["input_ids"])
            fh.write(json.dumps({"id": f["id"], "entrada": f["entrada"], "ids": ids, "largo_sin_truncar": sin}, ensure_ascii=False) + "\n")
    print("ids de %d textos (transformers %s, max_len %d)" % (len(filas), __import__("transformers").__version__, a.max_len))


if __name__ == "__main__":
    main()
