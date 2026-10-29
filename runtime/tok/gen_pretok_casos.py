#!/usr/bin/env python3
"""Casos del pretokenizador de Qwen2 para bpe_test.go: cadenas al azar y su partición con `regex.findall` (módulo `regex`, que sí tiene mirada adelante).
   python runtime/tok/gen_pretok_casos.py runtime/tok/testdata/qwen/pretok_casos.jsonl"""
import json
import random
import sys

import regex

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from exportar_bpe import REGEX_QWEN2  # noqa: E402
from referencia_bpe import ALFABETO  # noqa: E402

if __name__ == "__main__":
    rnd = random.Random(5)
    pat = regex.compile(REGEX_QWEN2)
    alfa = ALFABETO + [" ", " ", " ", "\n", "\n", "'", "s", "t", "a", "b", "1", "!", " ", " ", " ", "　", " ", "ª", "ः", "ͅ"]
    casos = ["", " ", "  ", "a  b", "a \n b", " \n ", "a\n\n b", "'s'S'rE'll'D", "x's y", "   x", "x   ", "x   \n", "1234", "ab12", "!!\n\nx", "a   b", "\r\n\r\nx"]
    casos += ["".join(rnd.choice(alfa) for _ in range(rnd.randint(1, 20))) for _ in range(6000)]
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        for s in casos:
            fh.write(json.dumps([s, pat.findall(s)], ensure_ascii=False) + "\n")
