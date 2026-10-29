#!/usr/bin/env python3
"""Casos de NFC para nfc_test.go: cadenas al azar (semilla fija) de caracteres con descomposición, marcas combinantes, jamo y texto normal,
con la salida de unicodedata.normalize('NFC').   python runtime/tok/gen_nfc_casos.py runtime/tok/testdata/nfc_casos.jsonl"""
import json
import random
import sys
import unicodedata as u


def main(sal):
    rnd = random.Random(7)
    desc = [c for c in range(0x110000) if not 0xD800 <= c < 0xE000 and u.decomposition(chr(c)) and not u.decomposition(chr(c)).startswith("<")]
    comb = [c for c in range(0x300, 0x370)] + [c for c in range(0x110000) if not 0xD800 <= c < 0xE000 and u.combining(chr(c))]
    jamo = list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176)) + list(range(0x11A8, 0x11C3)) + [0xAC00, 0xAC01, 0xD7A3]
    base = [ord(x) for x in "aeiounAEIOUNcsz ,.\n"] + [0xF1, 0xE9, 0x1F600, 0x200D, 0x3042]
    casos = ["", "hola", "café", "ñ", "ẹ́", "ẹ́", "각", "̈́", "Å", "ẛ̣", "á́",
             "ཱི", "Ω", "Å", "ñ" * 5]
    for _ in range(4000):
        k = rnd.randint(1, 12)
        pool = rnd.choice([desc, comb, jamo, base, base + comb, base + desc + comb + jamo])
        casos.append("".join(chr(rnd.choice(pool)) for _ in range(k)))
    with open(sal, "w", encoding="utf-8") as fh:
        for s in casos:
            fh.write(json.dumps([s, u.normalize("NFC", s)], ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
