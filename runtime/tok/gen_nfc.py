#!/usr/bin/env python3
"""Genera nfc_tablas.go (tablas de NFC) desde unicodedata de Python. Determinista para una misma versión de Unicode.

  python runtime/tok/gen_nfc.py runtime/tok/nfc_tablas.go

Contiene la descomposición canónica completa (sin Hangul, que es algorítmico), la clase de combinación (ccc) y las composiciones primarias
(pares que componen y no están excluidos). Los tests (nfc_test.go) comparan con testdata/nfc_casos.jsonl, que genera gen_nfc_casos.py
con unicodedata.normalize('NFC').
"""
import sys
import unicodedata as u

HANGUL = range(0xAC00, 0xD7A4)


def full(c):
    d = u.decomposition(chr(c))
    if not d or d.startswith("<"):
        return [c]
    out = []
    for h in d.split():
        out += full(int(h, 16))
    return out


def main(sal):
    dec, comp, ccc = {}, {}, {}
    for c in range(0x110000):
        if 0xD800 <= c < 0xE000 or c in HANGUL:
            continue
        ch = chr(c)
        k = u.combining(ch)
        if k:
            ccc[c] = k
        d = u.decomposition(ch)
        if d and not d.startswith("<"):
            dec[c] = full(c)
            par = [int(h, 16) for h in d.split()]
            if len(par) == 2 and u.normalize("NFC", ch) == ch:
                comp[(par[0], par[1])] = c
    rangos = []
    for c in sorted(ccc):
        if rangos and rangos[-1][1] + 1 == c and rangos[-1][2] == ccc[c]:
            rangos[-1][1] = c
        else:
            rangos.append([c, c, ccc[c]])
    L = ["// Código generado por gen_nfc.py (Unicode %s). No editar." % u.unidata_version, "", "package tok", "",
         "const versionUnicodeNFC = \"%s\"" % u.unidata_version, "",
         "// nfcDescomp: descomposición canónica completa (sin Hangul).", "var nfcDescomp = map[rune][]rune{"]
    L += ["\t0x%X: {%s}," % (c, ", ".join("0x%X" % x for x in dec[c])) for c in sorted(dec)]
    L += ["}", "", "// nfcComp: composición primaria (primero, segundo) → compuesto.", "var nfcComp = map[[2]rune]rune{"]
    L += ["\t{0x%X, 0x%X}: 0x%X," % (a, b, comp[(a, b)]) for a, b in sorted(comp)]
    L += ["}", "", "// nfcCCC: clases de combinación distintas de 0, por rangos {desde, hasta, clase}.", "var nfcCCC = [][3]int32{"]
    L += ["\t{0x%X, 0x%X, %d}," % tuple(r) for r in rangos]
    L += ["}", ""]
    open(sal, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
