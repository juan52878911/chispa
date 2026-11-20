"""Casi-duplicados de la batería v3 contra v2 y contra el entrenamiento, SIN texto de referencia: se compara con huellas.

Huella de un texto (una línea JSON por texto, sin el texto): `h` = sha256 del texto normalizado (minúsculas, espacios colapsados) y
`s` = sketch de los K menores hashes de sus n-gramas de 5 palabras (bottom-k). Jaccard estimado = fracción de los K menores hashes de la unión
que están en los dos. Con textos de menos de K n-gramas el sketch es el conjunto entero y el Jaccard es exacto.

    python -m model.datos.bateria_v3_solapes huellas --entrada DIR_O_FICHERO... --salida huellas-X.jsonl.gz --fuente NOMBRE
Umbrales (a documentar en el manifiesto): `h` igual -> descartar; Jaccard >= 0,8 -> descartar; 0,5 <= Jaccard < 0,8 -> marcar.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

N_GRAMA = 5
K_SKETCH = 32
UMBRAL_DESCARTE = 0.8
UMBRAL_MARCA = 0.5
CAMPOS_TEXTO = ("texto", "text", "sentence", "query", "utterance")


def normalizar(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def _h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def sketch(t, k=K_SKETCH, n=N_GRAMA):
    pal = re.findall(r"\w+", normalizar(t))
    if not pal:
        return []
    grams = {" ".join(pal[i:i + n]) for i in range(max(1, len(pal) - n + 1))}
    return sorted({_h64(g) for g in grams})[:k]


def huella(t):
    return {"h": hashlib.sha256(normalizar(t).encode("utf-8")).hexdigest(), "s": sketch(t)}


def jaccard(a, b, k=K_SKETCH):
    """Jaccard estimado entre dos sketches bottom-k (ya ordenados)."""
    if not a or not b:
        return 0.0
    union = sorted(set(a) | set(b))[:k]
    sa, sb = set(a), set(b)
    return sum(1 for x in union if x in sa and x in sb) / float(len(union))


class Referencia:
    """Índice de las filas de v3 (pocas decenas de miles); las huellas de referencia (millones) se leen en flujo y no se guardan."""

    def __init__(self):
        self.filas = []                       # (tarea, id, h, sketch)
        self.por_h = defaultdict(list)
        self.por_s = defaultdict(list)
        self.res = {}                         # indice de fila -> dict(fuente, tipo, jaccard)

    def añadir(self, tarea, ident, texto):
        hu = huella(texto)
        i = len(self.filas)
        self.filas.append((tarea, ident, hu["h"], hu["s"]))
        self.por_h[hu["h"]].append(i)
        for v in hu["s"]:
            self.por_s[v].append(i)
        return i

    def _anotar(self, i, fuente, tipo, j):
        a = self.res.get(i)
        if a is None or (tipo, j) > (a["tipo"], a["jaccard"]):
            self.res[i] = dict(fuente=fuente, tipo=tipo, jaccard=round(j, 4))

    def comparar(self, huellas, fuente):
        """huellas: iterable de dicts {h, s}. Anota en `res` la peor coincidencia de cada fila (exacto > casi > marca)."""
        n = 0
        for hu in huellas:
            n += 1
            for i in self.por_h.get(hu["h"], ()):
                self._anotar(i, fuente, "exacto", 1.0)
            cand = set()
            for v in hu["s"]:
                cand.update(self.por_s.get(v, ()))
            for i in cand:
                j = jaccard(self.filas[i][3], hu["s"])
                if j >= UMBRAL_DESCARTE:
                    self._anotar(i, fuente, "casi", j)
                elif j >= UMBRAL_MARCA:
                    self._anotar(i, fuente, "marca", j)
        return n

    def veredicto(self, i):
        a = self.res.get(i)
        if not a:
            return "limpio"
        return "descartar" if a["tipo"] in ("exacto", "casi") else "marcar"


def leer_huellas(ruta):
    ab = gzip.open if str(ruta).endswith(".gz") else open
    with ab(ruta, "rt", encoding="utf-8") as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                if set(d) - {"h", "s", "f"}:
                    raise ValueError("%s: una huella trae campos distintos de h, s, f (¿texto en claro?)" % ruta)
                yield d


def textos_de(ruta, sin_texto=None):
    """Textos de un .jsonl(.gz) con el campo de texto de la batería v1/v2 o de los conjuntos de entrenamiento. `sin_texto` (lista de 1 contador) cuenta las filas sin campo de texto."""
    ab = gzip.open if str(ruta).endswith(".gz") else open
    with ab(ruta, "rt", encoding="utf-8") as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                for c in CAMPOS_TEXTO:
                    if isinstance(d.get(c), str):
                        yield d[c]
                        break
                else:
                    if sin_texto is not None:
                        sin_texto[0] += 1


def escribir_huellas(entradas, salida, fuente):
    """Escribe huellas (sin texto) de todos los .jsonl/.jsonl.gz de `entradas`. Devuelve (huellas, ficheros, filas sin campo de texto)."""
    ficheros = []
    for e in entradas:
        e = Path(e)
        ficheros += sorted(p for p in e.rglob("*") if p.name.endswith((".jsonl", ".jsonl.gz"))) if e.is_dir() else [e]
    n, sin = 0, [0]
    with gzip.open(salida, "wt", encoding="utf-8") as out:
        for f in ficheros:
            for t in textos_de(f, sin):
                if t.strip():
                    d = huella(t)
                    d["f"] = fuente
                    out.write(json.dumps(d) + "\n")
                    n += 1
    return n, len(ficheros), sin[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("huellas")
    a.add_argument("--entrada", nargs="+", required=True)
    a.add_argument("--salida", required=True)
    a.add_argument("--fuente", required=True)
    x = ap.parse_args(argv)
    n, nf, sin = escribir_huellas(x.entrada, x.salida, x.fuente)
    print("huellas=%d ficheros=%d filas_sin_campo_de_texto=%d salida=%s" % (n, nf, sin, x.salida))
    return 1 if (not n or sin > 0.01 * (n + sin)) else 0


if __name__ == "__main__":
    sys.exit(main())
