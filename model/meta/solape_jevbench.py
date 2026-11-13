"""Solape entre el train de tsjev (filas ya convertidas) y los textos de JevBench: duplicados exactos y casi duplicados.

    python -m model.meta.solape_jevbench --tsjev DIR --jevbench RUTA --salida DIR_SAL [--split train] [--n-gramas 13]

Lee en streaming (una fila a la vez, sin cargar el conjunto) los `.jsonl`/`.jsonl.gz` de `--tsjev` (salida de `model.meta.convertir`, una
fila por línea con `texto`, `split`, `id`) y los compara con el `texto` de cada caso de JevBench (`model.evaluacion.jevbench`).
Normalización: minúsculas, todo lo que no sea letra o dígito (Unicode) pasa a espacio y los espacios se colapsan.
  · exactos: el texto normalizado de la fila de train es idéntico al de un caso de JevBench;
  · casi duplicados: comparten al menos un shingle de N palabras (13 por defecto) con el texto normalizado de un caso.
Los casos de JevBench con menos de N palabras no tienen shingles: solo pueden dar exacto, y se cuentan aparte (`sin_shingles`).
Escribe `DIR_SAL/solape.json` y `solape.md` con cuentas e ids de los casos de JevBench afectados y de una fila de train de ejemplo por caso
(no se copia ningún texto). Solo biblioteca estándar.
"""
import argparse
import json
import re
import sys
from pathlib import Path

N_GRAMAS = 13
_NO_ALFANUM = re.compile(r"[\W_]+", re.UNICODE)


def normalizar(texto):
    """Minúsculas, sin puntuación (todo lo no alfanumérico es un espacio) y espacios colapsados."""
    return _NO_ALFANUM.sub(" ", str(texto).lower()).strip()


def shingles(normalizado, n=N_GRAMAS):
    """Conjunto de hashes de los n-gramas de palabras de un texto ya normalizado (vacío si tiene menos de n palabras)."""
    p = normalizado.split()
    return {hash(" ".join(p[i:i + n])) for i in range(len(p) - n + 1)}


class Indice:
    """Textos de JevBench indexados por texto normalizado completo y por shingle."""

    def __init__(self, filas, n=N_GRAMAS):
        self.n = n
        self.exacto = {}      # texto normalizado -> [ids]
        self.shingle = {}     # hash del n-grama -> {ids}
        self.sin_shingles = []
        for r in filas:
            t = normalizar(r["texto"])
            self.exacto.setdefault(t, []).append(r["id"])
            s = shingles(t, n)
            if not s:
                self.sin_shingles.append(r["id"])
            for h in s:
                self.shingle.setdefault(h, set()).add(r["id"])
        self.ids = [r["id"] for r in filas]

    def comparar(self, texto):
        """(ids con texto idéntico, {id: n.º de shingles compartidos}) de una fila de train."""
        t = normalizar(texto)
        comunes = {}
        for h in shingles(t, self.n):
            for i in self.shingle.get(h, ()):
                comunes[i] = comunes.get(i, 0) + 1
        return self.exacto.get(t, []), comunes


def filas_train(ruta, split="train"):
    """Streaming de las filas del split `split` de los ficheros de `ruta` (fichero o directorio, jsonl o jsonl.gz)."""
    from model.l2.datos import abrir, ficheros
    for f in ficheros(str(ruta)):
        with abrir(f) as fh:
            for linea in fh:
                if not linea.strip():
                    continue
                fila = json.loads(linea)
                if split is None or fila.get("split") == split:
                    yield fila


def solape(filas_jev, train, n=N_GRAMAS):
    """Cuenta el solape de las filas de `train` (iterable) con `filas_jev`."""
    ix = Indice(filas_jev, n)
    exactos, casi, vistas = {}, {}, 0
    for fila in train:
        vistas += 1
        ex, comunes = ix.comparar(fila["texto"])
        for i in ex:
            d = exactos.setdefault(i, {"filas_train": 0, "ejemplo": fila.get("id")})
            d["filas_train"] += 1
        for i, c in comunes.items():
            d = casi.setdefault(i, {"filas_train": 0, "max_shingles": 0, "ejemplo": fila.get("id")})
            d["filas_train"] += 1
            if c > d["max_shingles"]:
                d["max_shingles"], d["ejemplo"] = c, fila.get("id")
    return {
        "n_gramas": n, "filas_train": vistas, "casos_jevbench": len(ix.ids),
        "exactos": {"casos": len(exactos), "ids": sorted(exactos), "detalle": {i: exactos[i] for i in sorted(exactos)}},
        "casi_duplicados": {"casos": len(casi), "ids": sorted(casi), "detalle": {i: casi[i] for i in sorted(casi)}},
        "sin_shingles": {"casos": len(ix.sin_shingles), "ids": sorted(ix.sin_shingles)},
        "cualquiera": {"casos": len(set(exactos) | set(casi))},
    }


def informe_md(r, split):
    L = ["# Solape tsjev (%s) con JevBench" % split, "",
         "%d filas de train leídas; %d casos de JevBench. Normalización: minúsculas, sin puntuación, espacios colapsados." % (r["filas_train"], r["casos_jevbench"]), "",
         "| medida | casos de JevBench |", "|---|---|",
         "| texto exacto | %d |" % r["exactos"]["casos"],
         "| comparte algún %d-grama de palabras | %d |" % (r["n_gramas"], r["casi_duplicados"]["casos"]),
         "| cualquiera de los dos | %d |" % r["cualquiera"]["casos"],
         "| sin %d-gramas (menos de %d palabras; solo exacto aplica) | %d |" % (r["n_gramas"], r["n_gramas"], r["sin_shingles"]["casos"]), ""]
    for titulo, clave in (("Exactos", "exactos"), ("Casi duplicados", "casi_duplicados")):
        L += ["## %s" % titulo, "", ", ".join(r[clave]["ids"]) or "ninguno", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tsjev", required=True, help="directorio (o fichero) con las filas convertidas de tsjev")
    ap.add_argument("--jevbench", required=True, help="clon del repo, datasets/public o un .jsonl")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--split", default="train", help="split de las filas a comparar (por defecto train; «todos» = todas)")
    ap.add_argument("--n-gramas", type=int, default=N_GRAMAS)
    a = ap.parse_args(argv)
    from model.evaluacion.jevbench import cargar_jevbench
    filas = cargar_jevbench(a.jevbench)
    r = solape(filas, filas_train(a.tsjev, None if a.split == "todos" else a.split), a.n_gramas)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "solape.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    md = informe_md(r, a.split)
    (sal / "solape.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
