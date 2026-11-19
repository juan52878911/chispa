"""Fixtures INVENTADOS de la batería v3: un espejo local de «HF» (CHISPA_V3_ESPEJO), ficheros file:// y un manifiesto con tareas que pasan y que quedan fuera.

    python -m model.datos.tests.fixtures_v3.generar DIR     # deja DIR/{espejo,url,huellas,manifiesto.json}
Todo el texto sale de un vocabulario inventado con semilla fija: nada real.
"""
import gzip
import json
import random
import sys
import zipfile
from pathlib import Path

from model.datos import bateria_v3 as b

PALABRAS = ("alfa beta gamma delta épsilon zeta eta theta iota kappa lambda sigma tau ípsilon fi ji psi omega "
            "mesa río luna cobre nube trigo faro isla puente arena viento lago cerro muelle bruma tinta vela").split()
VOCAB = ["%s%d" % (p, i) for p in PALABRAS for i in range(8)]


def texto(rng, n=30, marca=""):
    return " ".join(([marca] if marca else []) + [rng.choice(VOCAB) for _ in range(n)])


def _readme(licencia, cuerpo=""):
    cab = "---\nlicense: %s\n---\n" % licencia if licencia else "---\ntask_categories:\n- text-classification\n---\n"
    return cab + "# Ficha inventada\n" + cuerpo + "\n"


def _jsonl(ruta, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas), encoding="utf-8")


def cand(nombre, tipo, lector, fuente, **kw):
    c = dict(nombre=nombre, tipo=tipo, idioma="es", dominio="inventado", familia=nombre, fuente=fuente, url_ficha="https://example.invalid/" + nombre,
             licencia_esperada="cc-by-4.0", sintetico=True, clases=3, confianza_id="alta", limpieza_esperada="alta", no_solapa="inventada", riesgos="ninguno",
             lector=lector, pregunta="¿Categoría?")
    c.update(kw)
    return c


def generar(destino):
    d = Path(destino)
    rng = random.Random(7)
    esp, url, hue = d / "espejo" / "hf", d / "url", d / "huellas"
    cands = []
    # 1. choice jsonl con split oficial; duplicados exactos y conflicto dentro, y un texto repetido en el test de la referencia
    filas = {"train": [], "test": []}
    for sp, n in (("train", 30), ("test", 70)):
        for c in ("a", "b", "c"):
            for _ in range(n):
                filas[sp].append({"texto": texto(rng, marca="clase" + c), "etiqueta": "cat-" + c})
    filas["test"].append(dict(filas["test"][0]))                       # duplicado exacto
    filas["test"].append(dict(filas["test"][1], etiqueta="cat-b" if filas["test"][1]["etiqueta"] != "cat-b" else "cat-a"))   # mismo texto, otra etiqueta
    for sp in filas:
        _jsonl(esp / "ej__choice" / ("%s.jsonl" % sp), filas[sp])
    (esp / "ej__choice" / "README.md").write_text(_readme("cc-by-4.0"), encoding="utf-8")
    cands.append(cand("mini-choice", "choice", dict(formato="jsonl", ficheros=["*.jsonl"], texto="texto", etiqueta="etiqueta", split_por_fichero=dict(train="train", test="test")),
                      dict(tipo="hf", repo="ej/choice", patrones=["*.jsonl", "README.md"])))
    ref_exacto, ref_casi = filas["test"][2]["texto"], filas["test"][3]["texto"]
    # 2. yes/no por pares sin split oficial (partición propia con semilla), licencia MIT
    pares = [{"bueno": texto(rng, 14, "ok"), "malo": texto(rng, 14, "ko"), "ctx": texto(rng, 5)} for _ in range(160)]
    _jsonl(esp / "ej__pares" / "datos.jsonl", pares)
    (esp / "ej__pares" / "README.md").write_text(_readme("mit"), encoding="utf-8")
    cands.append(cand("mini-yesno", "yesno", dict(formato="jsonl", ficheros=["datos.jsonl"], pares=dict(bueno="bueno", malo="malo", contexto=["ctx"])),
                      dict(tipo="hf", repo="ej/pares", patrones=["*.jsonl", "README.md"]), clases=2, licencia_esperada="mit"))
    # 3. score con nota continua en csv -> 5 cuantiles
    import csv
    (esp / "ej__score").mkdir(parents=True)
    with open(esp / "ej__score" / "notas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tema", "argumento", "nota"])
        for i in range(400):
            w.writerow(["tema%d" % (i % 5), texto(rng, 20), round(rng.random(), 4)])
    (esp / "ej__score" / "README.md").write_text(_readme("cc0-1.0"), encoding="utf-8")
    cands.append(cand("mini-score", "score", dict(formato="csv", ficheros=["*.csv"], texto=["tema", "argumento"], etiqueta="nota", cuantiles=5),
                      dict(tipo="hf", repo="ej/score", patrones=["*.csv", "README.md"]), clases=5, licencia_esperada="cc0-1.0"))
    # 4. parquet (lector de pyarrow) con etiqueta_par y filtro
    import pyarrow as pa
    import pyarrow.parquet as pq
    (esp / "ej__par").mkdir(parents=True)
    n = 300
    tabla = pa.table({"text": [texto(rng, 12) for _ in range(n)], "lang": ["es" if i % 3 else "en" for i in range(n)],
                      "labels": [{"name": ["spam", "quality"], "value": [0.0, (i % 100) / 100.0]} for i in range(n)]})
    pq.write_table(tabla, str(esp / "ej__par" / "train.parquet"))
    (esp / "ej__par" / "README.md").write_text(_readme("apache-2.0"), encoding="utf-8")
    cands.append(cand("mini-parquet", "score", dict(formato="parquet", ficheros=["*.parquet"], texto="text", etiqueta_par=dict(col="labels", nombre="quality"), cuantiles=4,
                                                    filtro=dict(lang=["es"])), dict(tipo="hf", repo="ej/par", patrones=["*.parquet", "README.md"]), clases=4,
                      licencia_esperada="apache-2.0"))
    # 5-8. fuera por licencia
    for nom, rd, lic_esp in (("mini-nc", _readme("cc-by-nc-4.0"), "cc-by-4.0"), ("mini-sa", _readme("cc-by-4.0", "Se publica bajo CC BY-SA 4.0 por herencia."), "cc-by-4.0"),
                             ("mini-sinlic", _readme(None), "mit"), ("mini-tuits", _readme("mit", "Los textos son tuits recogidos de Twitter."), "mit")):
        carp = esp / ("ej__" + nom)
        _jsonl(carp / "d.jsonl", [{"texto": texto(rng), "etiqueta": "x"} for _ in range(5)])
        (carp / "README.md").write_text(rd, encoding="utf-8")
        cands.append(cand(nom, "choice", dict(formato="jsonl", ficheros=["*.jsonl"], texto="texto", etiqueta="etiqueta"), dict(tipo="hf", repo="ej/" + nom, patrones=["*.jsonl", "README.md"]),
                          licencia_esperada=lic_esp))
    # 9. fuente por URL (file://) con ficha de dominio público, csv sin cabecera en latin-1 y prefijo de etiqueta
    url.mkdir(parents=True)
    with open(url / "quejas.tsv", "w", encoding="latin-1", newline="") as fh:
        for i in range(260):
            comp = ("FRENOS" if i % 2 else "MOTOR") + ":detalle %d" % (i % 4)
            fh.write("\t".join(["id%d" % i, comp, texto(rng, 15, "queja")]) + "\n")
    (url / "ficha.txt").write_text("Los datos son de dominio público (obra del gobierno, inventada).", encoding="utf-8")
    cands.append(cand("mini-url", "choice", dict(formato="tsv", ficheros=["quejas.tsv"], columnas={"componente": 1, "texto": 2}, texto="texto", etiqueta="componente", prefijo_hasta=":",
                                                 codificacion="latin-1"),
                      dict(tipo="url", url=(url / "quejas.tsv").as_uri(), max_gb=1), url_ficha=(url / "ficha.txt").as_uri(), licencia_esperada="dominio-publico",
                      licencia_regex_ficha=r"dominio p[uú]blico", clases=2))
    # 10. zip de JSON con la licencia dentro
    with zipfile.ZipFile(url / "avisos.zip", "w") as z:
        z.writestr("repo-main/LICENSE.md", "Creative Commons Attribution 4.0 International (CC BY 4.0), texto inventado")
        for i in range(120):
            z.writestr("repo-main/avisos/a%d.json" % i, json.dumps({"summary": texto(rng, 8), "details": texto(rng, 10),
                                                                    "database_specific": {"severity": ["LOW", "MODERATE", "HIGH"][i % 3]}}))
    (url / "ficha2.txt").write_text("repo inventado", encoding="utf-8")
    cands.append(cand("mini-zip", "score", dict(formato="json_zip", ficheros=["*/avisos/*.json"], texto=["summary", "details"], etiqueta="database_specific.severity",
                                                mapa_etiqueta=dict(LOW="1", MODERATE="2", HIGH="3")),
                      dict(tipo="url", url=(url / "avisos.zip").as_uri(), max_gb=1), url_ficha=(url / "ficha2.txt").as_uri(), licencia_esperada="cc-by-4.0",
                      licencia_fichero_en_descarga="*/LICENSE.md", licencia_regex_ficha=r"CC BY 4\.0", clases=3))
    # 11. pendiente de lector
    cands.append(cand("mini-pendiente", "choice", dict(pendiente="XML por escribir"), dict(tipo="url", url="https://example.invalid/x.xml"), clases=None))
    # manifiesto
    man = b.manifiesto_inicial()
    man["tareas"] = [b.entrada_inicial(c) for c in cands]
    man["descartadas"], man["reservas_v2"] = [], []
    b.guardar_json(d / "manifiesto.json", man)
    # huellas de «referencia»: un texto exacto y un casi-duplicado (cambia la última palabra) de la tarea mini-choice, más relleno ajeno
    hue.mkdir(parents=True)
    from model.datos import bateria_v3_solapes as s
    otros = [texto(random.Random(99 + i), 30) for i in range(50)]
    ref_dir = d / "refs"
    _jsonl(ref_dir / "v2.jsonl", [{"texto": ref_exacto}, {"texto": ref_casi.rsplit(" ", 1)[0] + " palabra-nueva"}] + [{"texto": t} for t in otros])
    s.escribir_huellas([ref_dir], hue / "huellas-v2.jsonl.gz", "v2")
    return dict(manifiesto=d / "manifiesto.json", espejo=d / "espejo", huellas=hue, ref_exacto=ref_exacto, ref_casi=ref_casi)


if __name__ == "__main__":
    print({k: str(v)[:80] for k, v in generar(sys.argv[1]).items()})
