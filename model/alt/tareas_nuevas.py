"""R3: tres tareas en español que la batería no tiene, en el formato de la batería no vista. SOLO PARA MEDIR, nunca para entrenar.

    python -m model.alt.tareas_nuevas --hub DIR_HF_HUB --eurovoc eurovoc_descriptors.json --salida DIR [--tope 2000] [--k 8]

Fuentes (snapshot_download a --hub en la revisión fijada; revisión = sha de la instantánea):
  lexboe       BSC-LT/LexBOE @ d7263e2f (CC BY 4.0; reutilización del BOE con atribución)    tema legal-administrativo
  eurlex-n1    coastalcph/multi_eurlex @ d0318b7f, es (CC BY 4.0; EUR-Lex, © UE)            EUROVOC nivel 1, SOLO documentos con un
               único descriptor de nivel 1; fuera las clases con menos de MIN_CLASE documentos en test. Nombres de las clases: los
               descriptores en español de EuroVoc (© UE), del fichero `eurovoc_descriptors.json` de nlpaueb/multi-eurlex @ 078092d6.
  fakenews-es  mariagrandury/fake_news_corpus_spanish @ d12e4174 (CC BY 4.0)                 noticia falsa; SOLO zero-shot: sin soporte
Mismas reglas que `bateria_no_vista.construir`: soporte de k por clase sacado del split train (al azar, semilla fija), evaluación
del split test sin textos del soporte, barajada y con tope. El texto se corta a TOPE_CAR caracteres (los documentos de EUR-Lex
superan el contexto de 16 k del LLM); el corte va al manifiesto. Las etiquetas son las de las fuentes.
"""
import argparse
import csv
import json
import random
import sys
import tarfile
from collections import Counter
from pathlib import Path

from model.alt.bateria_no_vista import snapshot
from model.datos import comun

TOPE_CAR = 3000
MIN_CLASE = 20
ATRIBUCION = {
    "lexboe": "Fuente de los datos: Agencia Estatal Boletín Oficial del Estado (BOE), vía BSC-LT/LexBOE (CC BY 4.0). Datos modificados: convertidos al formato de la batería y cortados a 3 000 caracteres.",
    "eurlex-n1": "© Unión Europea, EUR-Lex (Decisión 2011/833/UE), vía coastalcph/multi_eurlex (CC BY 4.0); descriptores EuroVoc © Unión Europea. Datos modificados: solo documentos de un descriptor de nivel 1, cortados a 3 000 caracteres.",
    "fakenews-es": "Posadas-Durán et al., FakeNewsCorpusSpanish (CC BY 4.0), vía mariagrandury/fake_news_corpus_spanish. Datos modificados: titular y texto unidos y cortados a 3 000 caracteres.",
}
DESC_LEXBOE = {
    "Funcionarios y Personal": "empleo público y función pública: oposiciones, nombramientos, plazas", "Normativas": "leyes y reglamentos",
    "Administración": "organización y procedimientos de la administración", "Educación": "sistema educativo, centros, universidades y políticas",
    "Economía": "asuntos económicos y financieros", "Energía": "política e infraestructuras de energía", "Judicial": "tribunales, procesos y órganos judiciales",
    "Cultura": "instituciones y actividades culturales", "Salud": "salud pública y sanidad", "Transporte": "sistemas e infraestructuras de transporte",
    "Fuerzas": "fuerzas de seguridad y defensa", "Vivienda": "vivienda y urbanismo",
}
DESC_FAKE = {"false": "noticia verdadera, publicada por un medio fiable", "true": "noticia falsa o desinformación"}


def corta(t):
    t = (t or "").strip()
    return t if len(t) <= TOPE_CAR else t[:TOPE_CAR]


def lexboe(hub):
    raiz = snapshot(hub, "BSC-LT/LexBOE")
    filas = {s: comun.leer_tabla(raiz / (s + ".jsonl")) for s in ("train", "dev", "test")}
    clases = sorted({r["label"] for v in filas.values() for r in v})
    desc = {c: DESC_LEXBOE.get(c, c) for c in clases}
    items = [(s, dict(tarea="tema_legal", tipo="choice", pregunta="¿De qué materia trata este texto del Boletín Oficial del Estado?", texto=corta(r["sentence"]),
                      opciones=clases, descripciones=desc, etiqueta=r["label"], idioma="es", fuente="BSC-LT/LexBOE " + s, licencia="cc-by-4.0", uso_comercial=True))
             for s, v in filas.items() for r in v]
    f, _ = comun.numerar("lexboe", items)
    return f, dict(repo="BSC-LT/LexBOE", rev=raiz.name, licencia="cc-by-4.0", idioma="es", clases_en_datos=len(clases),
                   clases_sin_descripcion=[c for c in clases if c not in DESC_LEXBOE])


def eurlex(hub, eurovoc):
    raiz = snapshot(hub, "coastalcph/multi_eurlex")
    nombres = {k: v["es"] for k, v in json.loads(Path(eurovoc).read_text(encoding="utf-8")).items()}
    docs, total = {}, Counter()
    with tarfile.open(raiz / "data" / "multi_eurlex.tar.gz") as tar:
        for m in tar:
            nombre = Path(m.name).name
            if nombre not in ("train.jsonl", "dev.jsonl", "test.jsonl"):
                continue
            s = nombre.split(".")[0]
            docs[s] = []
            for l in tar.extractfile(m):
                d = json.loads(l)
                t, n1 = d["text"].get("es"), d["eurovoc_concepts"]["level_1"]
                if t is None:
                    continue
                total[s] += 1
                if len(n1) == 1:
                    docs[s].append((d["celex_id"], t, n1[0]))
    en_test = Counter(c for _, _, c in docs["test"])
    vale = sorted(c for c, n in en_test.items() if n >= MIN_CLASE)
    clases = sorted(nombres[c] for c in vale)
    desc = {nombres[c]: nombres[c] for c in vale}
    items = [(s, dict(tarea="tema_eurovoc", tipo="choice", pregunta="¿De qué ámbito temático (EuroVoc) es este texto legislativo de la Unión Europea?",
                      texto=corta(t), opciones=clases, descripciones=desc, etiqueta=nombres[c], idioma="es", fuente="coastalcph/multi_eurlex es %s %s" % (s, cid),
                      licencia="cc-by-4.0", uso_comercial=True))
             for s in ("train", "dev", "test") for cid, t, c in docs[s] if c in vale]
    f, _ = comun.numerar("eurlex-n1", items)
    meta = dict(repo="coastalcph/multi_eurlex", rev=raiz.name, licencia="cc-by-4.0", idioma="es", eurovoc="nlpaueb/multi-eurlex@078092d62633ab1fc85065cc5937e60edd805095",
                docs_es=dict(total), docs_un_descriptor={s: len(v) for s, v in docs.items()},
                test_por_clase={nombres.get(c, c): n for c, n in sorted(en_test.items(), key=lambda x: -x[1])},
                train_por_clase=dict(Counter(nombres.get(c, c) for _, _, c in docs["train"])))
    meta["clases_descartadas_menos_de_%d_en_test" % MIN_CLASE] = {nombres.get(c, c): n for c, n in en_test.items() if n < MIN_CLASE}
    return f, meta


def fakenews(hub):
    raiz = snapshot(hub, "mariagrandury/fake_news_corpus_spanish")
    filas = list(csv.DictReader(open(raiz / "test.csv", encoding="utf-8")))
    items = [("test", dict(tarea="noticia_falsa", tipo="noul", pregunta="¿La noticia es falsa (desinformación)?", texto=corta(r["HEADLINE"] + "\n" + r["TEXT"]),
                           opciones=["false", "true"], descripciones=DESC_FAKE, etiqueta="true" if r["CATEGORY"].strip().upper() == "FALSE" else "false",
                           idioma="es", fuente="mariagrandury/fake_news_corpus_spanish test " + r["ID"], licencia="cc-by-4.0", uso_comercial=True)) for r in filas]
    f, _ = comun.numerar("fakenews-es", items)
    return f, dict(repo="mariagrandury/fake_news_corpus_spanish", rev=raiz.name, licencia="cc-by-4.0", idioma="es", solo_zero_shot=True,
                   temas=dict(Counter(r["TOPICS"] for r in filas)))


def construir(hub, eurovoc, tope=2000, k=8, semilla=0):
    por, meta = {}, {}
    for corto, (filas, m) in (("lexboe", lexboe(hub)), ("eurlex-n1", eurlex(hub, eurovoc)), ("fakenews-es", fakenews(hub))):
        kk = 0 if corto == "fakenews-es" else k
        rng = random.Random("%s-%d" % (corto, semilla))
        sop = []
        for et in sorted({r["etiqueta"] for r in filas if r["split"] == "train"}):
            c = [r for r in filas if r["split"] == "train" and r["etiqueta"] == et]
            rng.shuffle(c)
            sop += c[:kk]
        textos_sop = {r["texto"] for r in sop}
        ev = [r for r in filas if r["split"] == "test" and r["texto"] not in textos_sop]
        rng.shuffle(ev)
        ev = ev[:tope]
        for r in sop:
            r["split"] = "train"
        por[corto] = {"train": sop, "test": ev}
        meta[corto] = dict(m, n_eval=len(ev), n_soporte=len(sop), k=kk, tope=tope, tope_caracteres=TOPE_CAR, cortados_eval=sum(1 for r in ev if len(r["texto"]) >= TOPE_CAR),
                           clases=len({r["etiqueta"] for r in ev}), eval_por_clase=dict(Counter(r["etiqueta"] for r in ev)), atribucion=ATRIBUCION[corto],
                           uso="solo medir, nunca entrenar")
    return por, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hub", required=True)
    ap.add_argument("--eurovoc", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--tope", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args(argv)
    por, meta = construir(a.hub, a.eurovoc, a.tope, a.k)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    for corto, d in por.items():
        with open(sal / (corto + ".jsonl"), "w", encoding="utf-8") as fh:
            for r in d["train"] + d["test"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (sal / "manifiesto.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({c: {k: v for k, v in m.items() if k in ("n_eval", "n_soporte", "clases", "cortados_eval", "docs_un_descriptor")} for c, m in meta.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
