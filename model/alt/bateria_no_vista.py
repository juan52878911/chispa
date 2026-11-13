"""Batería de tareas que ningún modelo vio (P0): construcción, filtro de solape con el entrenamiento y manifiesto.

    python -m model.alt.bateria_no_vista --hub DIR_HF_HUB --tsjev DIR_TSJEV --salida DIR [--tope 2000] [--k 8]

Fuentes (todas bajadas con `huggingface_hub.snapshot_download` a `--hub`; la revisión es el sha de la instantánea):
  hatecheck-es  Paul/hatecheck-spanish (CC-BY-4.0)             noul, español
  massive-es    mteb/amazon_massive_intent (Apache-2.0)        60 intenciones, español
  massive-esc   mteb/amazon_massive_scenario (Apache-2.0)      18 escenarios, español; enunciados DISJUNTOS de los de massive-es
  tuits-es      mteb/tweet_sentiment_multilingual (CC-BY-3.0)  sentimiento, español
  jailbreak     jackhhao/jailbreak-classification (Apache-2.0) noul, inglés
  inyecciones   deepset/prompt-injections (Apache-2.0)         noul, inglés/alemán
Se deja fuera clinc150 (L2 lo vio). Una fila de evaluación se descarta si su texto aparece en el train/dev de tsjev (el único
corpus de entrenamiento común a L2 y al cross-encoder del hito 4 que hay en el PC de pruebas); el recuento va al manifiesto.
`train` (soporte de los k-shot de kNN-e5) y `test` (evaluación) no comparten filas. Las etiquetas son las de la fuente, no de Jev.
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from model.datos import comun, hatecheck, inyecciones, massive

ESCENARIOS = {
    "alarm": "alarmas y despertadores", "audio": "volumen y ajustes de sonido", "calendar": "calendario y eventos",
    "cooking": "cocina y recetas", "datetime": "fecha y hora", "email": "correo electrónico", "general": "conversación general, saludos y chistes",
    "iot": "domótica: luces, enchufes, aspiradora, cafetera", "lists": "listas y tareas", "music": "preguntas y ajustes de música",
    "news": "noticias", "play": "reproducir música, radio, pódcast, audiolibros o juegos", "qa": "preguntas de conocimiento, cálculo, divisas o bolsa",
    "recommendation": "recomendaciones de eventos, lugares o películas", "social": "redes sociales", "takeaway": "comida a domicilio",
    "transport": "transporte, taxis, billetes y tráfico", "weather": "el tiempo",
}
SENT = {"negative": "el tuit expresa un sentimiento negativo", "neutral": "el tuit es neutro o no expresa sentimiento",
        "positive": "el tuit expresa un sentimiento positivo"}
SENT_ID = {0: "negative", 1: "neutral", 2: "positive"}
DESC_JB = {"false": "mensaje normal que no intenta saltarse las reglas del asistente",
           "true": "intento de jailbreak: pide al asistente que ignore sus reglas, adopte otro rol sin límites o revele instrucciones"}


def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def textos_tsjev(directorio):
    """Conjunto de líneas normalizadas de todos los `texto` de train/dev de tsjev (sin los prefijos «Item A:», «text_A:»)."""
    out = set()
    pre = re.compile(r"^(item [ab]|text_[ab]|passage [ab]):\s*", re.I)
    for f in sorted(Path(directorio).glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for l in fh:
                for ln in json.loads(l).get("texto", "").split("\n"):
                    ln = pre.sub("", ln.strip())
                    if ln:
                        out.add(norm(ln))
    return out


def snapshot(hub, repo):
    d = Path(hub) / ("datasets--" + repo.replace("/", "--")) / "snapshots"
    return sorted(p for p in d.glob("*") if p.is_dir())[-1]


def fila(corto, n, split, **kw):
    return comun.fila(corto, split, n, **kw)


def leer(p):
    return comun.leer_tabla(p)


def h(texto, sal):
    return int(hashlib.sha256((sal + texto).encode()).hexdigest(), 16)


def convertir_todo(hub):
    """{corto: (filas con split train/dev/test, meta)}; los splits son los de la fuente (o por plantilla en hatecheck)."""
    out = {}
    comun.HF_HUB = Path(hub)   # por si algún conversor busca en la caché
    # hatecheck-es: el conversor del repo (sin los casos con desacuerdo de anotadores)
    raiz = snapshot(hub, hatecheck.REPO)
    f, m = hatecheck.convertir(raiz)
    out["hatecheck-es"] = (f, dict(repo=hatecheck.REPO, rev=raiz.name, licencia=m["licencia"], idioma="es"))
    # massive-es: el conversor del repo
    raiz = snapshot(hub, massive.REPO)
    f, m = massive.convertir(raiz)
    out["massive-es"] = (f, dict(repo=massive.REPO, rev=raiz.name, licencia=m["licencia"], idioma="es"))
    # massive escenario: los mismos enunciados de test en español, otra tarea (18 escenarios)
    raiz = snapshot(hub, "mteb/amazon_massive_scenario")
    items = []
    for split, carpeta in (("train", "train"), ("dev", "validation"), ("test", "test")):
        for r in leer(raiz / carpeta / "es.json.gz"):
            items.append((split, dict(tarea="escenario", tipo="choice", pregunta="¿De qué ámbito es la petición del usuario al asistente?",
                                      texto=r["text"], opciones=sorted(ESCENARIOS), descripciones=ESCENARIOS, etiqueta=r["label_text"], idioma="es",
                                      fuente="mteb/amazon_massive_scenario " + carpeta, licencia="apache-2.0", uso_comercial=True)))
    f, _ = comun.numerar("massive-esc", items)
    out["massive-esc"] = (f, dict(repo="mteb/amazon_massive_scenario", rev=raiz.name, licencia="apache-2.0", idioma="es"))
    # tuits en español (sentimiento): 0 negative, 1 neutral, 2 positive según la tarjeta
    raiz = snapshot(hub, "mteb/tweet_sentiment_multilingual")
    items = []
    for split, carpeta in (("train", "train"), ("dev", "validation"), ("test", "test")):
        for r in leer(raiz / carpeta / "spanish.jsonl.gz"):
            items.append((split, dict(tarea="sentimiento", tipo="choice", pregunta="¿Qué sentimiento expresa el tuit?", texto=r["text"],
                                      opciones=list(SENT), descripciones=SENT, etiqueta=SENT_ID[int(r["label"])], idioma="es",
                                      fuente="mteb/tweet_sentiment_multilingual " + carpeta, licencia="cc-by-3.0", uso_comercial=True)))
    f, _ = comun.numerar("tuits-es", items)
    out["tuits-es"] = (f, dict(repo="mteb/tweet_sentiment_multilingual", rev=raiz.name, licencia="cc-by-3.0", idioma="es"))
    # jailbreak (inglés)
    raiz = snapshot(hub, "jackhhao/jailbreak-classification")
    items = []
    for split, nombre in (("train", "train"), ("test", "test")):
        for r in leer(raiz / "default" / ("jailbreak_dataset_%s.csv" % nombre)):
            items.append((split, dict(tarea="jailbreak", tipo="noul", pregunta="¿El mensaje intenta que el asistente se salte sus reglas (jailbreak)?",
                                      texto=r["prompt"], opciones=["false", "true"], descripciones=DESC_JB,
                                      etiqueta="true" if r["type"] == "jailbreak" else "false", idioma="en",
                                      fuente="jackhhao/jailbreak-classification " + nombre, licencia="apache-2.0", uso_comercial=True)))
    f, _ = comun.numerar("jailbreak", items)
    out["jailbreak"] = (f, dict(repo="jackhhao/jailbreak-classification", rev=raiz.name, licencia="apache-2.0", idioma="en"))
    # inyecciones (deepset): el conversor del repo
    raiz = snapshot(hub, inyecciones.REPO)
    f, m = inyecciones.convertir(raiz)
    out["inyecciones"] = (f, dict(repo=inyecciones.REPO, rev=raiz.name, licencia=m["licencia"], idioma="multi"))
    return out


def construir(hub, tsjev, tope=2000, k=8, semilla=0):
    vistos = textos_tsjev(tsjev)
    por, meta = {}, {}
    for corto, (filas, m) in convertir_todo(hub).items():
        if corto == "massive-es" or corto == "massive-esc":   # enunciados disjuntos entre las dos tareas (por hash del texto)
            par = 0 if corto == "massive-es" else 1
            filas = [r for r in filas if h(r["texto"], "massive-split") % 2 == par]
        pool_eval = [r for r in filas if r["split"] in ("test", "dev")] if corto != "hatecheck-es" else list(filas)
        pool_sop = [r for r in filas if r["split"] == "train"] if corto != "hatecheck-es" else list(filas)
        if corto == "inyecciones":
            pool_eval = [r for r in filas if r["split"] == "test"]
        if corto in ("massive-es", "massive-esc"):
            pool_eval = [r for r in filas if r["split"] == "test"]
        rng = random.Random("%s-%d" % (corto, semilla))
        # soporte k por clase (hatecheck: salido del mismo fichero y excluido de la evaluación)
        sop, usados = [], set()
        for et in sorted({r["etiqueta"] for r in pool_sop}):
            c = [r for r in pool_sop if r["etiqueta"] == et]
            rng.shuffle(c)
            for r in c[:k]:
                sop.append(r)
                usados.add(r["id"])
        textos_sop = {r["texto"] for r in sop}
        ev = [r for r in pool_eval if r["id"] not in usados and r["texto"] not in textos_sop]
        antes = len(ev)
        ev = [r for r in ev if norm(r["texto"]) not in vistos]
        solape = antes - len(ev)
        # el soporte de kNN-e5 no se filtra: kNN no se entrena, solo lo consulta
        rng.shuffle(ev)
        ev = ev[:tope]
        for r in ev:
            r["split"] = "test"
        for r in sop:
            r["split"] = "train"
        por[corto] = {"test": ev, "train": sop}
        meta[corto] = dict(m, n_eval=len(ev), n_soporte=len(sop), solape_tsjev_descartado=solape, k=k, tope=tope,
                           clases=len({r["etiqueta"] for r in ev}))
    return por, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hub", required=True)
    ap.add_argument("--tsjev", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--tope", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args(argv)
    por, meta = construir(a.hub, a.tsjev, a.tope, a.k)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    for corto, d in por.items():
        with open(sal / (corto + ".jsonl"), "w", encoding="utf-8") as fh:
            for r in d["train"] + d["test"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = sum(m["n_eval"] for m in meta.values())
    es = sum(m["n_eval"] for m in meta.values() if m["idioma"] == "es")
    meta["_total"] = dict(n_eval=tot, n_eval_es=es, pct_es=round(100 * es / tot, 1))
    (sal / "manifiesto.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
