"""Batería v2 de evaluación zero-/few-shot para un clasificador general de texto. SOLO PARA MEDIR, nunca para entrenar.

    python -m model.alt.bateria_v2 descargar --hub DIR_HF_HUB
    python -m model.alt.bateria_v2 construir --hub DIR_HF_HUB --salida DIR --manifiesto model/alt/bateria-v2-manifiesto.json

Formato de la batería v1 (`model.datos.comun.fila`): un .jsonl por tarea con el soporte (split train, k = 8 por clase, de oro) y el test
(tope 500, estratificado por clase con semilla fija). El texto se corta a 3 000 caracteres (se anota cuántos). La revisión de cada
fuente es el sha de HF (nombre de la carpeta de la instantánea). Los datos descargados NUNCA entran al repo: aquí solo hay código,
manifiesto y cifras.

Una tarea por FAMILIA (dataset original o derivados del mismo corpus). Dentro de cada tarea se quitan del test los textos duplicados
exactos (texto normalizado), los que tienen dos etiquetas distintas y los que están también en el soporte.

Partición dev/selladas: semilla fija, estratificada por tipo, con idiomas y dominios repartidos; se calcula SOLO con los metadatos de
las tareas (nunca con resultados). Los ficheros sellados solo se leen con `abrir_sellada(ruta, motivo)`, que anota la apertura en
`bateria-v2-aperturas.jsonl` antes de leer.
"""
import argparse
import getpass
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from model.alt.bateria_no_vista import snapshot
from model.datos import comun

TOPE_CAR = 3000
K = 8
TOPE_TEST = 500
SEMILLA_PARTICION = 20261115
N_DEV = 6
APERTURAS = Path(__file__).with_name("bateria-v2-aperturas.jsonl")


def norm(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


# ---------------------------------------------------------------------------------------------------------------- descarga
DESCARGAS = {   # repo -> patrones de snapshot_download
    "jumafernandez/consultas-unlu": ["data/train.csv", "data/test.csv", "README.md"],
    "gretelai/synthetic_pii_finance_multilingual": ["data/Spanish*", "README.md"],
    "mrcsgh/horeca-spanish-reviews": ["dataset.txt", "README.md"],
    "PoetryMTEB/DISCOPALAffectClassification": ["data/*", "label_taxonomy.json", "README.md"],
    "JQL-AI/JQL-Human-Edu-Annotations": ["human_aggregated_scores/*_es_*", "README.md"],
    "hsilvosa/congreso-debates": ["data/intervenciones/*", "README.md"],
    "MariaIsabel/PROMISE_NFR_translated": ["*.txt", "README.md"],
    "BEE-spoke-data/consumer-finance-complaints": ["data/train-00000-of-00007.parquet", "README.md"],
    "chibifire/kaggle-womens-ecom-clothing-reviews": ["data/*", "README.md"],
    "allenai/real-toxicity-prompts": ["*.jsonl", "README.md"],
    "Intel/polite-guard": ["data/*", "README.md"],
    "abigailhaddad/usajobs-scraping": ["data/2024_06.parquet", "README.md"],
    "kvenanzi/congressional-record-speeches-1995-2026": ["samples/govinfo_118.parquet", "README.md"],
    "librarian-bots/arxiv-metadata-snapshot": ["data/train-00005-of-00011.parquet", "README.md"],
}


def descargar(hub):
    """snapshot_download de cada fuente, en la revisión más reciente en este momento; el sha queda en el nombre de la carpeta."""
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for repo, pat in DESCARGAS.items():
        sha = api.dataset_info(repo).sha
        p = snapshot_download(repo, repo_type="dataset", revision=sha, allow_patterns=pat, cache_dir=str(hub))
        print("OK", repo, sha, p)


# ---------------------------------------------------------------------------------------------------------------- lectores
def _parquet(ruta, columnas=None, filtro=None, lote=2000):
    """Filas (dict) de un parquet leído por lotes y con columnas acotadas; `filtro(dict)->bool` opcional."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(ruta))
    for b in pf.iter_batches(batch_size=lote, columns=columnas):
        for r in b.to_pylist():
            if filtro is None or filtro(r):
                yield r


def _csv(ruta, **kw):
    import csv
    csv.field_size_limit(10 ** 8)
    with open(ruta, encoding=kw.pop("encoding", "utf-8-sig"), newline="") as fh:
        yield from csv.DictReader(fh, **kw)


def _por_clase_tope(filas, clave, tope, semilla):
    """Muestrea como mucho `tope` filas por clase (para no cargar millones de filas en memoria)."""
    rng = random.Random(semilla)
    g = defaultdict(list)
    for r in filas:
        g[clave(r)].append(r)
    out = []
    for c in sorted(g):
        v = g[c]
        rng.shuffle(v)
        out += v[:tope]
    return out


# ---------------------------------------------------------------------------------------------------------------- tareas
def _spec(tarea, tipo, pregunta, idioma, opciones=None, descripciones=None):
    return dict(tarea=tarea, tipo=tipo, pregunta=pregunta, idioma=idioma, opciones=opciones, descripciones=descripciones or {})


def t_unlu(raiz):
    items = []
    for s, f in (("train", "train.csv"), ("test", "test.csv")):
        for r in _csv(raiz / "data" / f):
            items.append(dict(texto=r["consulta"], etiqueta=r["clase"].strip(), split=s))
    clases = sorted({i["etiqueta"] for i in items})
    return items, _spec("tema_consulta_universitaria", "choice", "¿Sobre qué trata la consulta que un estudiante envía por correo a la administración de la universidad?", "es",
                        clases, {c: c for c in clases})


def t_gretel(raiz):
    items, desc = [], {}
    for s in ("train", "test"):
        for r in _parquet(next(raiz.glob("data/Spanish_%s-*.parquet" % s)), ["document_type", "document_description", "generated_text", "domain"]):
            items.append(dict(texto=r["generated_text"], etiqueta=r["document_type"], split=s))
            desc.setdefault(r["document_type"], r["document_description"])
    clases = sorted(desc)
    return items, _spec("tipo_documento_financiero", "choice", "¿Qué tipo de documento (financiero o de negocio) es este texto?", "es", clases, desc)


HORECA = {"1": "reseña muy negativa (1 estrella)", "2": "reseña negativa (2 estrellas)", "3": "reseña regular o mixta (3 estrellas)",
          "4": "reseña positiva (4 estrellas)", "5": "reseña excelente (5 estrellas)"}


def t_horeca(raiz):
    items = [dict(texto=r["texto"], etiqueta=str(int(r["estrellas"])), split=None) for r in _csv(raiz / "dataset.txt", delimiter="\t")]
    return items, _spec("estrellas_resena_hosteleria", "score", "¿Cuántas estrellas (1 a 5) da el autor de esta reseña a un local de hostelería?", "es", sorted(HORECA), HORECA)


DISCO = {"1": "valencia muy negativa: el soneto evoca un afecto desagradable o triste", "2": "valencia algo negativa", "3": "valencia algo positiva", "4": "valencia muy positiva: el soneto evoca un afecto agradable o alegre"}


def t_disco(raiz):
    items = []
    for s, f in (("train", "train"), ("test", "test")):
        for r in _parquet(next(raiz.glob("data/%s-*.parquet" % f))):
            nombres = [str(x) for x in r["dimensional_names"]]
            v = float(list(r["dimensional_scores"])[nombres.index("Valence")])
            items.append(dict(texto=r["poem"], etiqueta=str(int(round(v))), split=s))
    return items, _spec("valencia_soneto", "score", "¿Qué valencia afectiva (1 muy negativa a 4 muy positiva) evoca este soneto, según jueces expertos?", "es", sorted(DISCO), DISCO)


JQL = {"1": "sin valor educativo (0)", "2": "valor educativo mínimo (1)", "3": "valor educativo básico (2)", "4": "valor educativo bueno (3)",
       "5": "valor educativo muy bueno (4)", "6": "valor educativo excelente (5)"}


def t_jql(raiz):
    items = []
    for r in (json.loads(l) for l in open(next(raiz.glob("human_aggregated_scores/*_es_*.jsonl")), encoding="utf-8")):
        items.append(dict(texto=r["text"], etiqueta=str(int(round(float(r["score"]))) + 1), split=None))
    return items, _spec("valor_educativo_web", "score", "¿Qué valor educativo tiene este texto de la web para enseñar (0 nulo a 5 excelente; el id es la nota más uno)?", "es", sorted(JQL), JQL)


_GRUPO = re.compile(r"\(([^()]*(?:\([^()]*\))?[^()]*)\)\s*$")
GRUPOS = {"GP": "Grupo Popular", "GS": "Grupo Socialista", "GVOX": "Grupo VOX", "GPlu": "Grupo Plural", "GR": "Grupo Republicano",
          "GMx": "Grupo Mixto", "GV (EAJ-PNV)": "Grupo Vasco (EAJ-PNV)", "GSUMAR": "Grupo Sumar", "GCs": "Grupo Ciudadanos", "GCUP-EC-GC": "Grupo Confederal Unidas Podemos",
          "GEH Bildu": "Grupo EH Bildu", "GJxCAT": "Grupo Junts per Catalunya"}


def t_congreso(raiz):
    cnt, filas = Counter(), []
    for r in _parquet(next(raiz.glob("data/intervenciones/*.parquet")), ["orador", "cargo_orador", "speech_text", "tipo_intervencion", "legislature"]):
        m = _GRUPO.search((r["orador"] or "").strip())
        if not m or "Diputad" not in (r["cargo_orador"] or ""):
            continue
        g = m.group(1).strip()
        sur = r["orador"].split(",")[0].strip().upper()
        t = r["speech_text"] or ""
        mm = re.search(r"(?:El señor|La señora)\s+" + re.escape(sur) + r"[^:\n]{0,60}:", t)
        if not mm:
            continue                      # sin su turno localizable no hay texto propio del orador
        t = t[mm.end():].strip()
        if len(t) < 300:
            continue
        cnt[g] += 1
        filas.append((g, t))
    top = {g for g, n in cnt.items() if n >= 600}
    filas = _por_clase_tope([dict(g=g, t=t) for g, t in filas if g in top], lambda r: r["g"], 700, "congreso")
    clases = sorted({r["g"] for r in filas})
    return [dict(texto=r["t"], etiqueta=r["g"], split=None) for r in filas], _spec(
        "grupo_parlamentario", "choice", "¿A qué grupo parlamentario pertenece el diputado que pronuncia esta intervención en el Congreso de los Diputados de España?", "es",
        clases, {c: GRUPOS.get(c, c) for c in clases})


NFR = {"F": "requisito funcional: qué debe hacer el sistema", "A": "disponibilidad (availability)", "L": "requisito legal o normativo", "LF": "aspecto y apariencia de la interfaz (look and feel)",
       "MN": "mantenibilidad", "O": "requisito operativo: entorno de operación", "PE": "rendimiento (performance)", "SC": "escalabilidad", "SE": "seguridad",
       "US": "usabilidad", "FT": "tolerancia a fallos", "PO": "portabilidad"}


def t_promise(raiz):
    p = next(raiz.glob("*.txt"))
    items = [dict(texto=r["REQUIREMENT"], etiqueta=r["LABEL"].strip(), split=None) for r in _csv(p)]
    clases = sorted({i["etiqueta"] for i in items})
    return items, _spec("tipo_requisito_software", "choice", "¿Qué tipo de requisito de software es este (funcional o qué clase de requisito no funcional)?", "es", clases, {c: NFR.get(c, c) for c in clases})


def t_cfpb(raiz):
    ruta = next(raiz.glob("data/train-00000-*.parquet"))
    # la CFPB renombró dos productos en 2023: se unifican para que no haya dos clases que son la misma
    unir = {"Credit card": "Credit card or prepaid card", "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting or other personal consumer reports"}
    filas = [dict(r, Product=unir.get(r["Product"], r["Product"])) for r in _parquet(ruta, ["Product", "Consumer complaint narrative", "Date received"],
             lambda r: bool(r["Consumer complaint narrative"]) and r["Date received"] >= "2022-01-01")]
    cnt = Counter(r["Product"] for r in filas)
    top = [p for p, n in cnt.most_common(9) if n >= 800]
    filas = _por_clase_tope([r for r in filas if r["Product"] in top], lambda r: r["Product"], 900, "cfpb")
    clases = sorted({r["Product"] for r in filas})
    return [dict(texto=r["Consumer complaint narrative"], etiqueta=r["Product"], split=None) for r in filas], _spec(
        "producto_queja_financiera", "choice", "¿Sobre qué producto o servicio financiero trata esta queja de un consumidor?", "en", clases, {c: c for c in clases})


CLOTH = {"1": "very negative review (1 star)", "2": "negative review (2 stars)", "3": "mixed or lukewarm review (3 stars)", "4": "positive review (4 stars)", "5": "excellent review (5 stars)"}


def t_cloth(raiz):
    items = []
    for r in _parquet(next(raiz.glob("data/*.parquet")), ["rating", "title", "review_text"]):
        if not r["review_text"] or not r["rating"]:
            continue
        t = ((r["title"] + "\n") if r["title"] else "") + r["review_text"]
        items.append(dict(texto=t, etiqueta=str(int(r["rating"])), split=None))
    return items, _spec("estrellas_resena_ropa", "score", "¿Cuántas estrellas (1 a 5) da el cliente a esta prenda en su reseña de una tienda en línea?", "en", sorted(CLOTH), CLOTH)


RTP = {"1": "casi nada tóxico (puntuación Perspective < 0,2)", "2": "poco tóxico (0,2 a 0,4)", "3": "algo tóxico (0,4 a 0,6)", "4": "bastante tóxico (0,6 a 0,8)", "5": "muy tóxico (>= 0,8)"}


def t_rtp(raiz):
    items = []
    with open(next(raiz.glob("prompts.jsonl")), encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)["prompt"]
            if r.get("toxicity") is None or not (r.get("text") or "").strip():
                continue
            items.append(dict(texto=r["text"], etiqueta=str(min(5, int(r["toxicity"] * 5) + 1)), split=None))
    items = _por_clase_tope(items, lambda r: r["etiqueta"], 1500, "rtp")
    return items, _spec("toxicidad_texto_web", "score", "¿Cuánta toxicidad (1 casi nula a 5 muy alta) tiene este fragmento de texto de la web, medida con Perspective API?", "en", sorted(RTP), RTP)


POLITE = {"1": "impolite: rude, hostile or disrespectful", "2": "neutral: neither polite nor impolite", "3": "somewhat polite: courteous but plain or hedged", "4": "polite: clearly courteous and respectful"}
_PLV = {"impolite": "1", "neutral": "2", "somewhat polite": "3", "polite": "4"}


def t_polite(raiz):
    """Test: los 200 textos reales anotados de formaciones corporativas (test_lms); soporte: sintético de train_few_shot (se anota en `fuente`)."""
    items = []
    for r in _csv(raiz / "data" / "test" / "test_lms.csv"):
        items.append(dict(texto=r["text"], etiqueta=_PLV[r["label"].strip()], split="test", fuente_extra="test_lms"))
    sop = [dict(texto=r["text"], etiqueta=_PLV[r["label"].strip()], split="train", fuente_extra="train_few_shot (sintético: " + r["source"] + ")")
           for r in _csv(raiz / "data" / "train" / "train_few_shot.csv")]
    return items + _por_clase_tope(sop, lambda r: r["etiqueta"], 400, "polite"), _spec(
        "cortesia_atencion_cliente", "score", "¿Qué nivel de cortesía (1 descortés a 4 cortés) tiene este mensaje de atención al cliente?", "en", sorted(POLITE), POLITE)


_GS = re.compile(r"\bG[SG]\s*-?\s*\d{1,2}\b|\bgrade\s+\d{1,2}\b", re.I)
GRADO = {str(i): "nivel GS-%d de la escala salarial federal de EE. UU. (id = grado menos 4)" % (i + 4) for i in range(1, 12)}


def t_usajobs(raiz):
    cols = ["payScale", "minimumGrade", "positionTitle", "jobSummary", "majorDuties"]
    items = []
    for r in _parquet(next(raiz.glob("data/2024_06.parquet")), cols, lambda r: r["payScale"] == "GS" and (r["minimumGrade"] or "").isdigit() and 5 <= int(r["minimumGrade"]) <= 15):
        if "NOT AN OPEN ANNOUNCEMENT" in (r["jobSummary"] or "").upper():
            continue
        t = "\n".join(x for x in (r["positionTitle"], r["jobSummary"], r["majorDuties"]) if x)
        items.append(dict(texto=_GS.sub("GS-XX", t), etiqueta=str(int(r["minimumGrade"]) - 4), split=None))
    return items, _spec("nivel_puesto_federal", "score", "¿Qué nivel tiene este puesto de la administración federal de EE. UU. en la escala GS (1 = GS-5 ... 11 = GS-15), según su título y funciones?", "en", sorted(GRADO, key=int), GRADO)


ARXIV = {"cs.CL": "computación y lenguaje (PLN)", "cs.CV": "visión por computador", "cs.LG": "aprendizaje automático", "cs.CR": "criptografía y seguridad", "cs.RO": "robótica",
         "quant-ph": "física cuántica", "hep-ph": "física de altas energías: fenomenología", "hep-th": "física de altas energías: teoría", "gr-qc": "relatividad general y cosmología cuántica",
         "astro-ph.GA": "astrofísica de galaxias", "cond-mat.mtrl-sci": "ciencia de materiales", "cond-mat.mes-hall": "física mesoscópica y del nanoescala",
         "math.PR": "matemáticas: probabilidad", "math.NT": "matemáticas: teoría de números", "math.CO": "matemáticas: combinatoria", "math.AP": "matemáticas: ecuaciones en derivadas parciales",
         "math.OC": "matemáticas: optimización y control", "eess.IV": "ingeniería: procesamiento de imagen y vídeo"}
_LIC_OK = ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", )


def t_arxiv(raiz):
    """Solo artículos cuya licencia propia es CC BY 4.0 o CC0 (los metadatos de arXiv son CC0, pero así no hay dudas con los resúmenes)."""
    ok = ("http://creativecommons.org/licenses/by/4.0/", "http://creativecommons.org/publicdomain/zero/1.0/")
    filas = list(_parquet(next(raiz.glob("data/train-*.parquet")), ["categories", "title", "abstract", "license"], lambda r: r["license"] in ok and (r["categories"] or "").split()[:1] and r["categories"].split()[0] in ARXIV, lote=20000))
    filas = _por_clase_tope([dict(c=r["categories"].split()[0], t=re.sub(r"\s+", " ", r["title"]) + "\n" + re.sub(r"\s+", " ", r["abstract"]).strip()) for r in filas], lambda r: r["c"], 400, "arxiv")
    return [dict(texto=r["t"], etiqueta=r["c"], split=None) for r in filas], _spec(
        "categoria_arxiv", "choice", "¿Cuál es la categoría principal de arXiv de este artículo científico (título y resumen)?", "en", sorted(ARXIV), ARXIV)


def t_congress(raiz):
    items = []
    for r in _parquet(next(raiz.glob("samples/govinfo_118.parquet")), ["party", "text", "split"]):
        if r["party"] not in ("D", "R"):
            continue
        s = {"train": "train", "test": "test"}.get(r["split"])
        if s:
            items.append(dict(texto=r["text"], etiqueta=r["party"], split=s))
    return items, _spec("partido_discurso_congreso_eeuu", "choice", "¿A qué partido (D demócrata, R republicano) pertenece el congresista que pronuncia este discurso en el Congreso de EE. UU.?", "en",
                        ["D", "R"], {"D": "Partido Demócrata", "R": "Partido Republicano"})


# Metadatos de cada tarea: nada de aquí depende de resultados.
CC4 = "cc-by-4.0"
TAREAS = [
    dict(nombre="unlu-consultas", repo="jumafernandez/consultas-unlu", loader=t_unlu, familia="consultas-unlu", dominio="soporte", licencia=CC4,
         url_tarjeta="https://huggingface.co/datasets/jumafernandez/consultas-unlu", licencia_fuente="CC BY 4.0 declarada por los propios autores del corpus (Fernández y Errecalde, 2022; correos de estudiantes de la UNLu)",
         url_fuente="https://doi.org/10.1007/978-3-031-14599-5_6", sintetico=False),
    dict(nombre="finanzas-documentos", repo="gretelai/synthetic_pii_finance_multilingual", loader=t_gretel, familia="gretel-synthetic-pii-finance", dominio="banca", licencia="apache-2.0",
         url_tarjeta="https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual", licencia_fuente="Apache-2.0 declarada por Gretel; sintético generado con Gretel Navigator (LLM; no hay otra fuente)",
         url_fuente="https://gretel.ai/blog/synthetic-pii-finance-multilingual", sintetico="LLM (Gretel Navigator); Apache-2.0 declarada; términos del LLM subyacente no publicados: reserva"),
    dict(nombre="horeca-resenas", repo="mrcsgh/horeca-spanish-reviews", loader=t_horeca, familia="horeca-spanish-reviews", dominio="reseñas", licencia="mit",
         url_tarjeta="https://huggingface.co/datasets/mrcsgh/horeca-spanish-reviews", licencia_fuente="MIT declarada; reseñas sintéticas generadas con GPT-5 y Gemini 3 (según el notebook del repo): términos de OpenAI y Google permiten usar la salida; aquí solo se mide",
         url_fuente="https://huggingface.co/datasets/mrcsgh/horeca-spanish-reviews/blob/main/descripcion_dataset.ipynb", sintetico="LLM (GPT-5, Gemini 3)"),
    dict(nombre="disco-soneto-valencia", repo="PoetryMTEB/DISCOPALAffectClassification", loader=t_disco, familia="disco-pal", dominio="emoción", licencia="apache-2.0",
         url_tarjeta="https://huggingface.co/datasets/PoetryMTEB/DISCOPALAffectClassification", licencia_fuente="anotaciones DISCO PAL: Apache-2.0 (github.com/AlbertoBarbado/DISCO_PAL); sonetos DISCO: CC BY (github.com/pruizf/disco). Manda la más restrictiva: atribución",
         url_fuente="https://github.com/AlbertoBarbado/DISCO_PAL", sintetico=False),
    dict(nombre="jql-valor-educativo", repo="JQL-AI/JQL-Human-Edu-Annotations", loader=t_jql, familia="jql-edu", dominio="educación", licencia="odc-by",
         url_tarjeta="https://huggingface.co/datasets/JQL-AI/JQL-Human-Edu-Annotations", licencia_fuente="ODC-By declarada; 511 documentos web en inglés (FineWeb) anotados por 3 personas y TRADUCIDOS al español con DeepL; la traducción es un derivado: manda ODC-By",
         url_fuente="https://arxiv.org/abs/2505.22232", sintetico="traducción automática (DeepL) de texto original en inglés"),
    dict(nombre="congreso-grupo", repo="hsilvosa/congreso-debates", loader=t_congreso, familia="congreso-diputados", dominio="política", licencia=CC4,
         url_tarjeta="https://huggingface.co/datasets/hsilvosa/congreso-debates", licencia_fuente="información del sector público (Ley 37/2007, Directiva (UE) 2019/1024), Diarios de Sesiones del Congreso de los Diputados, reutilizable con atribución",
         url_fuente="https://www.congreso.es/datos-abiertos", sintetico=False),
    dict(nombre="promise-requisitos", repo="MariaIsabel/PROMISE_NFR_translated", loader=t_promise, familia="promise-nfr", dominio="software", licencia=CC4,
         url_tarjeta="https://huggingface.co/datasets/MariaIsabel/PROMISE_NFR_translated", licencia_fuente="traducción al español de PROMISE NFR (repositorio PROMISE, Zenodo): derivado; manda la más restrictiva de ambas (verificar la de PROMISE antes de uso comercial)",
         url_fuente="http://promise.site.uottawa.ca/SERepository/datasets-page.html", sintetico="traducción (automática o manual) del original en inglés"),
    dict(nombre="cfpb-quejas", repo="BEE-spoke-data/consumer-finance-complaints", loader=t_cfpb, familia="cfpb-consumer-complaints", dominio="quejas", licencia="cc0-1.0",
         url_tarjeta="https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints", licencia_fuente="obra del gobierno de EE. UU. (CFPB, Consumer Complaint Database): dominio público; los relatos se publican con consentimiento del consumidor",
         url_fuente="https://www.consumerfinance.gov/data-research/consumer-complaints/", sintetico=False),
    dict(nombre="ropa-resenas", repo="chibifire/kaggle-womens-ecom-clothing-reviews", loader=t_cloth, familia="womens-ecommerce-clothing-reviews", dominio="reseñas", licencia="cc0-1.0",
         url_tarjeta="https://huggingface.co/datasets/chibifire/kaggle-womens-ecom-clothing-reviews", licencia_fuente="CC0 declarada en Kaggle (nicapotato/womens-ecommerce-clothing-reviews); datos reales de un comercio, anonimizados",
         url_fuente="https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews", sintetico=False),
    dict(nombre="rtp-toxicidad", repo="allenai/real-toxicity-prompts", loader=t_rtp, familia="real-toxicity-prompts", dominio="toxicidad", licencia="apache-2.0",
         url_tarjeta="https://huggingface.co/datasets/allenai/real-toxicity-prompts", licencia_fuente="Apache-2.0 (repositorio allenai/real-toxicity-prompts); textos de OpenWebText; la puntuación es de Perspective API (continua, aquí en 5 tramos)",
         url_fuente="https://github.com/allenai/real-toxicity-prompts", sintetico=False),
    dict(nombre="polite-cortesia", repo="Intel/polite-guard", loader=t_polite, familia="intel-polite-guard", dominio="soporte", licencia="cdla-permissive-2.0",
         url_tarjeta="https://huggingface.co/datasets/Intel/polite-guard", licencia_fuente="CDLA-Permissive-2.0 (Intel). Test: 200 textos reales anotados de formaciones corporativas. Soporte: sintético (Llama 3.1 8B, Gemma 2 9B, Mixtral 8x7B): Llama exige «Built with Llama» al redistribuir",
         url_fuente="https://github.com/intel/polite-guard", sintetico="solo el soporte (LLM: Llama 3.1, Gemma 2, Mixtral)"),
    dict(nombre="usajobs-nivel", repo="abigailhaddad/usajobs-scraping", loader=t_usajobs, familia="usajobs", dominio="empleo", licencia="cc0-1.0",
         url_tarjeta="https://huggingface.co/datasets/abigailhaddad/usajobs-scraping", licencia_fuente="obra del gobierno federal de EE. UU., sin copyright (anuncios de usajobs.gov)",
         url_fuente="https://www.usajobs.gov/", sintetico=False),
    dict(nombre="arxiv-categoria", repo="librarian-bots/arxiv-metadata-snapshot", loader=t_arxiv, familia="arxiv", dominio="ciencia", licencia="cc0-1.0",
         url_tarjeta="https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot", licencia_fuente="metadatos de arXiv bajo CC0; además solo se usan artículos cuya licencia propia es CC BY 4.0 o CC0",
         url_fuente="https://www.kaggle.com/datasets/Cornell-University/arxiv", sintetico=False),
    dict(nombre="congreso-eeuu-partido", repo="kvenanzi/congressional-record-speeches-1995-2026", loader=t_congress, familia="congressional-record", dominio="política", licencia="cc0-1.0",
         url_tarjeta="https://huggingface.co/datasets/kvenanzi/congressional-record-speeches-1995-2026", licencia_fuente="Congressional Record: obra del gobierno de EE. UU., dominio público (GovInfo); recopilación CC0",
         url_fuente="https://www.govinfo.gov/app/collection/crec", sintetico=False),
]

DESCARTADOS = [
    dict(nombre="BRIGHTER (categorías e intensidades de emoción, es)", repo="brighter-dataset/BRIGHTER-emotion-categories", motivo="CC BY 4.0 en la tarjeta, pero el subconjunto en español son comentarios de YouTube (paper, sección de fuentes): la plataforma no permite redistribuir"),
    dict(nombre="SocialTOX (toxicidad en 3 niveles, comentarios de prensa)", repo="gplsi/SocialTOX", motivo="CC BY 4.0 en la tarjeta, pero son comentarios de lectores de 16 diarios y la tarjeta no dice nada de los derechos de los comentarios: licencia de la fuente sin verificar"),
    dict(nombre="SINAI/ALIA-es-discriminative-*", repo="SINAI/ALIA-es-discriminative-hate-speech", motivo="la familia ALIA está en la lista de exclusión pendiente de decisión (manifiesto-entrenamiento)"),
    dict(nombre="News Category Dataset (titulares de HuffPost, noticias)", repo="heegyu/news-category-dataset", motivo="la licencia se contradice (CC BY 4.0 en el paper, CC0 en copias, sin licencia verificable en Kaggle) y son titulares de HuffPost con copyright: sin verificar"),
    dict(nombre="mteb/SpanishNewsClassification (noticias, es)", repo="mteb/SpanishNewsClassification", motivo="MIT en la tarjeta, pero la fuente (MarcOrfilaCarreras/spanish-news) ya no existe en HF y no hay licencia verificable de las noticias"),
    dict(nombre="hacktoberfest-corpus-es/colmbian_spanish_news", repo="hacktoberfest-corpus-es/colmbian_spanish_news", motivo="CC BY 2.0 sin tarjeta ni fuente: son noticias de prensa con copyright, licencia sin verificar"),
    dict(nombre="PleIAs/ToxicCommons (es)", repo="PleIAs/ToxicCommons", motivo="el español son libros y periódicos antiguos en OCR muy sucio; anotación de toxicidad hecha por LLM sin validar"),
    dict(nombre="toxigen/toxigen-data", repo="toxigen/toxigen-data", motivo="sin licencia en la tarjeta y con formulario de acceso; texto generado con GPT-3"),
    dict(nombre="adewynter/RTP-LX (toxicidad 1-5, es)", repo="adewynter/RTP-LX", motivo="acceso restringido (gated) y licencia MIT dudosa para un derivado de RealToxicityPrompts traducido"),
    dict(nombre="TheFinAI/es-multifin (finanzas, es)", repo="TheFinAI/es-multifin", motivo="acceso restringido (gated)"),
    dict(nombre="Gabrielaz/spamspa (spam, es)", repo="Gabrielaz/spamspa", motivo="acceso restringido (gated) y sin tarjeta"),
    dict(nombre="somosnlp-hackathon-2022/unam_tesis", repo="somosnlp-hackathon-2022/unam_tesis", motivo="Apache-2.0 en la tarjeta, pero son datos raspados del catálogo TESIUNAM sin licencia de la fuente"),
    dict(nombre="somosnlp-hackathon-2022/readability-es-*", repo="somosnlp-hackathon-2022/readability-es-hackathon-pln-public", motivo="CC BY 4.0 en la tarjeta, pero incluye textos raspados de kwiziq (sitio comercial) y otros corpus sin licencia verificada"),
    dict(nombre="Mateo-Rua/titulos-productos-ecommerce-es", repo="Mateo-Rua/titulos-productos-ecommerce-es", motivo="Apache-2.0 en la tarjeta, pero deriva del MeLi Data Challenge 2019 (Mercado Libre) cuya licencia no es verificable"),
    dict(nombre="somosnlp-hackathon-2023/informes_discriminacion_gitana", repo="somosnlp-hackathon-2023/informes_discriminacion_gitana", motivo="informes de la Fundación Secretariado Gitano extraídos de PDF sin licencia de la fuente; parte sintética"),
    dict(nombre="PrevenIA/spanish-suicide-intent", repo="PrevenIA/spanish-suicide-intent", motivo="traducción de comentarios de Reddit y otras plataformas: no se puede redistribuir"),
    dict(nombre="neurocheckout-ai/synthetic-ecommerce-support-intents (es)", repo="neurocheckout-ai/synthetic-ecommerce-support-intents", motivo="CC BY 4.0 pero solo 10 filas de test en español (plantillas), insuficiente"),
    dict(nombre="NOSIBLE/financial-sentiment", repo="NOSIBLE/financial-sentiment", motivo="ODC-By, pero etiquetas votadas por 8 LLM (Grok, GPT, Gemini, Llama, Qwen...) con términos distintos y texto de un feed propietario"),
    dict(nombre="hakai-vulpes/congreso-opinion-publica", repo="hakai-vulpes/congreso-opinion-publica", motivo="misma familia que congreso-diputados (una tarea por familia)"),
    dict(nombre="Misc. XNLI, SIB-200, MASSIVE, GoEmotions, Amazon, banking77, civil_comments, tweet_sentiment, AG News...", repo="-", motivo="familias ya en el manifiesto de entrenamiento, en la exclusión pendiente (XNLI, SIB-200) o en la batería v1 (MASSIVE, hatecheck, tweets)"),
]


# ---------------------------------------------------------------------------------------------------------------- construcción
def _estratificar(filas, tope, rng):
    """Hasta `tope` filas repartidas lo más parejo posible entre clases (las pequeñas entran enteras)."""
    g = defaultdict(list)
    for r in filas:
        g[r["etiqueta"]].append(r)
    cl = sorted(g, key=lambda c: (len(g[c]), c))
    resto, out = tope, []
    for i, c in enumerate(cl):
        v = g[c]
        rng.shuffle(v)
        take = min(len(v), resto // (len(cl) - i))
        out += v[:take]
        resto -= take
    rng.shuffle(out)
    return out


def armar(corto, items, spec, fuente, licencia, k=K, tope=TOPE_TEST, semilla=0):
    """items = [{texto, etiqueta, split (train|test|None), fuente_extra?}] -> (filas, cifras)."""
    rng = random.Random("%s-%d" % (corto, semilla))
    cif = dict(n_origen=len(items))
    ok = [dict(i, texto=(i["texto"] or "").strip()) for i in items if (i["texto"] or "").strip() and i["etiqueta"] is not None]
    cif["vacios"] = len(items) - len(ok)
    for n, i in enumerate(ok):
        i["_n"] = n
    rng.shuffle(ok)
    tiene = any(i.get("split") for i in ok)
    pool_sop = [i for i in ok if i.get("split") == "train"] if tiene else list(ok)
    pool_ev = [i for i in ok if i.get("split") == "test"] if tiene else None
    # etiquetas distintas para un mismo texto en el pool de test: se descartan todas (ambiguas)
    def sin_dup(lista):
        et = defaultdict(set)
        for i in lista:
            et[norm(i["texto"])].add(i["etiqueta"])
        vistos, out, dup, conf = set(), [], 0, 0
        for i in lista:
            n = norm(i["texto"])
            if len(et[n]) > 1:
                conf += 1
                continue
            if n in vistos:
                dup += 1
                continue
            vistos.add(n)
            out.append(i)
        return out, dup, conf
    # soporte: k por clase de oro (si hay split train oficial, de él; si no, del conjunto)
    sop, usados = [], set()
    cands, _, _ = sin_dup(pool_sop)
    por_cl = defaultdict(list)
    for i in cands:
        por_cl[i["etiqueta"]].append(i)
    for c in sorted(por_cl):
        for i in por_cl[c][:k]:
            sop.append(i)
            usados.add(i["_n"])
    textos_sop = {norm(i["texto"]) for i in sop}
    pool_ev = [i for i in (pool_ev if tiene else ok) if i["_n"] not in usados]
    ev, dup, conf = sin_dup(pool_ev)
    cif.update(duplicados_test=dup, conflicto_test=conf)
    antes = len(ev)
    ev = [i for i in ev if norm(i["texto"]) not in textos_sop]
    cif["en_soporte_quitados"] = antes - len(ev)
    clases = list(spec["opciones"]) if spec["opciones"] else sorted({i["etiqueta"] for i in ev})
    cl_ev = {i["etiqueta"] for i in ev}
    if spec["tipo"] == "score":      # niveles 1..K contiguos con test: se recorta al prefijo contiguo
        K_ = 0
        while str(K_ + 1) in cl_ev:
            K_ += 1
        clases = [str(i) for i in range(1, K_ + 1)]
        cif["niveles_recortados"] = len([c for c in spec["opciones"] if c not in clases])
    else:
        clases = [c for c in clases if c in cl_ev]
    ev = [i for i in ev if i["etiqueta"] in clases]
    sop = [i for i in sop if i["etiqueta"] in clases]
    ev = _estratificar(ev, tope, rng)
    desc = {c: spec["descripciones"].get(c, c) for c in clases}
    cortados = 0
    filas_items = []
    for split, lista in (("train", sop), ("test", ev)):
        for i in lista:
            t = i["texto"]
            if len(t) > TOPE_CAR:
                cortados += 1
                t = t[:TOPE_CAR]
            f = fuente + (" " + i["fuente_extra"] if i.get("fuente_extra") else (" " + split))
            filas_items.append((split, dict(tarea=spec["tarea"], tipo=spec["tipo"], pregunta=spec["pregunta"], texto=t, opciones=clases, descripciones=desc,
                                           etiqueta=i["etiqueta"], idioma=spec["idioma"], fuente=f, licencia=licencia, uso_comercial=True)))
    filas, _ = comun.numerar(corto, filas_items)
    if spec["tipo"] == "score" and not comun.score_valido(clases):
        raise ValueError("%s: niveles score no contiguos %r" % (corto, clases))
    cif.update(n_test=len(ev), n_soporte=len(sop), clases=len(clases), cortados=cortados, test_por_clase=dict(Counter(i["etiqueta"] for i in ev)))
    return filas, cif


def particionar(tareas=None, n_dev=N_DEV, semilla=SEMILLA_PARTICION):
    """{nombre: 'dev'|'sellada'}. Solo usa metadatos (tipo, idioma, dominio). Dev: mitad score y mitad choice, con dominios distintos y
    ambos idiomas presentes en cada tipo cuando se puede; el sorteo es con semilla fija y se repite hasta cumplir las restricciones."""
    tareas = tareas or TAREAS
    meta = {t["nombre"]: t for t in tareas}
    tipos = {}
    for n, t in meta.items():
        tipos.setdefault(tipo_de(t), []).append(n)
    rng = random.Random(semilla)
    reparto = {}
    resto = n_dev
    ts = sorted(tipos, key=lambda x: (-len(tipos[x]), x))
    for i, t in enumerate(ts):
        reparto[t] = round(resto / (len(ts) - i))
        resto -= reparto[t]
    for _ in range(10000):
        dev = []
        for t in ts:
            dev += rng.sample(sorted(tipos[t]), reparto[t])
        dom = [meta[n]["dominio"] for n in dev]
        idi = {idioma_de(meta[n]) for n in dev}
        if len(set(dom)) == len(dom) and idi == {"es", "en"}:
            return {n: ("dev" if n in dev else "sellada") for n in meta}
    raise RuntimeError("sin partición válida")


def tipo_de(t):
    return {"t_horeca": "score", "t_disco": "score", "t_jql": "score", "t_cloth": "score", "t_rtp": "score", "t_polite": "score", "t_usajobs": "score"}.get(t["loader"].__name__, "choice")


def idioma_de(t):
    return {"t_cfpb": "en", "t_cloth": "en", "t_rtp": "en", "t_polite": "en", "t_usajobs": "en", "t_arxiv": "en", "t_congress": "en"}.get(t["loader"].__name__, "es")


def sha256_fichero(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def construir(hub, salida, manifiesto, semilla=0):
    salida = Path(salida)
    part = particionar()
    man = dict(version=2, descripcion="Batería v2: SOLO MEDIR, nunca entrenar. Una tarea por familia; soporte de k=%d por clase; test <= %d; texto <= %d caracteres." % (K, TOPE_TEST, TOPE_CAR),
               semilla_particion=SEMILLA_PARTICION, semilla_muestreo=semilla, tareas=[], descartados=DESCARTADOS)
    for t in TAREAS:
        raiz = snapshot(hub, t["repo"])
        items, spec = t["loader"](raiz)
        fuente = t["repo"]
        corto = t["nombre"]
        filas, cif = armar(corto, items, spec, fuente, t["licencia"], semilla=semilla)
        d = salida / ("dev" if part[corto] == "dev" else "selladas")
        d.mkdir(parents=True, exist_ok=True)
        p = d / (corto + ".jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in filas:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        e = dict(nombre=corto, particion=part[corto], repo=t["repo"], rev=raiz.name, licencia=t["licencia"], url_licencia_tarjeta=t["url_tarjeta"],
                 licencia_fuente=t["licencia_fuente"], url_fuente=t["url_fuente"], familia=t["familia"], dominio=t["dominio"], tipo=spec["tipo"], idioma=spec["idioma"],
                 sintetico=t["sintetico"], clases=cif["clases"], n_test=cif["n_test"], n_soporte=cif["n_soporte"], cortados=cif["cortados"], tope_caracteres=TOPE_CAR,
                 sha256=sha256_fichero(p), fichero=("dev/" if part[corto] == "dev" else "selladas/") + corto + ".jsonl")
        if part[corto] == "dev":   # el detalle por clase solo de desarrollo; de las selladas, solo el sha256 y los conteos pedidos
            e["detalle"] = {k: v for k, v in cif.items() if k != "test_por_clase"}
            e["test_por_clase"] = cif["test_por_clase"]
        man["tareas"].append(e)
        print(corto, part[corto], cif["n_test"], cif["n_soporte"], cif["clases"], cif["cortados"], flush=True)
    man["resumen"] = dict(tareas=len(man["tareas"]), dev=sum(1 for t in man["tareas"] if t["particion"] == "dev"), selladas=sum(1 for t in man["tareas"] if t["particion"] == "sellada"),
                          score=sum(1 for t in man["tareas"] if t["tipo"] == "score"), dominios=sorted({t["dominio"] for t in man["tareas"]}),
                          idiomas=dict(Counter(t["idioma"] for t in man["tareas"])))
    Path(manifiesto).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return man


# ---------------------------------------------------------------------------------------------------------------- selladas
def abrir_sellada(ruta, motivo, quien=None, registro=APERTURAS, manifiesto=None):
    """Lee un .jsonl sellado ANOTANDO antes la apertura (fecha, quién, motivo, ruta, sha256 actual). Sin motivo se niega.
    Devuelve la lista de filas; si se pasa el manifiesto, avisa en el registro si el sha256 no coincide con el sellado."""
    if not (motivo or "").strip() or len(motivo.strip()) < 8:
        raise PermissionError("abrir_sellada exige un motivo explícito (>= 8 caracteres): no se lee %s" % ruta)
    ruta = Path(ruta)
    sha = sha256_fichero(ruta)
    linea = dict(fecha=time.strftime("%Y-%m-%dT%H:%M:%S%z"), quien=quien or getpass.getuser(), motivo=motivo.strip(), fichero=ruta.name, sha256=sha)
    if manifiesto:
        sel = {t["nombre"] + ".jsonl": t["sha256"] for t in json.loads(Path(manifiesto).read_text(encoding="utf-8"))["tareas"]}
        linea["coincide_con_manifiesto"] = sel.get(ruta.name) == sha
    with open(registro, "a", encoding="utf-8") as fh:      # primero se anota, luego se lee
        fh.write(json.dumps(linea, ensure_ascii=False) + "\n")
    return [json.loads(l) for l in open(ruta, encoding="utf-8")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("descargar")
    a.add_argument("--hub", required=True)
    b = sp.add_parser("construir")
    b.add_argument("--hub", required=True)
    b.add_argument("--salida", required=True)
    b.add_argument("--manifiesto", required=True)
    x = ap.parse_args(argv)
    if x.cmd == "descargar":
        descargar(x.hub)
    else:
        construir(x.hub, x.salida, x.manifiesto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
