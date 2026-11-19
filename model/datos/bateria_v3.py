"""Batería v3 de evaluación (SOLO MEDIR, nunca entrenar). Constructor de la FASE 2: se ejecuta en una máquina con red (model/datos/aws/bateria_v3*.sh).

    python -m model.datos.bateria_v3 manifiesto-inicial --salida model/datos/bateria-v3-manifiesto.json
    python -m model.datos.bateria_v3 candidatas-md --salida model/datos/bateria-v3-candidatas.md
    python -m model.datos.bateria_v3 construir --manifiesto M.json --trabajo DIR --salida DIR [--huellas DIR] [--maximo-solape 0]
    python -m model.datos.bateria_v3 reservas-v2 --manifiesto M.json --trabajo DIR --salida DIR
    python -m model.datos.bateria_v3 auditar-log --datos DIR --log FICHERO

`construir`, por cada candidata: baja la ficha (README.md o página) -> licencia automática (bateria_v3_licencias) -> baja los datos con revisión fijada y sha256 ->
valida el esquema -> mide solapes contra v2 y el entrenamiento (huellas, sin texto) y contra las demás tareas de v3 -> parte en soporte (k = 8 por clase) y test
(<= 500, estratificado, con semilla) -> escribe `soporte/` y `sellada/`. Lo que no pasa queda en el manifiesto con estado «fuera» y su motivo.

El constructor SOLO imprime recuentos y estado de validación: nunca texto de las filas ni valores de columnas de texto. Los ficheros de `sellada/` solo se leen con
`abrir_sellada(ruta, motivo)`, que anota la apertura en `bateria-v3-aperturas.jsonl` ANTES de leer (como v2) y se niega si el sha256 no coincide con el manifiesto.
"""
import argparse
import csv
import fnmatch
import getpass
import bz2
import gzip
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import chain, islice
from pathlib import Path

from model.datos import bateria_v3_licencias as lic
from model.datos import bateria_v3_solapes as sol
from model.datos.bateria_v3_candidatas import CANDIDATAS, DESCARTADAS, PREGUNTAS  # noqa: F401

AQUI = Path(__file__).resolve().parent
MANIFIESTO = AQUI / "bateria-v3-manifiesto.json"
APERTURAS = AQUI / "bateria-v3-aperturas.jsonl"
TOPE_CAR = 3000
K = 8
TOPE_TEST = 500
SEMILLA = 20261119
MIN_TEST_TAREA = 40          # una tarea con menos filas de test se queda fuera: no mide nada
TIPOS = ("choice", "yesno", "score")
N_BAJO = 8                   # menos de 8 tareas construidas que cuenten (por tipo, o en español en total): se informa con la marca «n bajo»
BLOQUES = ("cuentan", "informativo")
LICENCIA_PENDIENTE = "por verificar en fase 2"
CKAN_DEFECTO = "https://catalog.data.gov"

PROCEDIMIENTO_ABRIR_SELLADA = (
    "Los ficheros de sellada/ solo se leen con model.datos.bateria_v3.abrir_sellada(ruta, motivo, manifiesto=...). La función anota ANTES de leer una línea en "
    "model/datos/bateria-v3-aperturas.jsonl (fecha, quién, motivo >= 8 caracteres, fichero, sha256 actual) y se niega si el sha256 no coincide con el de este manifiesto "
    "o si el fichero no está en una carpeta «sellada». El constructor no lee ni imprime texto de test; los ficheros se escriben con permisos 0444. "
    "Una apertura que no se anota antes de leer invalida la tarea como sellada (§17.26: las selladas de v2 ya se abrieron y no valen)."
)

RESERVAS_V2 = [
    dict(nombre="promise-requisitos", repo="MariaIsabel/PROMISE_NFR_translated", rev="19b4ba38fa2ae29a2767a85b778e0a3ebac6d25a", licencia_v2="cc-by-4.0",
         url_ficha="https://huggingface.co/datasets/MariaIsabel/PROMISE_NFR_translated", url_origen="http://promise.site.uottawa.ca/SERepository/datasets-page.html",
         url_origen_extra="https://zenodo.org/records/268542 (de memoria: copia PROMISE_exp en Zenodo; sin confirmar)",
         patron_origen=r"CC[ -]?BY|Creative Commons|public domain|licen[cs]e"),
    dict(nombre="ropa-resenas", repo="chibifire/kaggle-womens-ecom-clothing-reviews", rev="e84927fc6858015e012e01b252065d428e381def", licencia_v2="cc0-1.0",
         url_ficha="https://huggingface.co/datasets/chibifire/kaggle-womens-ecom-clothing-reviews",
         url_origen="https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews", patron_origen=r"CC0|public domain"),
    dict(nombre="rtp-toxicidad", repo="allenai/real-toxicity-prompts", rev="f21629712ffd6a3d13a54fd2807ccd521c55ef74", licencia_v2="apache-2.0",
         url_ficha="https://huggingface.co/datasets/allenai/real-toxicity-prompts", url_origen="https://github.com/allenai/real-toxicity-prompts",
         url_origen_extra="https://skylion007.github.io/OpenWebTextCorpus/ (OpenWebText, de memoria); términos de Perspective API en https://developers.perspectiveapi.com/ (sin confirmar)",
         patron_origen=r"Apache License"),
    dict(nombre="cfpb-quejas", repo="BEE-spoke-data/consumer-finance-complaints", rev="088cc7308d4afc2a880f2329d08e2f7a09188ec6", licencia_v2="cc0-1.0",
         url_ficha="https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints",
         url_origen="https://www.consumerfinance.gov/data-research/consumer-complaints/",
         url_origen_extra="https://www.consumerfinance.gov/complaint/data-use/ (de memoria: términos de uso de los datos; sin confirmar)",
         patron_origen=r"public domain|open government|data use|terms"),
]


def norm(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def sha256_fichero(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


class EsquemaError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------- manifiesto
def entrada_inicial(c):
    f = c["fuente"]
    e = dict(nombre=c["nombre"], estado="candidata", tipo=c["tipo"], idioma=c["idioma"], dominio=c["dominio"], familia=c["familia"],
             fuente={k: v for k, v in f.items()}, rev=None, url_ficha=c["url_ficha"], url_ficha_confirmada=c.get("url_ficha_confirmada", f["tipo"] == "hf"),
             licencia_esperada=c["licencia_esperada"], licencia_estado=LICENCIA_PENDIENTE, licencia=None, licencia_sha256="", licencia_motivo="",
             bloque=c.get("bloque", "cuentan"), motivo_bloque=c.get("motivo_bloque", ""), reemplazo_de=c.get("reemplazo_de", ""),
             sintetico=c["sintetico"], clases_esperadas=c["clases"], confianza_id=c["confianza_id"], limpieza_esperada=c["limpieza_esperada"],
             no_solapa=c["no_solapa"], riesgos=c["riesgos"], pregunta=c["pregunta"], lector=c["lector"],
             fichero_soporte="soporte/%s.jsonl" % c["nombre"], fichero_test="sellada/%s.jsonl" % c["nombre"], sha256="", sha256_soporte="", ficheros_origen_sha256={},
             n_soporte=None, n_test=None, clases=None, cifras=None, solapes=None, motivo_fuera="")
    for k in ("licencia_regex_ficha", "licencia_fichero_en_descarga"):
        if k in c:
            e[k] = c[k]
    return e


def solape_no_comprobado_inicial():
    """Fuentes de entrenamiento cuyo solape NO se comprueba en el ud (no son públicas o no se bajan): se comprueba en la máquina de Juan, solo con hashes."""
    from model.datos import bateria_v3_huellas as hue
    return hue.clasificar_entrenamiento(hue.cargar_entrenamiento())["no_comprobado"]


def manifiesto_inicial():
    return dict(
        version=3, fase="1 (sin red: diseño y constructor; nada descargado ni verificado)",
        descripcion="Batería v3: SOLO MEDIR, nunca entrenar. Una tarea por familia; soporte de k=%d por clase; test <= %d; texto <= %d caracteres. Sellada: el test solo se lee con abrir_sellada."
                    % (K, TOPE_TEST, TOPE_CAR),
        semilla=SEMILLA, k=K, tope_test=TOPE_TEST, tope_caracteres=TOPE_CAR, min_test_por_tarea=MIN_TEST_TAREA,
        licencias_admitidas=sorted(set(lic.ADMITIDAS.values())),
        licencias_fuera="-SA, -NC, -ND, licencia ausente/ambigua/contradictoria, texto de Twitter/X o Reddit, y todo lo que no se pueda verificar",
        solapes=dict(metodo="hash sha256 del texto normalizado + sketch bottom-%d de n-gramas de %d palabras (Jaccard estimado)" % (sol.K_SKETCH, sol.N_GRAMA),
                     umbral_descartar=sol.UMBRAL_DESCARTE, umbral_marcar=sol.UMBRAL_MARCA, contra=["v2 (dev y selladas)", "entrenamiento", "las demás tareas de v3"],
                     huellas_sha256={}),
        bloques=dict(cuentan="entra en el agregado y en las hipótesis selladas",
                     informativo="benchmark público, probable contaminación de L3: se construye y se sella igual, pero NO cuenta en el agregado ni en las hipótesis selladas",
                     n_bajo="menos de %d tareas construidas que cuenten (por tipo y en español en total): se informa con la marca «n bajo»" % N_BAJO),
        solape_no_comprobado=solape_no_comprobado_inicial(),
        procedimiento_abrir_sellada=PROCEDIMIENTO_ABRIR_SELLADA,
        aperturas="model/datos/bateria-v3-aperturas.jsonl",
        tareas=[entrada_inicial(c) for c in CANDIDATAS],
        descartadas=[dict(fuente=a, motivo=b, comprobado="de memoria, sin red") for a, b in DESCARTADAS],
        reservas_v2=[dict(r, estado="por verificar en fase 2 (paso reservas-v2)") for r in RESERVAS_V2],
    )


def cargar_manifiesto(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def guardar_json(ruta, d):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------- descargas
def _espejo():
    return os.environ.get("CHISPA_V3_ESPEJO")      # solo pruebas en seco: carpeta con hf/<owner>__<repo>/ en vez de la red


def _cumple(nombre, patrones):
    return any(fnmatch.fnmatch(nombre, p) for p in patrones)


def hf_resolver(repo, rev=None, patrones=None, api=None):
    """(sha de la revisión, bytes de lo que casa con los patrones). Sin `rev` toma la cabecera de HF en este momento: el sha queda en el manifiesto."""
    if _espejo():
        d = Path(_espejo()) / "hf" / repo.replace("/", "__")
        if not d.is_dir():
            raise FileNotFoundError("el espejo no trae %s" % repo)
        n = sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and (not patrones or _cumple(str(p.relative_to(d)), patrones)))
        return rev or "espejo-" + repo.replace("/", "__"), n
    from huggingface_hub import HfApi
    info = (api or HfApi()).dataset_info(repo, revision=rev, files_metadata=True)
    if getattr(info, "gated", False):
        raise EsquemaError("acceso restringido (gated): no se baja")
    n = sum((s.size or 0) for s in info.siblings if not patrones or _cumple(s.rfilename, patrones))
    return info.sha, n


def hf_listar(repo, rev, patrones):
    """[(ruta relativa, bytes)] de los ficheros del repo que casan con los patrones (para bajar fuentes enormes de uno en uno)."""
    if _espejo():
        d = Path(_espejo()) / "hf" / repo.replace("/", "__")
        return [(str(p.relative_to(d)), p.stat().st_size) for p in sorted(d.rglob("*")) if p.is_file() and _cumple(str(p.relative_to(d)), patrones)]
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo, revision=rev, files_metadata=True)
    return sorted((s.rfilename, s.size or 0) for s in info.siblings if _cumple(s.rfilename, patrones))


def hf_bajar(repo, rev, patrones, destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    if _espejo():
        d = Path(_espejo()) / "hf" / repo.replace("/", "__")
        for p in sorted(d.rglob("*")):
            r = str(p.relative_to(d))
            if p.is_file() and _cumple(r, patrones):
                (destino / r).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, destino / r)
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=patrones, local_dir=str(destino))


def url_leer(url, tope=None, destino=None, tope_descarga=None):
    """Descarga `url` (http, https o file) en `destino` por trozos y devuelve (sha256, bytes); sin destino devuelve el contenido (con `tope` de bytes)."""
    req = urllib.request.Request(url, headers={"User-Agent": "chispa-bateria-v3/1"})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        if destino is None:
            data = r.read(tope or (1 << 22))
            return data
        with open(destino, "wb") as out:
            for b in iter(lambda: r.read(1 << 20), b""):
                out.write(b)
                h.update(b)
                n += len(b)
                if tope_descarga and n > tope_descarga:
                    raise EsquemaError("la descarga pasa del tope de la tarea (%.1f GB): se corta" % (tope_descarga / 2 ** 30))
    return h.hexdigest(), n


def url_resolver(f):
    """URL concreta de una fuente por URL: resuelve el `{rev}` (sha de commit) y el índice de directorio si los hay. -> (url, rev)"""
    url, rev = f["url"], None
    if f.get("resolver_rev"):
        j = json.loads(url_leer(f["resolver_rev"]["url"]).decode("utf-8"))
        rev = j[f["resolver_rev"]["campo"]]
        url = url.replace("{rev}", rev)
    if f.get("indice"):
        html = url_leer(url).decode("utf-8", "replace")
        nombres = sorted(set(re.findall(r'href="([^"?]+)"', html)))
        ok = [n for n in nombres if re.fullmatch(f["patron"], n)]
        if not ok:
            raise FileNotFoundError("el índice %s no trae ningún fichero que case con /%s/" % (url, f["patron"]))
        url = url.rstrip("/") + "/" + ok[-1]
    return url, rev


def ckan_api(f, accion, **params):
    """Llamada a la API CKAN del catálogo oficial de la fuente (por defecto catalog.data.gov) -> `result`. En pruebas en seco lee <espejo>/ckan/<id>.json (o _busqueda.json)."""
    if _espejo():
        ruta = Path(_espejo()) / "ckan" / (re.sub(r"[^\w.\-]", "_", params["id"]) if accion == "package_show" else "_busqueda")
        ruta = ruta.with_name(ruta.name + ".json")
        if not ruta.exists():
            raise FileNotFoundError("el espejo no trae %s" % ruta.name)
        return json.loads(ruta.read_text(encoding="utf-8"))["result"]
    url = "%s/api/3/action/%s?%s" % (f.get("catalogo", CKAN_DEFECTO).rstrip("/"), accion, urllib.parse.urlencode(params))
    j = json.loads(url_leer(url, tope=1 << 25).decode("utf-8"))
    if not j.get("success"):
        raise EsquemaError("la API del catálogo no devolvió resultado (%s)" % accion)
    return j["result"]


def ckan_recurso(pkg, f):
    """Recurso descargable del paquete: de los `resources` con URL http(s) cuya URL casa con `recurso_regex`, el primero del catálogo (o el último por URL con `elegir`=«ultimo»)."""
    rx = re.compile(f.get("recurso_regex") or ".", re.I)
    rs = [r for r in pkg.get("resources") or [] if str(r.get("url") or "").startswith(("http://", "https://", "file://")) and rx.search(r["url"])]
    if rs and f.get("elegir") == "ultimo":
        rs = sorted(rs, key=lambda r: r["url"])[-1:]
    return rs[0] if rs else None


def ckan_paquete(f):
    """Paquete del catálogo: por nombre (`paquete`) y, si no existe o no tiene recurso que case, por búsqueda (`busqueda`, el primero por relevancia cuyo título case con `titulo_regex`
    y que tenga recurso). Lanza EsquemaError (motivo de la tarea fuera) si no hay paquete o recurso descargable."""
    motivos, pkg = [], None
    if f.get("paquete"):
        try:
            pkg = ckan_api(f, "package_show", id=f["paquete"])
            if not ckan_recurso(pkg, f):
                motivos.append("el paquete %s no trae recurso que case con /%s/" % (f["paquete"], f.get("recurso_regex")))
                pkg = None
        except Exception as e:                                # noqa: BLE001 - paquete que no existe: se prueba la búsqueda
            motivos.append("package_show %s: %s: %s" % (f["paquete"], type(e).__name__, str(e)[:100]))
    if pkg is None and f.get("busqueda"):
        try:
            rt = re.compile(f.get("titulo_regex") or ".", re.I)
            for p in ckan_api(f, "package_search", q=f["busqueda"], rows=20).get("results") or []:
                if rt.search(p.get("title") or "") and ckan_recurso(p, f):
                    pkg = p
                    break
            if pkg is None:
                motivos.append("package_search «%s»: ningún paquete con título /%s/ y recurso /%s/" % (f["busqueda"], f.get("titulo_regex"), f.get("recurso_regex")))
        except Exception as e:                                # noqa: BLE001
            motivos.append("package_search: %s: %s" % (type(e).__name__, str(e)[:100]))
    if pkg is None:
        raise EsquemaError("catálogo: " + "; ".join(motivos))
    return pkg


def ckan_descarga(pkg, rec, f):
    """URL de descarga: la del recurso del catálogo; si el recurso es un directorio (`indice` + `patron`) se resuelve el fichero que casa; si el recurso es una API o una página y la fuente
    trae `descarga`, se usa esa URL solo si es del MISMO dominio que algún recurso del catálogo (la agencia que lo publica)."""
    base = f.get("descarga") or rec["url"]
    if f.get("descarga"):
        dominios = {urllib.parse.urlparse(r.get("url") or "").netloc for r in pkg.get("resources") or []}
        if urllib.parse.urlparse(base).netloc not in dominios:
            raise EsquemaError("la URL de descarga %s no es de un dominio de los recursos del catálogo %s" % (base, sorted(dominios)))
    return url_resolver(dict(url=base, indice=True, patron=f["patron"]))[0] if f.get("indice") else base


def ckan_ficha(pkg, rec, f, descarga):
    ex = {e.get("key"): e.get("value") for e in pkg.get("extras") or [] if e.get("key") in ("accessLevel", "rights", "license", "licenseUrl")}
    return dict(catalogo=f.get("catalogo", CKAN_DEFECTO), paquete=pkg.get("name"), id=pkg.get("id"), titulo=pkg.get("title"), organizacion=(pkg.get("organization") or {}).get("title"),
                license_id=pkg.get("license_id"), license_title=pkg.get("license_title"), license_url=pkg.get("license_url"), metadata_modified=pkg.get("metadata_modified"),
                extras=ex, recurso=dict(url=rec["url"], formato=rec.get("format"), nombre=rec.get("name")), descarga=descarga)


def bajar_ficha(t, carpeta):
    """Baja solo la ficha de licencia de la tarea (README.md o página) y devuelve su texto; no descarga datos."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    f = t["fuente"]
    if f["tipo"] == "ckan":                       # licencia y descarga salen del registro oficial del catálogo (no de la web de la agencia)
        try:
            pkg = ckan_paquete(f)
        except EsquemaError as e:
            t["_error_ficha"] = str(e)
            return ""
        rec = ckan_recurso(pkg, f)
        try:
            t["recurso_url"] = ckan_descarga(pkg, rec, f)
        except Exception as e:                                # noqa: BLE001 - sin URL de descarga: la tarea queda fuera con su motivo
            t["_error_ficha"] = "catálogo: sin URL de descarga (%s: %s)" % (type(e).__name__, str(e)[:150])
            return ""
        txt = json.dumps(ckan_ficha(pkg, rec, f, t["recurso_url"]), ensure_ascii=False, indent=1, sort_keys=True)
        (carpeta / "ficha").mkdir(exist_ok=True)
        (carpeta / "ficha" / "ficha.json").write_text(txt, encoding="utf-8")
        return txt
    if f["tipo"] == "hf":
        sha, _ = hf_resolver(f["repo"], t.get("rev"), ["README.md"])
        t["rev"] = t.get("rev") or sha
        hf_bajar(f["repo"], t["rev"], ["README.md"], carpeta / "ficha")
        return (carpeta / "ficha" / "README.md").read_text(encoding="utf-8", errors="replace") if (carpeta / "ficha" / "README.md").exists() else ""
    try:
        txt = url_leer(t["url_ficha"], tope=1 << 21).decode("utf-8", "replace")
    except Exception as e:                         # noqa: BLE001 - cualquier fallo de red deja la tarea fuera con su motivo
        t["_error_ficha"] = "%s: %s" % (type(e).__name__, e)
        return ""
    (carpeta / "ficha").mkdir(exist_ok=True)
    (carpeta / "ficha" / "ficha.txt").write_text(txt, encoding="utf-8")
    return txt


def bajar_datos(t, carpeta):
    """Baja los datos (con tope de tamaño) y devuelve {fichero relativo: sha256}."""
    carpeta = Path(carpeta) / "datos"
    f = t["fuente"]
    tope = int(float(f.get("max_gb", 2)) * (1 << 30))
    if f["tipo"] == "hf":
        sha, n = hf_resolver(f["repo"], t["rev"], f["patrones"])
        if n > tope:
            raise EsquemaError("la descarga previsible (%.1f GB) pasa del tope de la tarea (%.1f GB)" % (n / 2 ** 30, tope / 2 ** 30))
        hf_bajar(f["repo"], t["rev"], f["patrones"], carpeta)
    else:
        carpeta.mkdir(parents=True, exist_ok=True)
        url, rev = (t["recurso_url"], None) if f["tipo"] == "ckan" else url_resolver(f)
        destino = carpeta / (re.sub(r"[^\w.\-]", "_", url.split("/")[-1].split("?")[-1]) or "descarga.bin")
        if destino.suffix == "" and not destino.name.endswith(".zip"):
            destino = destino.with_name(destino.name + ".bin")
        h, n = url_leer(url, destino=destino, tope_descarga=tope)
        t["rev"] = rev or "sha256:" + h
        if zipfile.is_zipfile(destino) and f.get("descomprimir"):
            with zipfile.ZipFile(destino) as z:
                z.extractall(carpeta)
            destino.unlink()
    out = {}
    for p in sorted(Path(carpeta).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(carpeta))] = sha256_fichero(p)
    if not out:
        raise EsquemaError("sin ficheros de datos tras la descarga (¿repo de solo script o patrones que no casan?)")
    return out


# ---------------------------------------------------------------------------------------------------------------- lectores
def _get(row, ruta):
    """Valor de una ruta con puntos en un dict anidado; None si falta."""
    v = row
    for p in ruta.split("."):
        if isinstance(v, dict) and p in v:
            v = v[p]
        elif isinstance(v, list) and p.isdigit() and int(p) < len(v):
            v = v[int(p)]
        else:
            return None
    return v


def _a_texto(v):
    if v is None:
        return ""
    if isinstance(v, dict):                          # un mensaje de chat ({"role", "content"}) o similar: solo su texto
        for k in ("content", "text", "texto"):
            if k in v:
                return _a_texto(v[k])
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(x for x in (_a_texto(x) for x in v) if x)
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _abrir_texto(ruta, codificacion):
    if str(ruta).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(ruta, "rb"), encoding=codificacion, errors="replace", newline="")
    return open(ruta, "r", encoding=codificacion, errors="replace", newline="")


# ---- lectores XML (tolerantes: el esquema real no se ha visto; la fase 2 lo valida y la tarea queda fuera con motivo si no encaja)
ULTIMO_SIN_MAPA = Counter()   # valores de etiqueta que el `mapa_etiqueta` no cubre (solo etiquetas cortas: nunca texto), para el motivo cuando no sale nada
ULTIMO_XML = {}          # etiquetas vistas por el último lector XML, para el motivo cuando no sale nada


def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _abrir_bin(ruta):
    r = str(ruta)
    if r.endswith(".bz2"):
        return bz2.open(r, "rb")
    if r.endswith(".gz"):
        return gzip.open(r, "rb")
    return open(r, "rb")


def _buscar(elem, ruta):
    """Primer descendiente de `elem` cuya cadena de etiquetas termina en `ruta` («A/B/C», sin distinguir mayúsculas ni espacios de nombres)."""
    segs = [x.lower() for x in ruta.split("/") if x]
    for e in elem.iter():
        if _local(e.tag).lower() != segs[0]:
            continue
        cur = [e]
        for sg in segs[1:]:
            cur = [h for c in cur for h in c if _local(h.tag).lower() == sg][:1]
            if not cur:
                break
        if cur:
            return cur[0]
    return None


def limpiar_html(t):
    """Texto de un campo XML que trae HTML escapado (p. ej. `&lt;br/&gt;`): sin etiquetas ni entidades."""
    t = html.unescape(t or "")
    t = re.sub(r"<[^>]{1,200}>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def _xml_seguro(cabecera):
    return b"<!ENTITY" not in cabecera


def _registros_xml(flujo, tag):
    """Elementos `tag` de un XML en flujo (iterparse); anota las etiquetas vistas. No se leen XML con entidades propias."""
    for _ev, e in ET.iterparse(flujo, events=("end",)):
        loc = _local(e.tag)
        ULTIMO_XML[loc] = ULTIMO_XML.get(loc, 0) + 1
        if loc.lower() == tag.lower():
            yield e
            e.clear()


def _filas_xml(ruta, lector):
    """Un registro XML -> una fila {campo: texto} con los `campos` del lector ({nombre: ruta de etiquetas}). Falta -> None."""
    with _abrir_bin(ruta) as fh:
        cab = fh.read(8192)
    if not _xml_seguro(cab):
        ULTIMO_XML["_xml_con_entidades_omitido"] = ULTIMO_XML.get("_xml_con_entidades_omitido", 0) + 1
        return
    try:
        with _abrir_bin(ruta) as fh:
            for e in _registros_xml(fh, lector["registro"]):
                fila = {}
                for nombre, ruta_et in lector["campos"].items():
                    x = _buscar(e, ruta_et)
                    fila[nombre] = limpiar_html("".join(x.itertext())) if x is not None else None
                yield fila
    except ET.ParseError:
        ULTIMO_XML["_xml_mal_formado"] = ULTIMO_XML.get("_xml_mal_formado", 0) + 1


_CAT = re.compile(r"\[\[\s*(?:categor[ií]a|category)\s*:\s*([^\]|]+?)\s*(?:\|[^\]]*)?\]\]", re.I)


def wikitexto_a_texto(w):
    """Wikitexto -> texto plano: sin plantillas, referencias, categorías, archivos, tablas, enlaces ni formato; corta en «Fuentes»/«Referencias»."""
    w = re.sub(r"<!--.*?-->", " ", w, flags=re.S)
    w = re.sub(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", " ", w, flags=re.S | re.I)
    for _ in range(6):                                                   # plantillas anidadas
        w2 = re.sub(r"\{\{[^{}]*\}\}", " ", w)
        if w2 == w:
            break
        w = w2
    w = re.sub(r"\{\|.*?\|\}", " ", w, flags=re.S)
    w = re.sub(r"\[\[\s*(?:archivo|file|imagen|image|categor[ií]a|category)\s*:(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]", " ", w, flags=re.I)
    m = re.search(r"^=+\s*(fuentes|referencias|enlaces externos|v[eé]ase tambi[eé]n|notas)\s*=+\s*$", w, re.I | re.M)
    if m:
        w = w[:m.start()]
    w = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", w)
    w = re.sub(r"\[(?:https?:)?//\S+\s*([^\]]*)\]", r"\1", w)
    w = re.sub(r"'{2,5}", "", w)
    w = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", w, flags=re.M)
    w = re.sub(r"^[*#:;]+\s*", "", w, flags=re.M)
    w = re.sub(r"<[^>]{1,200}>", " ", w)
    return re.sub(r"\s+", " ", html.unescape(w)).strip()


def _filas_mediawiki(ruta, lector):
    """Volcado XML de MediaWiki -> filas {titulo, texto, etiqueta}. Etiqueta = la ÚNICA categoría del conjunto cerrado `categorias` que trae el artículo
    (ninguna o varias -> etiqueta None: no es una fila etiquetable). Redirecciones y espacios de nombres distintos de `ns` se saltan."""
    cerrado = {c.lower(): c for c in lector["categorias"]}
    ns = str(lector.get("ns", "0"))
    with _abrir_bin(ruta) as fh:
        for e in _registros_xml(fh, "page"):
            if (_buscar(e, "ns") is not None and (_buscar(e, "ns").text or "").strip() != ns) or _buscar(e, "redirect") is not None:
                continue
            tx = _buscar(e, "revision/text")
            w = (tx.text or "") if tx is not None else ""
            if re.match(r"\s*#(redirect|redirecci[oó]n)", w, re.I):
                continue
            cats = {cerrado[c.lower()] for c in _CAT.findall(w) if c.lower() in cerrado}
            ti = _buscar(e, "title")
            yield dict(titulo=(ti.text or "") if ti is not None else "", texto=wikitexto_a_texto(w), etiqueta=cats.pop() if len(cats) == 1 else None)


def _filas_fichero(ruta, formato, lector):
    cod = lector.get("codificacion", "utf-8")
    if formato == "parquet":
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(str(ruta)).iter_batches(batch_size=2000):
            for r in b.to_pylist():
                yield r
    elif formato in ("csv", "tsv"):
        yield from _filas_csv(ruta, formato, lector, cod)
    elif formato == "jsonl":
        yield from _filas_json(ruta, cod)
    elif formato == "xml":
        yield from _filas_xml(ruta, lector)
    elif formato == "mediawiki":
        yield from _filas_mediawiki(ruta, lector)
    else:
        raise EsquemaError("formato de lector desconocido: %s" % formato)


def _filas_csv(ruta, formato, lector, cod):
    """csv/tsv con cabecera. Regla escrita del delimitador: el `separador` del lector si lo hay; si no, «,» (csv) o tabulador (tsv) y, si con eso la cabecera sale en UNA sola
    columna que contiene tabulador, «|» o «;», se vuelve a leer con el primero de ellos (un tsv guardado como .csv, un csv con «|»)."""
    csv.field_size_limit(1 << 28)
    d = lector.get("separador") or ("\t" if formato == "tsv" else ",")
    cols = lector.get("columnas")
    for intento in (0, 1):
        kw = dict(delimiter=d, quoting=csv.QUOTE_NONE) if d == "\t" else dict(delimiter=d)
        with _abrir_texto(ruta, cod) as fh:
            rd = csv.reader(fh, **kw)
            if isinstance(cols, dict):                       # sin cabecera: nombre -> posición
                for fila in rd:
                    yield {k: (fila[i] if i < len(fila) else None) for k, i in cols.items()}
                return
            cab = next(rd, None)
            if cab is None:
                return
            otro = [x for x in ("\t", "|", ";") if x in cab[0] and x != d] if len(cab) == 1 and not lector.get("separador") else []
            if otro and intento == 0:
                d = otro[0]
                continue                                    # cabecera de una sola columna con otro delimitador dentro: se reintenta
            for fila in rd:
                yield dict(zip(cab, fila))
            return


def _expandir_json(v):
    """Un JSON de un solo valor: lista de registros, o un objeto que contiene la(s) lista(s) de registros ({"data": [...]}, {"train": [...]}) o registros por clave; si no, el propio objeto."""
    if isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                yield x
    elif isinstance(v, dict):
        listas = [x for x in v.values() if isinstance(x, list) and x and all(isinstance(e, dict) for e in x)]
        if listas:
            for l in listas:
                yield from l
        elif v and all(isinstance(x, dict) for x in v.values()):
            yield from v.values()
        else:
            yield v


def _valores_json(texto):
    dec, i, n = json.JSONDecoder(), 0, len(texto)
    while True:
        while i < n and texto[i] in " \t\r\n,\ufeff":
            i += 1
        if i >= n:
            return
        v, i = dec.raw_decode(texto, i)
        yield v


def _filas_json(ruta, cod):
    """json/jsonl tolerante: un registro por línea (jsonl); o todo el fichero (lista, objeto con listas, objetos de varias líneas uno tras otro)."""
    with _abrir_texto(ruta, cod) as fh:
        lineas = (l for l in fh if l.strip())
        l1 = next(lineas, None)
        if l1 is None:
            return
        try:
            v1 = json.loads(l1)
        except ValueError:
            v1 = None
        if v1 is not None and not isinstance(v1, list):
            l2 = next(lineas, None)
            if l2 is None:                                   # un fichero de una sola línea: puede ser {"data": [...]}
                yield from _expandir_json(v1)
                return
            yield v1
            yield json.loads(l2)
            for l in lineas:
                yield json.loads(l)
            return
        valores = _valores_json(l1 + "".join(lineas))        # lista o objetos de varias líneas
        v1 = next(valores, None)
        v2 = next(valores, None)
        if v2 is None:
            yield from _expandir_json(v1)
            return
        for v in chain([v1, v2], valores):
            yield from (_expandir_json(v) if isinstance(v, list) else [v])


def _nn(s):
    """Nombre de columna normalizado: sin mayúsculas, acentos ni signos («Label 1» = «label_1» = «label1»)."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(s).lower()).encode("ascii", "ignore").decode())


def _tipo(v):
    return type(v).__name__ if v is not None else "None"


def _clave_real(nombre, fila):
    """Nombre de columna (con la ruta tras el primer punto intacta) que existe en `fila`: exacto o, si no, normalizado. Alternativas con «||»: gana la primera que exista. None si ninguna."""
    for alt in str(nombre).split("||"):
        seg, _, resto = alt.partition(".")
        real = seg if seg in fila else next((k for k in fila if _nn(k) == _nn(seg)), None)
        if real is not None:
            return real + ("." + resto if resto else "")
    return None


def _auto_columnas(auto, muestra):
    """Regla escrita para fijar texto y etiqueta cuando no se conoce el esquema: la ETIQUETA es la primera columna (en el orden del fichero) cuyo nombre casa con `etiqueta_regex`,
    no es un identificador/idioma/fecha/enlace y trae entre 2 y `max_clases` valores distintos cortos en la muestra; el TEXTO son las columnas de cadena (máx. 2, en el orden del fichero) cuya
    longitud media >= 30 caracteres y >= la mitad de la mayor (o las que casen con `texto_regex`). Devuelve (texto, etiqueta, resumen) o lanza EsquemaError con nombres y tipos."""
    fila = muestra[0]
    ni = re.compile(r"^(id|idx|index|ids|split|lang|language|idioma|url|link|date|fecha|source|fuente|author|autor|title|titulo|text|texto)\d*$")
    er = re.compile(auto.get("etiqueta_regex") or r"label|categor|class|clase|target|sentiment|sentimiento|sector|topic|tema|tipo|type|severity|score|rating|nota|capitulo|chapter|genre|genero|winner|stance")
    et = None
    for k in fila:
        if not er.search(_nn(k)) or (ni.match(_nn(k)) and not auto.get("etiqueta_regex")):
            continue
        vals = [_a_texto(r.get(k)).strip() for r in muestra if r.get(k) is not None and not isinstance(r.get(k), (dict, list))]
        d = set(vals)
        if 2 <= len(d) <= int(auto.get("max_clases", 30)) and max(len(x) for x in d) <= 40 and len(vals) >= 0.9 * len(muestra):
            et = k
            break
    if et is None:
        raise EsquemaError("auto: ninguna columna de etiqueta (2-%s valores cortos) entre %s" % (auto.get("max_clases", 30), {k: _tipo(v) for k, v in list(fila.items())[:30]}))
    if auto.get("texto_regex"):
        tx = [k for k in fila if k != et and re.search(auto["texto_regex"], _nn(k))][:2]
    else:
        med = {k: sum(len(_a_texto(r.get(k))) for r in muestra) / len(muestra) for k in fila if k != et and isinstance(fila[k], (str, list, dict))}
        top = max(med.values(), default=0)
        tx = [k for k, m in med.items() if m >= 30 and m >= top / 2][:2]
    if not tx:
        raise EsquemaError("auto: ninguna columna de texto (media >= 30 caracteres) entre %s" % {k: _tipo(v) for k, v in list(fila.items())[:30]})
    return tx, et, dict(texto=tx, etiqueta=et, valores_etiqueta=len({_a_texto(r.get(et)).strip() for r in muestra}))


def _resolver_lector(lector, muestra):
    """(lector con los nombres de columna resueltos contra el primer registro de `muestra`, faltan). Nombres exactos o normalizados; «a||b» = alternativas; `auto` fija texto y etiqueta."""
    fila, l, falta = muestra[0], dict(lector), []
    elegido = None
    if lector.get("auto") is not None:
        tx, et, elegido = _auto_columnas(lector["auto"], muestra)
        l.update(texto=tx, etiqueta=et)

    def r(n):
        real = _clave_real(n, fila)
        if real is None:
            falta.append(str(n).split("||")[0])
        return real or n

    if isinstance(l.get("texto"), str):
        l["texto"] = r(l["texto"])
    elif l.get("texto"):
        l["texto"] = [r(c) for c in l["texto"]]
    for k in ("etiqueta", "split_col"):
        if l.get(k) and not (k == "split_col" and _clave_real(l[k], fila) is None):
            l[k] = r(l[k])
    for k in ("umbral", "contar_patron", "etiqueta_par"):
        if l.get(k):
            l[k] = dict(l[k], col=r(l[k]["col"]))
    if l.get("pares"):
        p = dict(l["pares"])
        p["bueno"], p["malo"] = r(p["bueno"]), r(p["malo"])
        p["contexto"] = [r(c) for c in p.get("contexto") or []]
        l["pares"] = p
    if l.get("filtro"):
        l["filtro"] = {r(c): v for c, v in l["filtro"].items()}
    return l, sorted(set(falta)), elegido


def _formato_de(ruta):
    n = str(ruta).lower()
    for suf in (".gz", ".bz2"):
        if n.endswith(suf):
            n = n[:-len(suf)]
    return {"parquet": "parquet", "csv": "csv", "tsv": "tsv", "txt": "csv", "json": "jsonl", "jsonl": "jsonl", "ndjson": "jsonl"}.get(n.rsplit(".", 1)[-1])



def _filas_zip(zpath, patrones, lista=None):
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith("/") or not _cumple(n, patrones):
                continue
            with z.open(n) as fh:
                try:
                    d = json.load(fh)
                except ValueError:
                    continue
            if lista:                              # un fichero con muchos registros bajo la clave `lista`
                for r in (_get(d, lista) or []):
                    yield n, r
            else:
                yield n, d


def _ficheros(carpeta, patrones):
    carpeta = Path(carpeta)
    vistos, out = set(), []
    for p in sorted(carpeta.rglob("*")):
        if p.is_file():
            r = str(p.relative_to(carpeta))
            if r not in vistos and any(fnmatch.fnmatch(r, g) or fnmatch.fnmatch(p.name, g) or fnmatch.fnmatch(r, g.replace("**/", "")) for g in patrones):
                vistos.add(r)
                out.append((r, p))
    return out


def _split_de(nombre, fila, lector):
    for s, pat in (lector.get("split_por_fichero") or {}).items():
        if re.search(pat, nombre):
            return s
    c = lector.get("split_col")
    if c and fila.get(c) is not None:
        v = str(fila[c]).lower()
        return "train" if v in ("train", "training") else "test" if v in ("test", "dev", "validation", "val", "valid") else None
    return None


class Reservorio:
    """Guarda como mucho `tope` ítems por etiqueta (muestreo uniforme con semilla); sin tope guarda todo."""

    def __init__(self, tope, rng):
        self.tope, self.rng, self.vistos, self.items = tope, rng, Counter(), defaultdict(list)

    def añadir(self, clave, it):
        self.vistos[clave] += 1
        l = self.items[clave]
        if not self.tope or len(l) < self.tope:
            l.append(it)
        else:
            j = self.rng.randrange(self.vistos[clave])
            if j < self.tope:
                l[j] = it

    def todos(self):
        return [i for c in sorted(self.items) for i in self.items[c]]


def extraer_items(t, carpeta, semilla=SEMILLA):
    """Lee los datos de una tarea con su `lector` -> (items, cifras). Un ítem: dict(texto, etiqueta, split, grupo?). Lanza EsquemaError con el motivo."""
    lector = t["lector"]
    if lector.get("pendiente"):
        raise EsquemaError("pendiente de lector: %s" % lector["pendiente"])
    carpeta = Path(carpeta) / "datos"
    rng = random.Random("%s-%d-lectura" % (t["nombre"], semilla))
    res = Reservorio(lector.get("muestra_por_clase"), rng)
    cif = Counter()
    formato = lector["formato"]
    ULTIMO_XML.clear()
    ULTIMO_SIN_MAPA.clear()
    omitidos = {}                    # fichero -> columnas y tipos (nunca texto) de los ficheros que no traen el esquema pedido

    def fuentes_zip():
        zips = sorted(carpeta.rglob("*.zip"))
        if not zips:
            raise EsquemaError("sin zip en la descarga")
        it = ((n, f) for z in zips for n, f in _filas_zip(z, lector["ficheros"], lector.get("lista")))
        primero = next(it, None)
        if primero is not None:
            lec, falta, elegido = _resolver_lector(lector, [primero[1]])
            if falta:
                omitidos[primero[0]] = {k: _tipo(v) for k, v in list(primero[1].items())[:40]}
            else:
                t["esquema_elegido"] = elegido
                yield lec, primero[0], chain([primero[1]], (f for _, f in it))

    def fuentes_ficheros():
        fl = _ficheros(carpeta, lector["ficheros"])
        if not fl:
            raise EsquemaError("ningún fichero casa con %s" % lector["ficheros"])
        for r, p in fl:
            fm = _formato_de(p) if formato == "auto" else formato
            if fm is None:
                omitidos[r] = "formato no reconocido"
                continue
            it = _filas_fichero(p, fm, lector)
            muestra = list(islice(it, 2000 if lector.get("auto") is not None else 1))
            if not muestra:
                continue
            if fm in ("xml", "mediawiki"):
                lec, falta, elegido = lector, [], None
            else:
                try:
                    lec, falta, elegido = _resolver_lector(lector, muestra)
                except EsquemaError as e:
                    omitidos[r] = str(e)
                    continue
            if falta:
                omitidos[r] = {k: _tipo(v) for k, v in list(muestra[0].items())[:40]}
                continue
            if elegido:
                t.setdefault("esquema_elegido", {})[r] = elegido
            yield lec, r, chain(muestra, it)

    for lec, nombre, filas in (fuentes_zip() if formato == "json_zip" else fuentes_ficheros()):
        for fila in filas:
            cif["filas_leidas"] += 1
            if any(_a_texto(_get(fila, col)).strip().lower() not in [str(v).lower() for v in vals] for col, vals in (lec.get("filtro") or {}).items()):
                cif["filtradas"] += 1
                continue
            split = _split_de(nombre, fila, lec)
            for texto, et, grupo in _items_de_fila(lec, fila, cif, nombre):
                if et is None or not (texto or "").strip():
                    cif["sin_texto_o_etiqueta"] += 1
                    continue
                res.añadir("_" if lec.get("cuantiles") else str(et), dict(texto=texto, etiqueta=et, split=split, grupo=grupo))
    if omitidos and not res.vistos:
        raise EsquemaError("faltan columnas esperadas en todos los ficheros; columnas y tipos que traen (primeros %d): %s" % (min(len(omitidos), 6), dict(list(omitidos.items())[:6])))
    if omitidos:
        cif["ficheros_omitidos_por_esquema"] = len(omitidos)
    items = res.todos()
    if not items:
        extra = ""
        if formato in ("xml", "mediawiki"):
            extra = "; etiquetas XML vistas: %s" % ", ".join("%s=%d" % kv for kv in sorted(ULTIMO_XML.items(), key=lambda kv: -kv[1])[:15])
        if ULTIMO_SIN_MAPA:
            extra += "; valores de etiqueta sin mapa (más frecuentes): %s" % ", ".join("%r=%d" % kv for kv in ULTIMO_SIN_MAPA.most_common(8))
        raise EsquemaError("no salió ninguna fila utilizable (leídas %d, filtradas %d, sin texto o etiqueta %d)%s" % (cif["filas_leidas"], cif["filtradas"], cif["sin_texto_o_etiqueta"], extra))
    if lector.get("cuantiles"):
        _a_cuantiles(items, int(lector["cuantiles"]))
    return items, dict(cif)


def _etiqueta(lector, fila):
    if lector.get("umbral"):
        u = lector["umbral"]
        try:
            v = float(_get(fila, u["col"]))
        except (TypeError, ValueError):
            return None
        return "si" if v >= u["si"] else "no" if v <= u["no"] else None
    if lector.get("contar_patron"):
        c = lector["contar_patron"]
        n = _a_texto(_get(fila, c["col"])).count(c["patron"])
        return str(min(max(n, c["min"]), c["max"]) - c["min"] + 1)
    if lector.get("etiqueta_par"):
        p = lector["etiqueta_par"]
        d = _get(fila, p["col"]) or {}
        nombres, valores = d.get("name") or [], d.get("value") or []
        if p["nombre"] not in nombres:
            return None
        v = valores[nombres.index(p["nombre"])]
        return float(v) if v is not None else None
    v = _get(fila, lector["etiqueta"])
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            return None
        v = v[0]
    if v is None or (isinstance(v, str) and not v.strip()):
        if lector.get("vacio_es") is None:
            return None
        v = lector["vacio_es"]
    if isinstance(v, bool):
        v = str(v)
    elif isinstance(v, float) and v == int(v) and not lector.get("cuantiles"):
        v = str(int(v))
    if lector.get("cuantiles"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    v = str(v).strip()
    if lector.get("prefijo_hasta"):
        v = v.split(lector["prefijo_hasta"])[0].strip()
    m = lector.get("mapa_etiqueta")
    if m is not None:
        r = m.get(v)
        if r is None:                                    # tolerante a mayúsculas y signos: «Fake» = «fake»
            r = next((x for k, x in m.items() if _nn(k) == _nn(v) and _nn(v)), None)
        if r is None and len(v) <= 40 and len(ULTIMO_SIN_MAPA) < 200:
            ULTIMO_SIN_MAPA[v] += 1
        return r
    return v or None


def _texto(lector, fila):
    t = lector.get("texto")
    cols = [t] if isinstance(t, str) else list(t or [])
    return "\n".join(x for x in (_a_texto(_get(fila, c)).strip() for c in cols) if x)


def _items_de_fila(lector, fila, cif, nombre=""):
    if lector.get("pares"):
        p = lector["pares"]
        ctx = "\n".join(_a_texto(_get(fila, c)).strip() for c in (p.get("contexto") or []))
        g = "p%d" % cif["filas_leidas"]
        for col, et in ((p["bueno"], "si"), (p["malo"], "no")):
            yield ((ctx + "\n" if ctx else "") + _a_texto(_get(fila, col)).strip(), et, g)
        return
    if lector.get("etiqueta_desde_carpeta"):
        carpeta = nombre.split("/")[0] if "/" in nombre else None
        yield _texto(lector, fila), (lector.get("mapa_etiqueta") or {}).get(carpeta, carpeta), None
        return
    yield _texto(lector, fila), _etiqueta(lector, fila), None


def _a_cuantiles(items, n):
    vals = sorted(i["etiqueta"] for i in items)
    cortes = [vals[min(len(vals) - 1, (len(vals) * q) // n)] for q in range(1, n)]
    for i in items:
        i["etiqueta"] = str(1 + bisect_right(cortes, i["etiqueta"]) if cortes else 1)
        i["etiqueta"] = str(min(int(i["etiqueta"]), n))


# ---------------------------------------------------------------------------------------------------------------- armado
def _estratificar(filas, tope, rng):
    if len(filas) <= tope:
        return list(filas)
    por = defaultdict(list)
    for f in filas:
        por[f["etiqueta"]].append(f)
    for l in por.values():
        rng.shuffle(l)
    out, claves = [], sorted(por)
    while len(out) < tope and any(por.values()):
        for c in claves:
            if por[c] and len(out) < tope:
                out.append(por[c].pop())
    return out


def armar(t, items, solape=None, semilla=SEMILLA, k=K, tope=TOPE_TEST):
    """items -> (soporte, test, cifras). `solape(i)` opcional devuelve 'limpio' | 'marcar' | 'descartar' para el ítem i del índice (posición en `items`)."""
    rng = random.Random("%s-%d" % (t["nombre"], semilla))
    lector = t["lector"]
    cif = dict(n_origen=len(items))
    ok = []
    for n, it in enumerate(items):
        x = dict(it, _n=n)
        x["texto"] = x["texto"].strip()
        if len(x["texto"]) > TOPE_CAR:
            x["texto"] = x["texto"][:TOPE_CAR]
            cif["cortados"] = cif.get("cortados", 0) + 1
        ok.append(x)
    # clases: las más frecuentes y con un mínimo
    cuenta = Counter(i["etiqueta"] for i in ok)
    cl = [c for c, n in cuenta.most_common() if n >= int(lector.get("min_por_clase", 0))]
    if lector.get("clases_max"):
        cl = cl[: int(lector["clases_max"])]
    cif["clases_quitadas"] = len(cuenta) - len(cl)
    ok = [i for i in ok if i["etiqueta"] in cl]
    # textos con dos etiquetas: fuera; duplicados exactos: uno solo
    et = defaultdict(set)
    for i in ok:
        et[norm(i["texto"])].add(i["etiqueta"])
    vistos, lim = set(), []
    cif["conflicto"] = cif["duplicados"] = 0
    for i in ok:
        n = norm(i["texto"])
        if len(et[n]) > 1:
            cif["conflicto"] += 1
        elif n in vistos:
            cif["duplicados"] += 1
        else:
            vistos.add(n)
            lim.append(i)
    ok = lim
    if solape:
        cif["solape_descartadas"] = cif["solape_marcadas"] = 0
        lim = []
        for i in ok:
            v = solape(i["_n"])
            if v == "descartar":
                cif["solape_descartadas"] += 1
                continue
            if v == "marcar":
                cif["solape_marcadas"] += 1
                i["marca_solape"] = True
            lim.append(i)
        ok = lim
    rng.shuffle(ok)
    pool_sop = [i for i in ok if i.get("split") == "train"]
    pool_ev = [i for i in ok if i.get("split") == "test"]
    tiene = bool(pool_sop and pool_ev)         # con split oficial solo si trae los dos; si no, partición propia con semilla
    if not tiene:
        pool_sop, pool_ev = [], []
    sop, cont = [], Counter()
    if tiene:
        for i in pool_sop:
            if cont[i["etiqueta"]] < k:
                cont[i["etiqueta"]] += 1
                sop.append(i)
    else:
        grupos, orden = defaultdict(list), []
        for i in ok:                                   # los miembros de un grupo (p. ej. un par bueno/malo) van siempre del mismo lado
            g = i.get("grupo") if i.get("grupo") is not None else "_%d" % i["_n"]
            if g not in grupos:
                orden.append(g)
            grupos[g].append(i)
        for g in orden:
            if any(cont[i["etiqueta"]] < k for i in grupos[g]):
                for i in grupos[g]:
                    cont[i["etiqueta"]] += 1
                    sop.append(i)
            else:
                pool_ev += grupos[g]
    textos_sop = {norm(i["texto"]) for i in sop}
    usados = {i["_n"] for i in sop}
    ev = [i for i in pool_ev if i["_n"] not in usados and norm(i["texto"]) not in textos_sop]
    cif["en_soporte_quitados"] = len(pool_ev) - len(ev)
    if t["tipo"] == "score":              # niveles 1..K contiguos con test: se recorta al prefijo contiguo
        pres = {i["etiqueta"] for i in ev}
        K_ = 0
        while str(K_ + 1) in pres:
            K_ += 1
        clases = [str(i) for i in range(1, K_ + 1)]
    elif t["tipo"] == "yesno":
        clases = ["si", "no"]
    else:
        pres = {i["etiqueta"] for i in ev}
        clases = sorted(c for c in cl if c in pres)
    ev = [i for i in ev if i["etiqueta"] in clases]
    sop = [i for i in sop if i["etiqueta"] in clases]
    ev = _estratificar(ev, tope, rng)
    cif["clases"] = clases
    cif["soporte_por_clase"] = dict(Counter(i["etiqueta"] for i in sop))
    cif["test_por_clase"] = dict(Counter(i["etiqueta"] for i in ev))
    cif["clases_con_menos_de_k"] = sorted(c for c in clases if cif["soporte_por_clase"].get(c, 0) < k)
    return sop, ev, cif


def a_fila(t, i, split, n, licencia, repo):
    clases = t["_clases"]
    f = {"id": "%s:%s:%d" % (t["nombre"], split, n), "tarea": t["nombre"], "tipo": t["tipo"], "pregunta": t["pregunta"], "texto": i["texto"], "opciones": clases,
         "descripciones": {c: c for c in clases}, "etiqueta": i["etiqueta"], "idioma": t["idioma"], "split": split, "fuente": repo, "licencia": licencia, "uso_comercial": True}
    if i.get("marca_solape"):
        f["marca_solape"] = True
    return f


def escribir_jsonl(ruta, filas, solo_lectura=False):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        ruta.chmod(0o644)
    with open(ruta, "w", encoding="utf-8") as fh:
        for f in filas:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    if solo_lectura:
        ruta.chmod(0o444)
    return sha256_fichero(ruta)


# ---------------------------------------------------------------------------------------------------------------- orquestación
def _fuera(t, motivo, **extra):
    t.update(estado="fuera", motivo_fuera=motivo, **extra)
    log("FUERA  %-26s %s" % (t["nombre"], motivo[:200]))


def evaluar_licencia_tarea(t, texto_ficha, trabajo):
    f = t["fuente"]
    if f["tipo"] == "ckan":
        if t.get("_error_ficha"):
            return dict(estado="fuera", licencia="", motivo=t["_error_ficha"], avisos=[], sha256_texto="")
        r = lic.evaluar_ckan(json.loads(texto_ficha))
        if r["estado"] == "admitida":
            r["avisos"].append("licencia tomada del registro de %s (license_id %s)" % (f.get("catalogo", CKAN_DEFECTO), json.loads(texto_ficha)["license_id"]))
            t["ckan_license_id"] = json.loads(texto_ficha)["license_id"]
    elif f["tipo"] == "hf":
        r = lic.evaluar_tarjeta(texto_ficha, esperada=t["licencia_esperada"])
    else:
        if t.get("_error_ficha"):
            return dict(estado="fuera", licencia="", motivo="no se pudo bajar la ficha %s (%s)" % (t["url_ficha"], t["_error_ficha"]), avisos=[], sha256_texto="")
        r = lic.evaluar_ficha(texto_ficha, t.get("licencia_regex_ficha"), lic.normalizar(t["licencia_esperada"]) if lic.normalizar(t["licencia_esperada"]) in lic.ADMITIDAS else "dominio-publico")
        if r["estado"] == "admitida":
            r["licencia"] = lic.normalizar(t["licencia_esperada"])
    return r


def _guardar_licencia(t, r, texto, salida):
    d = Path(salida) / "licencias"
    d.mkdir(parents=True, exist_ok=True)
    (d / (t["nombre"] + ".txt")).write_text(texto or "", encoding="utf-8")
    t.update(licencia_sha256=lic.sha256_texto(texto or ""), licencia=r["licencia"] or None, licencia_motivo=r["motivo"] or "; ".join(r["avisos"]),
             licencia_estado="verificada en fase 2 (automática: admitida)" if r["estado"] == "admitida" else "fuera: %s" % r["motivo"])


def construir(manifiesto, trabajo, salida, huellas=None, solo=None, k=K, tope=TOPE_TEST):
    """Pipeline completo de la fase 2. Devuelve el manifiesto actualizado (también escrito en salida/manifiesto.json)."""
    man = cargar_manifiesto(manifiesto)
    trabajo, salida = Path(trabajo), Path(salida)
    admitidas, pool = [], {}
    for t in man["tareas"]:
        if solo and t["nombre"] not in solo:
            continue
        carp = trabajo / t["nombre"]
        if t["lector"].get("pendiente"):
            _fuera(t, "pendiente de lector: %s" % t["lector"]["pendiente"])
            t["estado"] = "pendiente"
            continue
        try:
            texto = bajar_ficha(t, carp)
            if not t.get("licencia_fichero_en_descarga"):
                r = evaluar_licencia_tarea(t, texto, trabajo)
                _guardar_licencia(t, r, texto, salida)
                if r["estado"] != "admitida":
                    _fuera(t, "licencia: " + r["motivo"])
                    continue
            t["ficheros_origen_sha256"] = bajar_datos(t, carp)
            if t.get("licencia_fichero_en_descarga"):         # la licencia está dentro de la descarga (zip): se comprueba antes de leer los datos
                texto2 = _licencia_en_zip(carp / "datos", t["licencia_fichero_en_descarga"])
                r2 = lic.evaluar_ficha(texto2, t.get("licencia_regex_ficha"), lic.normalizar(t["licencia_esperada"]))
                _guardar_licencia(t, r2, texto2, salida)
                if r2["estado"] != "admitida":
                    _fuera(t, "licencia: " + r2["motivo"])
                    continue
            items, cif = extraer_items(t, carp)
        except EsquemaError as e:
            _fuera(t, "esquema/descarga: %s" % e)
            continue
        except Exception as e:                                # noqa: BLE001 - un fallo de red o de lectura deja SOLO esa tarea fuera
            _fuera(t, "error %s: %s" % (type(e).__name__, str(e)[:300]))
            continue
        pool[t["nombre"]] = (items, cif)
        admitidas.append(t)
        log("OK     %-26s licencia=%s rev=%s filas_utiles=%d clases_origen=%d" % (t["nombre"], t["licencia"], str(t["rev"])[:12], len(items), len({i["etiqueta"] for i in items})))
    # solapes: las filas de TODAS las tareas admitidas contra v2, entrenamiento y entre sí
    ref = sol.Referencia()
    pos = {}
    for t in admitidas:
        for n, i in enumerate(pool[t["nombre"]][0]):
            pos[(t["nombre"], n)] = ref.añadir(t["nombre"], n, i["texto"])
    resumen_h = {}
    if huellas:
        for f in sorted(Path(huellas).glob("huellas-*.jsonl.gz")):
            resumen_h[f.name] = dict(sha256=sha256_fichero(f), huellas=ref.comparar(sol.leer_huellas(f), f.name.replace("huellas-", "").replace(".jsonl.gz", "")))
            log("HUELLAS %s: %d" % (f.name, resumen_h[f.name]["huellas"]))
    est = Path(huellas or ".") / "estado-huellas.json"
    if huellas and est.exists():                     # lo que dejó bateria_v3_huellas: qué se comprobó y qué queda «solape no comprobado»
        man["solapes"]["estado_huellas"] = json.loads(est.read_text(encoding="utf-8"))
        man["solape_no_comprobado"] = man["solapes"]["estado_huellas"].get("no_comprobado", man.get("solape_no_comprobado", []))
    man["solapes"]["huellas_sha256"] = {n: v["sha256"] for n, v in resumen_h.items()}
    man["solapes"]["huellas_leidas"] = {n: v["huellas"] for n, v in resumen_h.items()}
    man["solapes"]["huellas_usadas"] = bool(huellas)
    _cruzadas(ref)
    for t in admitidas:
        items, cif0 = pool[t["nombre"]]
        veredicto = lambda n, nom=t["nombre"]: ref.veredicto(pos[(nom, n)])      # noqa: E731
        sop, ev, cif = armar(t, items, solape=veredicto, k=k, tope=tope)
        por_fuente = Counter(ref.res[pos[(t["nombre"], n)]]["fuente"] for n in range(len(items)) if pos[(t["nombre"], n)] in ref.res)
        cif.update(cif0)
        cif["solape_por_fuente"] = dict(por_fuente)
        if len(ev) < MIN_TEST_TAREA or len(cif["clases"]) < 2:
            _fuera(t, "tras limpiar y medir solapes quedan %d filas de test y %d clases (mínimo %d y 2)" % (len(ev), len(cif["clases"]), MIN_TEST_TAREA), cifras=cif)
            continue
        t["_clases"] = cif["clases"]
        repo = t["fuente"].get("repo") or (t.get("recurso_url") if t["fuente"]["tipo"] == "ckan" else t["fuente"]["url"].split("{rev}")[0])
        filas_s = [a_fila(t, i, "train", n, t["licencia"], repo) for n, i in enumerate(sop)]
        filas_t = [a_fila(t, i, "test", n, t["licencia"], repo) for n, i in enumerate(ev)]
        t["sha256_soporte"] = escribir_jsonl(salida / t["fichero_soporte"], filas_s)
        t["sha256"] = escribir_jsonl(salida / t["fichero_test"], filas_t, solo_lectura=True)
        t.update(estado="construida", n_soporte=len(sop), n_test=len(ev), clases=cif["clases"], cifras=cif, solapes=dict(descartadas=cif.get("solape_descartadas", 0),
                 marcadas=cif.get("solape_marcadas", 0), por_fuente=dict(por_fuente)), motivo_fuera="")
        t.pop("_clases", None)
        log("HECHA  %-26s soporte=%d test=%d clases=%d solape(desc/marc)=%s/%s" % (t["nombre"], len(sop), len(ev), len(cif["clases"]), cif.get("solape_descartadas"), cif.get("solape_marcadas")))
    for t in man["tareas"]:
        t.pop("_error_ficha", None)
    man["fase"] = "2 (construida)"
    man["resumen"] = resumen(man)
    guardar_json(salida / "manifiesto.json", man)
    guardar_json(salida / "resumen.json", man["resumen"])
    return man


def _licencia_en_zip(carpeta, patron):
    for z in sorted(Path(carpeta).rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if fnmatch.fnmatch(n, patron):
                    return zf.read(n).decode("utf-8", "replace")
    return ""


def _cruzadas(ref):
    """Solapes entre tareas distintas de v3: la fila que coincide se anota con fuente «v3:<otra tarea>» (descartar si es casi-duplicado)."""
    for i, (tarea, _, h, s) in enumerate(ref.filas):
        cand = set(ref.por_h.get(h, ()))
        for v in s:
            cand.update(ref.por_s.get(v, ()))
        for j in cand:
            if j == i or ref.filas[j][0] == tarea:
                continue
            jac = 1.0 if ref.filas[j][2] == h else sol.jaccard(s, ref.filas[j][3])
            if jac >= sol.UMBRAL_DESCARTE:
                ref._anotar(i, "v3:" + ref.filas[j][0], "exacto" if jac == 1.0 else "casi", jac)
            elif jac >= sol.UMBRAL_MARCA:
                ref._anotar(i, "v3:" + ref.filas[j][0], "marca", jac)


def resumen(man):
    r = dict(por_estado=dict(Counter(t["estado"] for t in man["tareas"])), por_tipo={}, fuera={}, n_bajo_minimo=N_BAJO)
    hechas = lambda ts, **kw: [t for t in ts if t["estado"] == "construida" and all(t[k] == v for k, v in kw.items())]    # noqa: E731
    for tp in TIPOS:
        ts = [t for t in man["tareas"] if t["tipo"] == tp]
        cu = hechas(ts, bloque="cuentan")
        r["por_tipo"][tp] = dict(candidatas=len(ts), construidas=len(hechas(ts)), cuentan=len(cu), informativo=len(hechas(ts, bloque="informativo")), n_bajo=len(cu) < N_BAJO,
                                 es=len(hechas(ts, idioma="es", bloque="cuentan")), en=len(hechas(ts, idioma="en", bloque="cuentan")),
                                 es_informativo=len(hechas(ts, idioma="es", bloque="informativo")), en_informativo=len(hechas(ts, idioma="en", bloque="informativo")))
    r["por_idioma"] = {i: dict(cuentan=len(hechas(man["tareas"], idioma=i, bloque="cuentan")), informativo=len(hechas(man["tareas"], idioma=i, bloque="informativo")),
                               candidatas=sum(t["idioma"] == i for t in man["tareas"])) for i in ("es", "en")}
    r["por_idioma"]["es"]["n_bajo"] = r["por_idioma"]["es"]["cuentan"] < N_BAJO
    for t in man["tareas"]:
        if t["estado"] in ("fuera", "pendiente"):
            r["fuera"][t["nombre"]] = t["motivo_fuera"]
    return r


def reservas_v2(manifiesto, trabajo, salida):
    """Baja la tarjeta (revisión de v2) de las 4 fuentes con reserva y la pasa por la comprobación automática; guarda texto y sha256."""
    man = cargar_manifiesto(manifiesto)
    out = []
    for r in man["reservas_v2"]:
        t = dict(nombre=r["nombre"], rev=r["rev"], fuente=dict(tipo="hf", repo=r["repo"]), url_ficha=r["url_ficha"], licencia_esperada=r["licencia_v2"])
        txt = bajar_ficha(t, Path(trabajo) / "reservas" / r["nombre"])
        ev = lic.evaluar_tarjeta(txt, esperada=r["licencia_v2"])
        _guardar_licencia(t, ev, txt, Path(salida) / "reservas-v2")
        origen = ""
        try:
            origen = url_leer(r["url_origen"], tope=1 << 21).decode("utf-8", "replace")
            (Path(salida) / "reservas-v2" / (r["nombre"] + ".origen.txt")).write_text(origen, encoding="utf-8")
        except Exception as e:                            # noqa: BLE001
            origen = ""
            ev["avisos"].append("origen %s no descargable: %s" % (r["url_origen"], type(e).__name__))
        out.append(dict(nombre=r["nombre"], tarjeta=dict(estado=ev["estado"], licencia=ev["licencia"], motivo=ev["motivo"], avisos=ev["avisos"], sha256=ev["sha256_texto"]),
                        origen=dict(url=r["url_origen"], sha256=lic.sha256_texto(origen) if origen else "", bytes=len(origen), menciona=bool(origen and re.search(r["patron_origen"], origen, re.I)),
                                    patron=r["patron_origen"])))
        log("RESERVA %-22s tarjeta=%s licencia=%s origen_descargado=%s menciona=%s" % (r["nombre"], ev["estado"], ev["licencia"], bool(origen), out[-1]["origen"]["menciona"]))
    guardar_json(Path(salida) / "reservas-v2.json", out)
    return out


# ---------------------------------------------------------------------------------------------------------------- selladas
def abrir_sellada(ruta, motivo, quien=None, registro=APERTURAS, manifiesto=None):
    """Lee un .jsonl sellado ANOTANDO antes la apertura (fecha, quién, motivo, fichero, sha256 actual). Sin motivo (>= 8 caracteres) se niega.
    Si se pasa el manifiesto y el sha256 no coincide con el sellado, anota la apertura fallida y se niega. Solo abre ficheros de una carpeta `sellada`."""
    if len((motivo or "").strip()) < 8:
        raise PermissionError("abrir_sellada exige un motivo explícito (>= 8 caracteres): no se lee %s" % ruta)
    ruta = Path(ruta)
    if ruta.parent.name != "sellada" or ruta.suffix != ".jsonl":
        raise ValueError("abrir_sellada solo abre .jsonl de una carpeta «sellada»: %s" % ruta)
    sha = sha256_fichero(ruta)
    linea = dict(fecha=time.strftime("%Y-%m-%dT%H:%M:%S%z"), quien=quien or getpass.getuser(), motivo=motivo.strip(), fichero=ruta.name, sha256=sha)
    esperado = None
    if manifiesto:
        esperado = {t["nombre"] + ".jsonl": t["sha256"] for t in cargar_manifiesto(manifiesto)["tareas"] if t.get("sha256")}.get(ruta.name)
        linea["coincide_con_manifiesto"] = (esperado == sha) if esperado else None
    Path(registro).parent.mkdir(parents=True, exist_ok=True)
    with open(registro, "a", encoding="utf-8") as fh:          # primero se anota, luego se lee
        fh.write(json.dumps(linea, ensure_ascii=False) + "\n")
    if esperado and esperado != sha:
        raise ValueError("el sha256 de %s no coincide con el del manifiesto: el fichero cambió desde el sellado" % ruta.name)
    return [json.loads(l) for l in open(ruta, encoding="utf-8")]


def auditar_log(datos, log_ruta, registro=None, textos_extra=()):
    """Comprueba que el log no contiene texto de test (los primeros 40 caracteres normalizados de cada fila). Abre las selladas con abrir_sellada y
    anota las aperturas en `registro` (aperturas de la máquina, no las del repo). Imprime solo recuentos. Devuelve el nº de filas cuyo texto aparece."""
    datos = Path(datos)
    registro = registro or (datos / "aperturas-auditoria.jsonl")
    contenido = norm(Path(log_ruta).read_text(encoding="utf-8", errors="replace"))
    hallazgos = filas = 0
    for p in sorted((datos / "sellada").glob("*.jsonl")):
        for f in abrir_sellada(p, "auditoría de que el log no filtra texto de test (solo cuenta coincidencias)", registro=registro):
            filas += 1
            frag = norm(f["texto"])[:40]
            if len(frag) >= 20 and frag in contenido:
                hallazgos += 1
    extra = n_extra = 0
    if textos_extra:                                  # textos de v2 (todos) y una muestra de cada fuente del entrenamiento: ni uno puede compartir 8 palabras seguidas con el log
        from model.datos import bateria_v3_huellas as hue
        n_extra, extra = hue.fugas_en_log(contenido, hue.textos_de_dirs(textos_extra))
    log("AUDITORIA log=%s filas_de_test_revisadas=%d coincidencias=%d textos_v2_y_entrenamiento_revisados=%d coincidencias_v2_y_entrenamiento=%d" % (
        Path(log_ruta).name, filas, hallazgos, n_extra, extra))
    return hallazgos + extra


def candidatas_md(man):
    L = ["# Batería v3: candidatas y descartadas (FASE 1, sin red)", "",
         "Generado por `python -m model.datos.bateria_v3 candidatas-md` a partir de `bateria_v3_candidatas.py`. **Nada de esto está verificado**: ids, columnas, URL y licencias vienen de memoria",
         "y del repo; la licencia de cada una queda «por verificar en fase 2» y la comprueba el ud contra la lista admitida (CC0, CC BY, MIT, Apache-2.0, ODC-BY, CDLA-Permissive, dominio público).",
         "",
         "**Dos bloques.** `cuentan` entra en el agregado y en las hipótesis selladas; `informativo` (benchmarks públicos, probable contaminación de L3) se construye y se sella igual pero NO cuenta.",
         "Las candidatas con «reemplazo de» se añadieron para mantener ≥ %d que cuenten por tipo; lo que no resuelva en fase 2 queda fuera con su motivo y, si un tipo baja de %d, se informa con la marca «n bajo»." % (N_BAJO, N_BAJO), "",
         "| tipo | candidatas | cuentan | informativo | reemplazos | cuentan en es | cuentan en en |", "|---|---|---|---|---|---|---|"]
    for tp in TIPOS:
        ts = [t for t in man["tareas"] if t["tipo"] == tp]
        cu = [t for t in ts if t["bloque"] == "cuentan"]
        L.append("| %s | %d | %d | %d | %d | %d | %d |" % (tp, len(ts), len(cu), len(ts) - len(cu), sum(bool(t["reemplazo_de"]) for t in ts),
                                                         sum(t["idioma"] == "es" for t in cu), sum(t["idioma"] == "en" for t in cu)))
    L.append("")
    for tp, nombre in (("choice", "choice (clases con nombre)"), ("yesno", "sí/no"), ("score", "score (niveles ordinales)")):
        ts = [t for t in man["tareas"] if t["tipo"] == tp]
        L += ["## %s: %d candidatas (es %d, en %d; cuentan %d, informativo %d)" % (nombre, len(ts), sum(t["idioma"] == "es" for t in ts), sum(t["idioma"] == "en" for t in ts),
                                                                               sum(t["bloque"] == "cuentan" for t in ts), sum(t["bloque"] == "informativo" for t in ts)), "",
              "| tarea | bloque | idioma | dominio | fuente (id o URL) | licencia esperada | ficha | clases | sintética | id | lector | por qué no solapa | riesgos |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in ts:
            f = t["fuente"]
            src = f.get("repo") or f.get("url") or "catalogo %s: %s" % (f.get("catalogo", CKAN_DEFECTO), f.get("paquete") or f.get("busqueda"))
            bl = t["bloque"] + (" (%s)" % t["motivo_bloque"] if t["motivo_bloque"] else "") + (" · reemplazo de: %s" % t["reemplazo_de"] if t["reemplazo_de"] else "")
            L.append("| %s | %s | %s | %s | `%s` (rev: se fija en fase 2) | %s (por verificar) | %s%s | %s | %s | %s | %s | %s | %s |" % (
                t["nombre"], bl, t["idioma"], t["dominio"], src, t["licencia_esperada"], t["url_ficha"], "" if t["url_ficha_confirmada"] else " (URL sin confirmar)", t["clases_esperadas"] or "?",
                "sí: %s" % t["sintetico"] if t["sintetico"] else "no", t["confianza_id"], "PENDIENTE" if t["lector"].get("pendiente") else "listo (sin probar)", t["no_solapa"], t["riesgos"]))
        L.append("")
    L += ["## Descartadas (con motivo)", "", "| fuente | motivo |", "|---|---|"] + ["| %s | %s |" % (d["fuente"], d["motivo"]) for d in man["descartadas"]] + [""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("manifiesto-inicial")
    a.add_argument("--salida", default=str(MANIFIESTO))
    b = sp.add_parser("candidatas-md")
    b.add_argument("--manifiesto", default=str(MANIFIESTO))
    b.add_argument("--salida", required=True)
    c = sp.add_parser("construir")
    c.add_argument("--manifiesto", default=str(MANIFIESTO))
    c.add_argument("--trabajo", required=True)
    c.add_argument("--salida", required=True)
    c.add_argument("--huellas")
    c.add_argument("--solo", nargs="*")
    d = sp.add_parser("reservas-v2")
    d.add_argument("--manifiesto", default=str(MANIFIESTO))
    d.add_argument("--trabajo", required=True)
    d.add_argument("--salida", required=True)
    e = sp.add_parser("auditar-log")
    e.add_argument("--datos", required=True)
    e.add_argument("--log", required=True)
    e.add_argument("--textos-extra", nargs="*", default=[], help="carpetas con los jsonl de v2 y la muestra del entrenamiento: ningún texto suyo puede estar en el log")
    x = ap.parse_args(argv)
    if x.cmd == "manifiesto-inicial":
        guardar_json(x.salida, manifiesto_inicial())
        log("manifiesto inicial: %d candidatas, %d descartadas" % (len(CANDIDATAS), len(DESCARTADAS)))
    elif x.cmd == "candidatas-md":
        Path(x.salida).write_text(candidatas_md(cargar_manifiesto(x.manifiesto)), encoding="utf-8")
    elif x.cmd == "construir":
        man = construir(x.manifiesto, x.trabajo, x.salida, huellas=x.huellas, solo=x.solo)
        log("RESUMEN " + json.dumps(man["resumen"], ensure_ascii=False))
    elif x.cmd == "reservas-v2":
        reservas_v2(x.manifiesto, x.trabajo, x.salida)
    else:
        return 1 if auditar_log(x.datos, x.log, textos_extra=x.textos_extra) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
