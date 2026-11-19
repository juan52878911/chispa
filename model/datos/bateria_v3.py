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
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from bisect import bisect_right
from collections import Counter, defaultdict
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
LICENCIA_PENDIENTE = "por verificar en fase 2"

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
             sintetico=c["sintetico"], clases_esperadas=c["clases"], confianza_id=c["confianza_id"], limpieza_esperada=c["limpieza_esperada"],
             no_solapa=c["no_solapa"], riesgos=c["riesgos"], pregunta=c["pregunta"], lector=c["lector"],
             fichero_soporte="soporte/%s.jsonl" % c["nombre"], fichero_test="sellada/%s.jsonl" % c["nombre"], sha256="", sha256_soporte="", ficheros_origen_sha256={},
             n_soporte=None, n_test=None, clases=None, cifras=None, solapes=None, motivo_fuera="")
    for k in ("licencia_regex_ficha", "licencia_fichero_en_descarga"):
        if k in c:
            e[k] = c[k]
    return e


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
    n = sum((s.size or 0) for s in info.siblings if not patrones or _cumple(s.rfilename, patrones))
    return info.sha, n


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


def bajar_ficha(t, carpeta):
    """Baja solo la ficha de licencia de la tarea (README.md o página) y devuelve su texto; no descarga datos."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    f = t["fuente"]
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
        url, rev = url_resolver(f)
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
        else:
            return None
    return v


def _a_texto(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_a_texto(x) for x in v)
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _abrir_texto(ruta, codificacion):
    if str(ruta).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(ruta, "rb"), encoding=codificacion, errors="replace", newline="")
    return open(ruta, "r", encoding=codificacion, errors="replace", newline="")


def _filas_fichero(ruta, formato, lector):
    cod = lector.get("codificacion", "utf-8")
    if formato == "parquet":
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(str(ruta)).iter_batches(batch_size=2000):
            for r in b.to_pylist():
                yield r
    elif formato in ("csv", "tsv"):
        csv.field_size_limit(1 << 28)
        kw = dict(delimiter="\t", quoting=csv.QUOTE_NONE) if formato == "tsv" else {}
        with _abrir_texto(ruta, cod) as fh:
            rd = csv.reader(fh, **kw)
            cols = lector.get("columnas")
            if isinstance(cols, dict):                       # sin cabecera: nombre -> posición
                for fila in rd:
                    yield {k: (fila[i] if i < len(fila) else None) for k, i in cols.items()}
            else:
                cab = next(rd, None)
                if cab is None:
                    return
                for fila in rd:
                    yield dict(zip(cab, fila))
    elif formato == "jsonl":
        with _abrir_texto(ruta, cod) as fh:
            primero = fh.read(1)
            fh.seek(0)
            if primero == "[":                               # un .json con una lista
                for r in json.load(fh):
                    yield r
            else:
                for l in fh:
                    if l.strip():
                        yield json.loads(l)
    else:
        raise EsquemaError("formato de lector desconocido: %s" % formato)


def _filas_zip(zpath, patrones):
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith("/") or not _cumple(n, patrones):
                continue
            with z.open(n) as fh:
                try:
                    yield n, json.load(fh)
                except ValueError:
                    continue


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
    if formato == "json_zip":
        zips = sorted(carpeta.rglob("*.zip"))
        if not zips:
            raise EsquemaError("sin zip en la descarga")
        fuentes = ((n, f) for z in zips for n, f in _filas_zip(z, lector["ficheros"]))
    else:
        fl = _ficheros(carpeta, lector["ficheros"])
        if not fl:
            raise EsquemaError("ningún fichero casa con %s" % lector["ficheros"])
        fuentes = ((r, fila) for r, p in fl for fila in _filas_fichero(p, formato, lector))
    cols_vistas = None
    for nombre, fila in fuentes:
        if cols_vistas is None:
            cols_vistas = set(fila)
            _comprobar_columnas(lector, fila)
        cif["filas_leidas"] += 1
        if any(_a_texto(_get(fila, col)).strip() not in vals for col, vals in (lector.get("filtro") or {}).items()):
            cif["filtradas"] += 1
            continue
        split = _split_de(nombre, fila, lector)
        for texto, et, grupo in _items_de_fila(lector, fila, cif, nombre):
            if et is None or not (texto or "").strip():
                cif["sin_texto_o_etiqueta"] += 1
                continue
            res.añadir("_" if lector.get("cuantiles") else str(et), dict(texto=texto, etiqueta=et, split=split, grupo=grupo))
    items = res.todos()
    if not items:
        raise EsquemaError("no salió ninguna fila utilizable (leídas %d, filtradas %d, sin texto o etiqueta %d)" % (cif["filas_leidas"], cif["filtradas"], cif["sin_texto_o_etiqueta"]))
    if lector.get("cuantiles"):
        _a_cuantiles(items, int(lector["cuantiles"]))
    return items, dict(cif)


def _comprobar_columnas(lector, fila):
    pedidas = []
    for c in ([lector["texto"]] if isinstance(lector.get("texto"), str) else list(lector.get("texto") or [])):
        pedidas.append(c)
    for k in ("etiqueta",):
        if lector.get(k):
            pedidas.append(lector[k])
    if lector.get("etiqueta_par"):
        pedidas.append(lector["etiqueta_par"]["col"])
    if lector.get("contar_patron"):
        pedidas.append(lector["contar_patron"]["col"])
    if lector.get("pares"):
        p = lector["pares"]
        pedidas += [p["bueno"], p["malo"]] + list(p.get("contexto") or [])
    pedidas += list((lector.get("filtro") or {}).keys())
    falta = [c for c in pedidas if _get(fila, c) is None and c.split(".")[0] not in fila]
    if falta:
        raise EsquemaError("faltan columnas esperadas %s; la fila trae %s" % (sorted(set(falta)), sorted(fila)[:40]))


def _etiqueta(lector, fila):
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
        return m.get(v)
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
    origen = t.get("no_solapa", "")
    if f["tipo"] == "hf":
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
        repo = t["fuente"].get("repo") or t["fuente"]["url"].split("{rev}")[0]
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
    r = dict(por_estado=dict(Counter(t["estado"] for t in man["tareas"])), por_tipo={}, fuera={})
    for tp in TIPOS:
        ts = [t for t in man["tareas"] if t["tipo"] == tp]
        r["por_tipo"][tp] = dict(candidatas=len(ts), construidas=sum(t["estado"] == "construida" for t in ts), es=sum(t["idioma"] == "es" and t["estado"] == "construida" for t in ts),
                                 en=sum(t["idioma"] == "en" and t["estado"] == "construida" for t in ts))
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


def auditar_log(datos, log_ruta, registro=None):
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
    log("AUDITORIA log=%s filas_de_test_revisadas=%d coincidencias=%d" % (Path(log_ruta).name, filas, hallazgos))
    return hallazgos


def candidatas_md(man):
    L = ["# Batería v3: candidatas y descartadas (FASE 1, sin red)", "",
         "Generado por `python -m model.datos.bateria_v3 candidatas-md` a partir de `bateria_v3_candidatas.py`. **Nada de esto está verificado**: ids, columnas, URL y licencias vienen de memoria",
         "y del repo; la licencia de cada una queda «por verificar en fase 2» y la comprueba el ud contra la lista admitida (CC0, CC BY, MIT, Apache-2.0, ODC-BY, CDLA-Permissive, dominio público).", ""]
    for tp, nombre in (("choice", "choice (clases con nombre)"), ("yesno", "sí/no"), ("score", "score (niveles ordinales)")):
        ts = [t for t in man["tareas"] if t["tipo"] == tp]
        L += ["## %s: %d candidatas (es %d, en %d)" % (nombre, len(ts), sum(t["idioma"] == "es" for t in ts), sum(t["idioma"] == "en" for t in ts)), "",
              "| tarea | idioma | dominio | fuente (id o URL) | licencia esperada | ficha | clases | sintética | id | lector | por qué no solapa | riesgos |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in ts:
            f = t["fuente"]
            src = f.get("repo") or f["url"]
            L.append("| %s | %s | %s | `%s` (rev: se fija en fase 2) | %s (por verificar) | %s%s | %s | %s | %s | %s | %s | %s |" % (
                t["nombre"], t["idioma"], t["dominio"], src, t["licencia_esperada"], t["url_ficha"], "" if t["url_ficha_confirmada"] else " (URL sin confirmar)", t["clases_esperadas"] or "?",
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
        return 1 if auditar_log(x.datos, x.log) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
