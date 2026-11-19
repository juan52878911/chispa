"""Huellas de v2 y del entrenamiento DENTRO del ud de la fase 2 (c7i): los textos entran en la instancia y solo salen hashes y sketches.

    python -m model.datos.bateria_v3_huellas v2 --dev dev.tgz --selladas selladas.tgz --trabajo DIR --salida DIR_HUELLAS
    python -m model.datos.bateria_v3_huellas entrenamiento --manifiesto model/alt/manifiesto-entrenamiento.json --trabajo DIR --salida DIR_HUELLAS --muestra DIR_MUESTRA

v2: se extraen los dos .tgz (cuyo sha256 ya comprobó el ud) en carpetas nuevas, con extracción segura, y se hace la huella de cada texto como en `bateria_v3_solapes`.
entrenamiento: las fuentes PÚBLICAS (id de Hugging Face, con revisión si el manifiesto la trae) se bajan en la propia instancia y se hace la huella de CADA celda de texto
(≥ 20 caracteres) de sus ficheros de datos, sin saber qué columna usó el entrenamiento: es un superconjunto, así que solo puede descartar de más, nunca de menos.
Lo que no es público (PQRS u otras propias, o lo que falle al bajar) NO se baja: queda en `estado-huellas.json` como «solape no comprobado» con su motivo; comprobarlo es de Juan, en su máquina, solo con hashes.
Nada de esto imprime texto: solo recuentos, estados y motivos. Los textos de v2 (y una muestra de cada fuente del entrenamiento) se quedan en la instancia hasta que `bateria_v3 auditar-log` comprueba que el log no los filtra.
"""
import argparse
import gzip
import json
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

from model.datos import bateria_v3 as b
from model.datos import bateria_v3_solapes as sol

MIN_CAR = 20                     # celdas más cortas no son texto de un ejemplo (etiquetas, ids, fechas)
PATRONES_DATOS = ["*.parquet", "*.csv", "*.csv.gz", "*.tsv", "*.json", "*.jsonl", "*.jsonl.gz", "*.json.gz"]
IGNORAR = ("dataset_infos.json", "dataset_info.json", "config.json", "state.json", "package.json")
TOPE_GB_DEFECTO = 8
TOPES_GB = {"tasksource/tasksource-jev-typed-decisions": 14}
# Fuentes enormes (tsjev: 134,5 GB): no caben en el disco ni en el tiempo de una sola vez. Se baja, se huella y se BORRA fichero a fichero (el disco aguanta el fichero más grande, no la
# suma); sin tope de tamaño total: manda el tope de tiempo de las huellas y los ficheros que no den tiempo cuentan como «sin tiempo» (fuente «parcial», que va a solape_no_comprobado).
POR_FICHEROS = {"tasksource/tasksource-jev-typed-decisions"}
TOPE_FICHERO_GB = 40
MAX_HUELLAS_POR_FICHEROS = 4_000_000     # cupo de huellas de una fuente enorme (memoria y tamaño del fichero de huellas); lo que falte cuenta como «sin tiempo»
FORMATO = {".parquet": "parquet", ".csv": "csv", ".tsv": "tsv", ".json": "jsonl", ".jsonl": "jsonl"}
RUTA_ENTRENAMIENTO = b.AQUI.parent / "alt" / "manifiesto-entrenamiento.json"
ANONIMO = re.compile(r"@\w+")
MUESTRA_AUDITORIA = 5000         # textos por fuente del entrenamiento que se guardan para auditar el log (los de v2 se revisan enteros)


def cargar_entrenamiento(ruta=None):
    return json.loads(Path(ruta or RUTA_ENTRENAMIENTO).read_text(encoding="utf-8"))


def repos_de(fuente):
    """Ids de HF de un campo `fuente` («owner/repo», «owner/repo_1 y _2»); vacío si no es un id de Hugging Face."""
    out = re.findall(r"(?<![\w/.-])([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)", fuente or "")
    for suf in re.findall(r"\by\s+(_\d+)\b", fuente or ""):
        if out and re.search(r"_\d+$", out[0]):
            out.append(re.sub(r"_\d+$", suf, out[0]))
    return list(dict.fromkeys(out))


def clasificar_entrenamiento(m):
    """-> {publicos: [{nombre, repo, rev}], no_comprobado: [{nombre, motivo}]}, solo con lo que dice el manifiesto (sin red)."""
    pub, no = [], []
    for c in m.get("conjuntos", []):
        repos = repos_de(c.get("fuente"))
        if not repos:
            no.append(dict(nombre=c["nombre"], fuente=c.get("fuente"), motivo="no es una fuente pública de Hugging Face (propia o sin id): no se baja; el solape se comprueba en la máquina de Juan, solo con hashes"))
            continue
        for r in repos:
            pub.append(dict(nombre=c["nombre"], repo=r, rev=c.get("revision") or c.get("rev")))
    for grupo, motivo in (("mientras_Juan_no_decida", "fuera del entrenamiento hasta que Juan decida: sin textos aquí; si entra, el solape se comprueba en su máquina, solo con hashes"),
                          ("medicion_solo", "solo medición (batería v1), no es entrenamiento: sus huellas, si se quieren, las hace Juan con `huellas --fuente v1`")):
        for nombre in (m.get("exclusiones") or {}).get(grupo, []):
            no.append(dict(nombre=nombre, fuente=None, motivo=motivo))
    return dict(publicos=pub, no_comprobado=no)


# ------------------------------------------------------------------------------------------------------------ extracción segura
def extraer_tgz(tgz, destino):
    """Extrae un .tgz en una carpeta NUEVA y vacía rechazando rutas absolutas, `..`, enlaces y ficheros especiales. -> nº de ficheros."""
    destino = Path(destino)
    if destino.exists() and any(destino.iterdir()):
        raise RuntimeError("la carpeta de extracción %s no está vacía" % destino)
    destino.mkdir(parents=True, exist_ok=True)
    base = destino.resolve()
    ok, n = [], 0
    with tarfile.open(tgz, "r:gz") as tf:
        for mb in tf.getmembers():
            nombre = mb.name
            if nombre.startswith("/") or ".." in Path(nombre).parts:
                raise RuntimeError("%s trae una ruta peligrosa (%s)" % (tgz, nombre[:60]))
            if Path(nombre).name.startswith("._"):
                continue
            if not (mb.isfile() or mb.isdir()):
                raise RuntimeError("%s trae un enlace o fichero especial (%s)" % (tgz, nombre[:60]))
            if not (base / nombre).resolve().is_relative_to(base):
                raise RuntimeError("%s se sale de la carpeta (%s)" % (tgz, nombre[:60]))
            mb.mode = 0o644 if mb.isfile() else 0o755
            mb.uid = mb.gid = 0
            ok.append(mb)
            n += mb.isfile()
        tf.extractall(destino, members=ok)
    return n


def jsonl_de(carpeta):
    return sorted(p for p in Path(carpeta).rglob("*") if p.is_file() and p.name.endswith((".jsonl", ".jsonl.gz")))


# ------------------------------------------------------------------------------------------------------------ lectura genérica
def _cadenas(v):
    if isinstance(v, str):
        if len(v) >= MIN_CAR:
            yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _cadenas(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _cadenas(x)


def _formato(p):
    n = p.name[:-3] if p.name.endswith(".gz") else p.name
    return FORMATO.get(Path(n).suffix.lower())


def textos_de_fichero(p, tope=None):
    """Celdas de texto (≥ MIN_CAR) de un fichero de datos genérico; sin duplicados exactos dentro del fichero. Un fichero ilegible lanza la excepción."""
    fm = _formato(p)
    if fm is None or p.name in IGNORAR:
        return
    vistos, n = set(), 0
    for fila in b._filas_fichero(p, fm, {}):
        for t in _cadenas(fila):
            k = hash(t)
            if k in vistos:
                continue
            vistos.add(k)
            yield t
            n += 1
            if tope and n >= tope:
                return


def textos_de_dirs(dirs):
    """Textos de los ficheros de las carpetas dadas (jsonl con campo de texto, o datos genéricos), para auditar el log."""
    for d in dirs:
        for p in sorted(x for x in Path(d).rglob("*") if x.is_file()):
            try:
                if p.name.endswith((".jsonl", ".jsonl.gz")):
                    yield from sol.textos_de(p)
                else:
                    yield from textos_de_fichero(p)
            except Exception:                                  # noqa: BLE001 - un fichero ilegible no impide auditar el resto
                continue


def fugas_en_log(log_norm, textos, n=8):
    """(textos revisados, textos con ≥ 1 secuencia de `n` palabras idéntica a una del log). Solo hashes de n-gramas del log; no guarda ni imprime texto."""
    def grams(t):
        pal = re.findall(r"\w+", t.lower())
        return {b_h(" ".join(pal[i:i + n])) for i in range(len(pal) - n + 1)}
    b_h = sol._h64
    del_log = grams(log_norm)
    rev = fugas = 0
    for t in textos:
        rev += 1
        if del_log and not del_log.isdisjoint(grams(t)):
            fugas += 1
    return rev, fugas


# ------------------------------------------------------------------------------------------------------------ comandos
def _estado(salida, clave, valor):
    ruta = Path(salida) / "estado-huellas.json"
    d = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    d[clave] = valor
    if clave == "entrenamiento":
        d["no_comprobado"] = valor["no_comprobado"]
    b.guardar_json(ruta, d)


def huellas_v2(dev, selladas, trabajo, salida, n_dev=6, n_selladas=8):
    trabajo, salida = Path(trabajo), Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    dirs, cuenta = [], {}
    for nombre, tgz, esperados in (("dev", dev, n_dev), ("selladas", selladas, n_selladas)):
        d = trabajo / ("v2-" + nombre)
        extraer_tgz(tgz, d)
        fs = jsonl_de(d)
        cuenta[nombre] = len(fs)
        if len(fs) != esperados:
            raise RuntimeError("v2 %s: %d ficheros jsonl y se esperaban %d" % (nombre, len(fs), esperados))
        dirs.append(d)
    n, nf, sin = sol.escribir_huellas(dirs, salida / "huellas-v2.jsonl.gz", "v2")
    b.log("HUELLAS v2: huellas=%d ficheros=%s filas_sin_campo_de_texto=%d" % (n, cuenta, sin))
    if not n or sin > 0.01 * (n + sin):
        raise RuntimeError("v2: %d huellas y %d filas sin campo de texto (más del 1 %%)" % (n, sin))
    _estado(salida, "v2", dict(huellas=n, ficheros=cuenta, filas_sin_campo_de_texto=sin, sha256_dev_y_selladas="comprobado por el ud antes de abrirlas"))
    return n


def huellas_entrenamiento(manifiesto, trabajo, salida, muestra, tope_min=40):
    m = cargar_entrenamiento(manifiesto)
    cl = clasificar_entrenamiento(m)
    trabajo, salida, muestra = Path(trabajo), Path(salida), Path(muestra)
    salida.mkdir(parents=True, exist_ok=True)
    muestra.mkdir(parents=True, exist_ok=True)
    comprobados, no = [], list(cl["no_comprobado"])
    t0, total = time.time(), 0
    with gzip.open(salida / "huellas-entrenamiento.jsonl.gz", "wt", encoding="utf-8") as out:
        for c in cl["publicos"]:
            carp = trabajo / "entrenamiento" / c["repo"].replace("/", "__")
            try:
                if time.time() - t0 > tope_min * 60:
                    raise TimeoutError("sin tiempo (tope de %d min para las huellas del entrenamiento)" % tope_min)
                sha, nbytes = b.hf_resolver(c["repo"], c["rev"], PATRONES_DATOS)
                por_ficheros = c["repo"] in POR_FICHEROS
                tope = TOPES_GB.get(c["repo"], TOPE_GB_DEFECTO) * (1 << 30)
                if nbytes > tope and not por_ficheros:
                    raise b.EsquemaError("la descarga previsible (%.1f GB) pasa del tope (%.1f GB)" % (nbytes / 2 ** 30, tope / 2 ** 30))
                if not nbytes:
                    raise b.EsquemaError("sin ficheros de datos (¿solo script de carga?)")
                if por_ficheros:
                    lista = [(r, tam) for r, tam in b.hf_listar(c["repo"], sha, PATRONES_DATOS) if _formato(Path(r)) and Path(r).name not in IGNORAR]
                    if any(tam > TOPE_FICHERO_GB * (1 << 30) for _, tam in lista):
                        raise b.EsquemaError("hay un fichero de más de %d GB: no cabe en el disco" % TOPE_FICHERO_GB)
                    ficheros = [carp / r for r, _ in lista]
                else:
                    b.hf_bajar(c["repo"], sha, PATRONES_DATOS, carp)
                    ficheros = [p for p in sorted(carp.rglob("*")) if p.is_file() and _formato(p) and p.name not in IGNORAR and ".cache" not in p.parts]
                n = leidos = ilegibles = 0
                vistos, mues = set(), []
                sin_tiempo = 0
                for p in ficheros:
                    if time.time() - t0 > tope_min * 60 or (por_ficheros and n >= MAX_HUELLAS_POR_FICHEROS):    # tope global (o cupo): lo que falta queda como parcial, no tira la corrida
                        sin_tiempo += 1
                        continue
                    try:
                        if por_ficheros:
                            b.hf_bajar(c["repo"], sha, [str(p.relative_to(carp))], carp)
                        for t in textos_de_fichero(p):
                            for tt in ((t, ANONIMO.sub("@usuario", t)) if "@" in t and c["nombre"] == "ciberacoso" else (t,)):   # ciberacoso se entrenó con @menciones anonimizadas
                                h = sol.huella(tt)
                                if h["h"] in vistos:
                                    continue
                                vistos.add(h["h"])
                                if len(mues) < MUESTRA_AUDITORIA:
                                    mues.append(tt)
                                h["f"] = "entrenamiento"
                                out.write(json.dumps(h) + "\n")
                                n += 1
                        leidos += 1
                    except Exception as e:                      # noqa: BLE001 - un fichero ilegible se cuenta; si no se lee ninguno la fuente queda sin comprobar
                        ilegibles += 1
                        b.log("HUELLAS entrenamiento %s: fichero ilegible (%s)" % (c["nombre"], type(e).__name__))
                    finally:
                        if por_ficheros:
                            p.unlink(missing_ok=True)              # libera el disco antes del siguiente fichero
                if not leidos or not n:
                    raise b.EsquemaError("ningún fichero legible con texto (ficheros %d, ilegibles %d)" % (len(ficheros), ilegibles))
                if ilegibles or sin_tiempo:
                    no.append(dict(nombre=c["nombre"], fuente=c["repo"], parcial=True, motivo="%d de %d ficheros ilegibles y %d sin tiempo: la fuente se comprobó solo en parte" % (ilegibles, len(ficheros), sin_tiempo)))
                with gzip.open(muestra / (c["repo"].replace("/", "__") + ".jsonl.gz"), "wt", encoding="utf-8") as mf:     # solo para auditar el log; se borra con el resto
                    mf.writelines(json.dumps({"texto": x}, ensure_ascii=False) + "\n" for x in mues)
                comprobados.append(dict(nombre=c["nombre"], repo=c["repo"], rev=sha, ficheros=len(ficheros), ilegibles=ilegibles, sin_tiempo=sin_tiempo, huellas=n))
                total += n
                b.log("HUELLAS entrenamiento %-26s %s rev=%s ficheros=%d huellas=%d" % (c["nombre"], c["repo"], sha[:12], len(ficheros), n))
            except Exception as e:                              # noqa: BLE001 - cualquier fallo deja SOLO esa fuente como «solape no comprobado» con su motivo
                no.append(dict(nombre=c["nombre"], fuente=c["repo"], motivo="no se pudo comprobar en el ud: %s: %s" % (type(e).__name__, str(e)[:200])))
                b.log("HUELLAS entrenamiento %-26s NO COMPROBADO: %s" % (c["nombre"], str(e)[:200]))
            finally:
                shutil.rmtree(carp, ignore_errors=True)         # libera disco: los textos ya son hashes y la muestra de auditoría
    b.log("HUELLAS entrenamiento: huellas=%d comprobados=%d no_comprobado=%d" % (total, len(comprobados), len(no)))
    _estado(salida, "entrenamiento", dict(huellas=total, comprobados=comprobados, no_comprobado=no, metodo="toda celda de texto >= %d caracteres de los ficheros de datos de la fuente (superconjunto de lo entrenado)" % MIN_CAR,
                                          manifiesto_entrenamiento_sha256=b.sha256_fichero(manifiesto or RUTA_ENTRENAMIENTO)))
    return total, comprobados, no


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("v2")
    a.add_argument("--dev", required=True)
    a.add_argument("--selladas", required=True)
    a.add_argument("--trabajo", required=True)
    a.add_argument("--salida", required=True)
    a.add_argument("--n-dev", type=int, default=6)
    a.add_argument("--n-selladas", type=int, default=8)
    e = sp.add_parser("entrenamiento")
    e.add_argument("--manifiesto")
    e.add_argument("--trabajo", required=True)
    e.add_argument("--salida", required=True)
    e.add_argument("--muestra", required=True, help="carpeta donde queda una muestra de textos por fuente, solo para auditar el log")
    e.add_argument("--tope-min", type=int, default=40)
    x = ap.parse_args(argv)
    try:
        if x.cmd == "v2":
            huellas_v2(x.dev, x.selladas, x.trabajo, x.salida, x.n_dev, x.n_selladas)
        else:
            huellas_entrenamiento(x.manifiesto, x.trabajo, x.salida, x.muestra, x.tope_min)
    except Exception as ex:                                      # noqa: BLE001
        b.log("FALLO huellas %s: %s: %s" % (x.cmd, type(ex).__name__, str(ex)[:300]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
