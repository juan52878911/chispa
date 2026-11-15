#!/usr/bin/env python3
"""Banco de latencia de L0, L1 y L2 de Chispa (int8) con 1 vCPU, pensado para correr dentro de `docker run --cpus=1 --memory=1800m`.

    python3 bench.py --dir ARTEFACTOS [--chispa modelo-int8.chispa] [--tok-e5 tokenizer.json] [--tok-l1 tokenizer.json]
                     [--fase frio|caliente] [--n 400] [--textos F.jsonl] [--opciones 6] [--hilos 1] --salida J.json

Solo numpy + onnxruntime + tokenizers (sin torch ni el repo). Lee del `.chispa` v3 las secciones L0TF (tipo 8), L1 (3), L2 (4) y TORR (9)
con el mismo contenedor que `format/chispa3`; cada capa se puede dar también suelta (`--l0tf`, `--l1`, `--l2-onnx`, `--torr-onnx`).
`--l0tf` admite un `.chispa` v3 entero (se saca su L0TF) o la sección suelta. Una capa que falta (o que no se sabe leer aquí) queda en `omitidos` con el motivo y el resto se mide.

Qué mide, por petición de un texto (lote 1, 1 hilo intra-op, secuencias sin relleno):
  L0   TF-IDF de caracteres (char_wb 2-5, sublineal) + regresión logística, como `format/chispa3/l0tf.go`.
  L1   tokenizar con el tokenizador del potion, media de vectores int8 (con su escala), normalizar y puntuar contra K opciones.
  L2   tokenizar con el de e5 («query: » + texto, tope --max-tok), una pasada del ONNX int8 (ids -> S) y MaxSim contra las torres.
  torres  (solo con TORR) la torre al vuelo de --opciones opciones nuevas en un lote: lo que cuesta un esquema zero-shot sin caché.
  cadena  L0 + L1 + L2 ejecutadas las tres (PEOR CASO de la cascada: la real se detiene en la primera que acepta);
  cadena_zero_shot  L1 + L2 + torres.
Los vectores de las opciones y de las torres son sintéticos (las secciones ESQUEMAS no se leen): el coste de la puntuación es el mismo,
la decisión no es la real. Fase `frio`: una sola petición tras arrancar el proceso, con la carga incluida; fase `caliente`: --n peticiones
distintas tras calentar. Las latencias de cada capa en `caliente` salen de las mismas peticiones de la cadena (no de corridas aparte).
"""
import argparse
import json
import math
import mmap
import os
import platform
import resource
import struct
import sys
import time

T_INICIO = time.perf_counter()

TIPOS = {"META": 1, "L0": 2, "L1": 3, "L2": 4, "ESQUEMAS": 5, "CALIB": 6, "CONJ": 7, "L0TF": 8, "TORR": 9, "SOPL1": 10, "L0MULTI": 11}
TAM_CAB, TAM_ENTRADA = 32, 56   # format/chispa3/chispa3.go
SEC_PESOS, SEC_ESCALAS = 6, 7   # runtime/l1/formato.go
PALABRAS = ("quiero solicitar información sobre el estado de mi pedido reclamo por cobro duplicado factura servicio internet no funciona "
            "cita médica urgente medicamento autorización eps derecho de petición queja gracias buenos días necesito cambiar la dirección "
            "de entrega producto dañado devolución garantía tornillo martillo cemento pintura precio cotización disponible sucursal "
            "horario atención cancelar suscripción pago tarjeta débito cuenta bloqueada contraseña acceso error aplicación celular "
            "señor señora por favor ayuda inmediata respuesta plazo días hábiles radicado número documento cédula teléfono correo").split()


# ---------------------------------------------------------------- utilidades
def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    a, b = int(math.floor(k)), int(math.ceil(k))
    return xs[a] + (xs[b] - xs[a]) * (k - a)


def resumen(ms):
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "p50_ms": round(pct(ms, 50), 3), "p95_ms": round(pct(ms, 95), 3), "p99_ms": round(pct(ms, 99), 3),
            "media_ms": round(sum(ms) / len(ms), 3), "max_ms": round(max(ms), 3)}


def leer_primera(ruta, defecto=None):
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return defecto


def arranque_proceso_s():
    """Segundos desde que arrancó el proceso (interprete incluido) hasta ahora, de /proc; si no hay /proc, desde el inicio de este módulo."""
    try:
        with open("/proc/self/stat", encoding="utf-8") as f:
            campos = f.read().rsplit(")", 1)[1].split()
        ticks = int(campos[19])                      # starttime (campo 22)
        hz = os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime", encoding="utf-8") as f:
            arriba = float(f.read().split()[0])
        return arriba - ticks / hz
    except (OSError, ValueError, IndexError):
        return time.perf_counter() - T_INICIO


def rss_pico_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def cgroup():
    out = {}
    for clave, ruta in (("cpu_max", "/sys/fs/cgroup/cpu.max"), ("memory_max", "/sys/fs/cgroup/memory.max"),
                        ("memory_swap_max", "/sys/fs/cgroup/memory.swap.max"), ("memory_peak_bytes", "/sys/fs/cgroup/memory.peak"),
                        ("memory_current_bytes", "/sys/fs/cgroup/memory.current")):
        v = leer_primera(ruta)
        if v is not None:
            out[clave] = v
    return out


def entorno(ort, np, tokenizers):
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    return {"cpu": cpu, "nucleos_visibles": os.cpu_count(),
            "afinidad": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "carga_1_5_15": os.getloadavg() if hasattr(os, "getloadavg") else None, "cgroup": cgroup(),
            "python": platform.python_version(), "numpy": np.__version__, "onnxruntime": ort.__version__, "tokenizers": tokenizers.__version__}


# ---------------------------------------------------------------- contenedor .chispa v3
def secciones_chispa(mm):
    """{tipo: (offset, longitud)} dentro de `mm` (mmap/bytes) de un `.chispa` v3. Lanza ValueError si no lo es."""
    if len(mm) < TAM_CAB:
        raise ValueError("fichero demasiado corto para ser un .chispa")
    ver = struct.unpack_from("<H", mm, 8)[0]
    if ver != 3:
        raise ValueError("versión %d (se esperaba 3)" % ver)
    n, total = struct.unpack_from("<IQ", mm, 12)[0], struct.unpack_from("<Q", mm, 16)[0]
    if total != len(mm):
        raise ValueError("la cabecera declara %d bytes y hay %d" % (total, len(mm)))
    inicio = TAM_CAB + n * TAM_ENTRADA
    out = {}
    for i in range(n):
        t, _, off, ln = struct.unpack_from("<IIQQ", mm, TAM_CAB + i * TAM_ENTRADA)
        if inicio + off + ln > len(mm):
            raise ValueError("sección %d fuera del fichero" % t)
        out[t] = (inicio + off, ln)
    return out


# ---------------------------------------------------------------- L0 (L0TF)
class L0TF:
    """Lee la sección L0TF (disposición en `format/chispa3/l0tf.go`) y puntúa como sklearn char_wb(2,5) sublineal + logística."""

    def __init__(self, buf):
        import numpy as np
        self.np = np
        v, V, C, dtype = struct.unpack_from("<IIII", buf, 0)
        if v != 1 or dtype not in (1, 2) or not (1 <= V <= 1 << 24) or not (2 <= C <= 1 << 16):
            raise ValueError("L0TF versión=%d V=%d C=%d dtype=%d no soportado" % (v, V, C, dtype))
        p = 16
        self.etiquetas = []
        for _ in range(C):
            (ln,) = struct.unpack_from("<H", buf, p)
            self.etiquetas.append(bytes(buf[p + 2:p + 2 + ln]).decode("utf-8"))
            p += 2 + ln
        self.inter = np.frombuffer(buf, dtype="<f4", count=C, offset=p)
        p += 4 * C
        self.idf = np.frombuffer(buf, dtype="<f4", count=V, offset=p)
        p += 4 * V
        offs = np.frombuffer(buf, dtype="<u4", count=V + 1, offset=p)
        p += 4 * (V + 1)
        blob = bytes(buf[p:p + int(offs[-1])])
        p += int(offs[-1])
        self.vocab = {blob[offs[i]:offs[i + 1]].decode("utf-8"): i for i in range(V)}   # la ruta Go busca en binario sin construirlo
        self.coef = np.frombuffer(buf, dtype="<f4" if dtype == 1 else "<f2", count=V * C, offset=p).reshape(V, C)
        self.V, self.C = V, C

    def clasificar(self, texto):
        np = self.np
        cuentas = {}
        for pal in texto.lower().split():
            w = " " + pal + " "
            for n in range(2, 6):
                for i in range(0, len(w) - n + 1):
                    j = self.vocab.get(w[i:i + n])
                    if j is not None:
                        cuentas[j] = cuentas.get(j, 0) + 1
        if not cuentas:
            return np.full(self.C, 1.0 / self.C)
        idx = np.fromiter(cuentas.keys(), dtype=np.int64, count=len(cuentas))
        tf = np.fromiter(cuentas.values(), dtype=np.float32, count=len(cuentas))
        x = (1.0 + np.log(tf)) * self.idf[idx]
        x /= max(float(np.linalg.norm(x)), 1e-12)
        z = x @ self.coef[idx].astype(np.float32) + self.inter
        e = np.exp(z - z.max())
        return e / e.sum()


# ---------------------------------------------------------------- L1 (.chispa-l1 + tokenizador del potion)
class L1:
    """Embedding estático int8 (`runtime/l1/README.md`): ids del tokenizador, media de filas con su escala, normaliza y puntúa K opciones."""

    def __init__(self, buf, tok_json, opciones, semilla=1):
        import numpy as np
        from tokenizers import Tokenizer
        self.np = np
        if bytes(buf[:4]) != b"CHL1":
            raise ValueError("no es un .chispa-l1 (magia CHL1)")
        ver, self.dim, self.vocab = struct.unpack_from("<III", buf, 8)
        if ver != 1:
            raise ValueError("versión .chispa-l1 %d" % ver)
        sec = [struct.unpack_from("<QQ", buf, 36 + 16 * i) for i in range(9)]
        (op, lp), (oe, le) = sec[SEC_PESOS], sec[SEC_ESCALAS]
        if lp != self.vocab * self.dim or le != 4 * self.vocab or op + lp > len(buf) or oe + le > len(buf):
            raise ValueError("secciones de pesos/escalas incoherentes")
        self.w = np.frombuffer(buf, dtype=np.int8, count=lp, offset=op).reshape(self.vocab, self.dim)
        self.s = np.frombuffer(buf, dtype="<f4", count=self.vocab, offset=oe)
        self.tok = Tokenizer.from_file(tok_json)
        self.tok.no_padding()
        rng = np.random.RandomState(semilla)
        m = rng.standard_normal((opciones, self.dim)).astype(np.float32)
        self.opc = m / np.linalg.norm(m, axis=1, keepdims=True)

    def puntuar(self, texto):
        np = self.np
        ids = [i for i in self.tok.encode(texto, add_special_tokens=False).ids if 0 <= i < self.vocab]
        if not ids:
            return np.zeros(len(self.opc), dtype=np.float32)
        idx = np.asarray(ids, dtype=np.int64)
        v = (self.w[idx].astype(np.float32) * self.s[idx, None]).mean(axis=0)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        return self.opc @ v


# ---------------------------------------------------------------- L2 y torres (ONNX)
def sesion_ort(ort, onnx, hilos):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = hilos, 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx, o, providers=["CPUExecutionProvider"])


def tokenizador_e5(ruta, max_tok):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(ruta)
    t.no_padding()
    t.enable_truncation(max_length=max_tok)
    return t


class L2:
    def __init__(self, ort, np, onnx, tok, hilos, max_tok, opciones, semilla=2):
        self.np, self.tok = np, tok
        self.s = sesion_ort(ort, onnx, hilos)
        self.nombres = {i.name for i in self.s.get_inputs()}
        if not {"input_ids", "attention_mask"} <= self.nombres:
            raise ValueError("el ONNX de L2 no tiene input_ids y attention_mask: %s" % sorted(self.nombres))
        self.salida = self.s.get_outputs()[0].name
        self.opciones, self.semilla, self.dim = opciones, semilla, None

    def codificar(self, texto, prefijo="query: "):
        np = self.np
        ids = np.asarray([self.tok.encode(prefijo + texto).ids], dtype=np.int64)
        return ids, np.ones_like(ids)

    def puntuar(self, texto):
        np = self.np
        ids, mask = self.codificar(texto)
        s = self.s.run([self.salida], {"input_ids": ids, "attention_mask": mask})[0]
        if self.dim is None:
            self.dim = s.shape[-1]
            rng = np.random.RandomState(self.semilla)
            t = rng.standard_normal((self.opciones, self.dim)).astype(np.float32)
            self.torres = t / np.linalg.norm(t, axis=1, keepdims=True)
        s = s[0] if s.ndim == 3 else s                # (K, d) ranuras
        return (s @ self.torres.T).max(axis=0), ids.shape[1]


class Torres:
    def __init__(self, ort, np, onnx, tok, hilos, prefijo, max_len):
        self.np, self.tok, self.prefijo, self.max_len = np, tok, prefijo, max_len
        self.s = sesion_ort(ort, onnx, hilos)
        self.salida = self.s.get_outputs()[0].name

    def calcular(self, textos):
        np = self.np
        self.tok.enable_truncation(max_length=self.max_len)
        enc = [self.tok.encode(self.prefijo + t).ids for t in textos]
        T = max(len(e) for e in enc)
        pad = 1                                   # <pad> de XLM-R; el valor no importa con la máscara a 0
        ids = np.full((len(enc), T), pad, dtype=np.int64)
        mask = np.zeros((len(enc), T), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, :len(e)], mask[i, :len(e)] = e, 1
        return self.s.run([self.salida], {"input_ids": ids, "attention_mask": mask})[0], T


def parsear_torr(b):
    """TORR v1/v2 (`format/chispa3/torr.go`): (prefijo, max_len, bytes ONNX)."""
    ver, dim, max_len, lpref = struct.unpack_from("<IIII", b, 0)
    if ver not in (1, 2):
        raise ValueError("TORR versión %d" % ver)
    cab = 16 if ver == 1 else 20
    return bytes(b[cab:cab + lpref]).decode("utf-8"), max_len, bytes(b[cab + lpref:])


# ---------------------------------------------------------------- textos
def generar_textos(n, semilla=0, ruta=None):
    import numpy as np
    rng = np.random.RandomState(semilla)
    if ruta:
        out = []
        with open(ruta, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if l:
                    t = json.loads(l).get("texto")
                    if isinstance(t, str) and t.strip():
                        out.append(t)
        if not out:
            raise ValueError("%s no tiene filas con el campo «texto»" % ruta)
        return [out[i % len(out)] for i in range(n)], len(out)
    lens = np.clip(np.exp(rng.normal(math.log(18), 0.6, n)), 3, 120).astype(int)   # mediana 18 palabras, cola hasta 120
    return [" ".join(PALABRAS[j] for j in rng.randint(0, len(PALABRAS), k)) + " %d" % i for i, k in enumerate(lens)], 0


def textos_torre(rng, k):
    return ["%s · %s · %s" % (" ".join(PALABRAS[j] for j in rng.randint(0, len(PALABRAS), 6)), PALABRAS[rng.randint(len(PALABRAS))],
                              " ".join(PALABRAS[j] for j in rng.randint(0, len(PALABRAS), 10))) for _ in range(k)]


# ---------------------------------------------------------------- carga
def cargar(a, np, ort, tokenizers):
    """Abre lo que haya. Devuelve (capas, omitidos, carga_ms, artefactos)."""
    capas, omitidos, carga, arts = {}, {}, {}, {}
    ruta_chispa = a.chispa or os.path.join(a.dir, "modelo-int8.chispa")
    bufs, sec = {}, {}
    mm = None
    if os.path.exists(ruta_chispa):
        t = time.perf_counter()
        with open(ruta_chispa, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            sec = secciones_chispa(mm)
            arts["chispa"] = {"ruta": ruta_chispa, "bytes": len(mm), "secciones": sorted(k for k, v in TIPOS.items() if v in sec)}
        except ValueError as e:
            omitidos["chispa"] = "no se pudo leer %s: %s" % (ruta_chispa, e)
        carga["chispa_abrir_ms"] = round((time.perf_counter() - t) * 1000, 2)
    else:
        arts["chispa"] = None

    def seccion(nombre):
        if TIPOS[nombre] in sec:
            o, l = sec[TIPOS[nombre]]
            return memoryview(mm)[o:o + l]
        return None

    def fichero(ruta):
        if ruta and os.path.exists(ruta):
            with open(ruta, "rb") as fh:
                return memoryview(mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))
        return None

    def seccion_de_fichero(ruta, nombre):
        """Si `ruta` es un .chispa v3 (p. ej. uno de `chispa train` con L0TF), la sección `nombre` de ese fichero; si no, None."""
        b = fichero(ruta)
        if b is None:
            return None
        try:
            o, l = secciones_chispa(b)[TIPOS[nombre]]
            return b[o:o + l]
        except (ValueError, KeyError, struct.error):
            return None

    # L0
    t = time.perf_counter()
    b = seccion_de_fichero(a.l0tf, "L0TF") or fichero(a.l0tf) or seccion("L0TF")
    if b is None:
        omitidos["L0"] = ("sin sección L0TF ni --l0tf" + ("; el .chispa trae la sección L0 de kindling v2 (modelo lineal), que este banco no sabe leer"
                                                           if TIPOS["L0"] in sec else ""))
    else:
        try:
            capas["L0"] = L0TF(b)
        except (ValueError, struct.error) as e:
            omitidos["L0"] = "L0TF ilegible: %s" % e
    carga["L0_ms"] = round((time.perf_counter() - t) * 1000, 2)

    # L1
    t = time.perf_counter()
    b = fichero(a.l1) or seccion("L1")
    tok_l1 = a.tok_l1 or os.path.join(a.dir, "tokenizer-potion.json")
    if b is None:
        omitidos["L1"] = "sin sección L1 ni --l1 (.chispa-l1)"
    elif not os.path.exists(tok_l1):
        omitidos["L1"] = "falta el tokenizer.json del potion (%s)" % tok_l1
    else:
        try:
            capas["L1"] = L1(b, tok_l1, a.opciones)
        except Exception as e:  # noqa: BLE001
            omitidos["L1"] = "L1 ilegible: %s" % e
    carga["L1_ms"] = round((time.perf_counter() - t) * 1000, 2)

    # L2
    t = time.perf_counter()
    tok_e5 = a.tok_e5 or os.path.join(a.dir, "tokenizer-e5.json")
    b = fichero(a.l2_onnx) or seccion("L2")
    if b is None:
        omitidos["L2"] = "sin sección L2 ni --l2-onnx"
    elif not os.path.exists(tok_e5):
        omitidos["L2"] = "falta el tokenizer.json de e5 (%s)" % tok_e5
    else:
        try:
            capas["L2"] = L2(ort, np, bytes(b), tokenizador_e5(tok_e5, a.max_tok), a.hilos, a.max_tok, a.opciones)
        except Exception as e:  # noqa: BLE001
            omitidos["L2"] = "no se pudo crear la sesión de L2: %s" % e
    carga["L2_ms"] = round((time.perf_counter() - t) * 1000, 2)

    # torres al vuelo
    t = time.perf_counter()
    b = seccion("TORR")
    if a.torr_onnx and os.path.exists(a.torr_onnx):
        with open(a.torr_onnx, "rb") as f:
            raw, pref, mx = f.read(), "passage: ", a.max_tok
    elif b is not None:
        try:
            pref, mx, raw = parsear_torr(b)
        except (ValueError, struct.error) as e:
            raw, pref, mx = None, "", 0
            omitidos["torres"] = "TORR ilegible: %s" % e
    else:
        raw, pref, mx = None, "", 0
        omitidos.setdefault("torres", "sin sección TORR ni --torr-onnx")
    if raw is not None:
        if not os.path.exists(tok_e5):
            omitidos["torres"] = "falta el tokenizer.json de e5 (%s)" % tok_e5
        else:
            try:
                capas["torres"] = Torres(ort, np, raw, tokenizador_e5(tok_e5, mx), a.hilos, pref, mx)
            except Exception as e:  # noqa: BLE001
                omitidos["torres"] = "no se pudo crear la sesión de la torre: %s" % e
    carga["torres_ms"] = round((time.perf_counter() - t) * 1000, 2)
    return capas, omitidos, carga, arts


# ---------------------------------------------------------------- una petición
def peticion(capas, texto, rng):
    """Una petición con todas las capas disponibles. Devuelve ({capa: ms}, tokens_l2)."""
    t = {}
    toks = None
    c0 = time.perf_counter()
    if "L0" in capas:
        a = time.perf_counter()
        capas["L0"].clasificar(texto)
        t["L0"] = (time.perf_counter() - a) * 1000
    if "L1" in capas:
        a = time.perf_counter()
        capas["L1"].puntuar(texto)
        t["L1"] = (time.perf_counter() - a) * 1000
    if "L2" in capas:
        a = time.perf_counter()
        _, toks = capas["L2"].puntuar(texto)
        t["L2"] = (time.perf_counter() - a) * 1000
    t["cadena"] = (time.perf_counter() - c0) * 1000
    if "torres" in capas:
        a = time.perf_counter()
        capas["torres"].calcular(textos_torre(rng, capas["_opciones"]))
        t["torres"] = (time.perf_counter() - a) * 1000
        t["cadena_zero_shot"] = sum(t.get(k, 0.0) for k in ("L1", "L2", "torres"))
    return t, toks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="/artefactos")
    ap.add_argument("--chispa")
    ap.add_argument("--l0tf")
    ap.add_argument("--l1")
    ap.add_argument("--l2-onnx")
    ap.add_argument("--torr-onnx")
    ap.add_argument("--tok-e5")
    ap.add_argument("--tok-l1")
    ap.add_argument("--fase", choices=("frio", "caliente"), default="caliente")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--calentar", type=int, default=20)
    ap.add_argument("--textos")
    ap.add_argument("--opciones", type=int, default=6)
    ap.add_argument("--hilos", type=int, default=1)
    ap.add_argument("--max-tok", type=int, default=128)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida")
    a = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", str(a.hilos))

    t0 = time.perf_counter()
    import numpy as np
    import onnxruntime as ort
    import tokenizers
    import_ms = round((time.perf_counter() - t0) * 1000, 2)

    res = {"fase": a.fase, "hilos": a.hilos, "max_tok": a.max_tok, "opciones": a.opciones, "import_ms": import_ms, "entorno": entorno(ort, np, tokenizers)}
    capas, omitidos, carga, arts = cargar(a, np, ort, tokenizers)
    capas["_opciones"] = a.opciones
    res.update({"carga_ms": carga, "artefactos": arts, "omitidos": omitidos, "capas_medidas": sorted(k for k in capas if not k.startswith("_"))})
    if not res["capas_medidas"]:
        res["error"] = "ninguna capa disponible: nada que medir"
    rng = np.random.RandomState(a.semilla + 7)

    if a.fase == "frio":
        textos, _ = generar_textos(2, a.semilla, a.textos)
        textos[0] = textos[0] if a.textos else " ".join(PALABRAS[j] for j in rng.randint(0, len(PALABRAS), 18))   # ~18 palabras, la mediana
        t1, k1 = peticion(capas, textos[0], rng)
        res["tiempo_hasta_primera_respuesta_desde_arranque_proceso_ms"] = round(arranque_proceso_s() * 1000, 1)
        t2, _ = peticion(capas, textos[1], rng)
        res["primera_peticion_ms"] = {k: round(v, 3) for k, v in t1.items()}
        res["segunda_peticion_ms"] = {k: round(v, 3) for k, v in t2.items()}
        res["tokens_l2_primera"] = k1
    else:
        textos, n_fuente = generar_textos(a.n + a.calentar, a.semilla, a.textos)
        res["textos"] = {"origen": a.textos or "sintéticos (lognormal, mediana 18 palabras, 3-120)", "n": a.n, "calentamiento": a.calentar,
                         "filas_distintas_en_la_fuente": n_fuente or None}
        regs, toks = [], []
        for i, tx in enumerate(textos):
            t, k = peticion(capas, tx, rng)
            if i >= a.calentar:
                regs.append(t)
                toks.append(k)
        out = {}
        for capa in ("L0", "L1", "L2", "torres", "cadena", "cadena_zero_shot"):
            xs = [r[capa] for r in regs if capa in r]
            if xs:
                out[capa] = resumen(xs)
        if "L2" in capas:
            cubos = {}
            for r, k in zip(regs, toks):
                if k is None:
                    continue
                c = "<=16" if k <= 16 else "<=32" if k <= 32 else "<=64" if k <= 64 else ">64"
                cubos.setdefault(c, []).append(r["L2"])
            out["L2_por_tokens"] = {c: resumen(v) for c, v in sorted(cubos.items())}
            out["tokens_l2"] = {"p50": pct([k for k in toks if k], 50), "p95": pct([k for k in toks if k], 95), "max": max(k for k in toks if k)}
        res["caliente"] = out
    res["rss_pico_mb"] = rss_pico_mb()
    res["cgroup_al_final"] = cgroup()
    if a.salida:
        with open(a.salida, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    c = res.get("caliente", {})
    print("fase=%s capas=%s omitidos=%s rss_pico_mb=%s %s" % (a.fase, res["capas_medidas"], sorted(omitidos), res["rss_pico_mb"],
          " ".join("%s_p50=%s_p99=%s" % (k, v.get("p50_ms"), v.get("p99_ms")) for k, v in c.items() if isinstance(v, dict) and "p50_ms" in v)
          or "primera_peticion_ms=%s" % res.get("primera_peticion_ms")))
    return 0 if res["capas_medidas"] else 2


if __name__ == "__main__":
    sys.exit(main())
