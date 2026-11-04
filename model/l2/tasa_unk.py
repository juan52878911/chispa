"""Fracción de tokens que la poda de vocabulario de L2 manda a <unk>, por corpus. Corre entero sin red y solo imprime agregados.

    python -m model.l2.tasa_unk --textos DIR_O_JSONL [--vocab model/bench/vocab-podado-e5-small.json] --tokenizador TOKENIZER.json_o_DIR \\
        [--campos texto] [--prefijo "query: "] [--max-tok 128] [--anonimizar] [--salida tasa_unk.json]
    python -m model.l2.tasa_unk … --barrido 30000 40000 55000 80000 120000 [--textos-poda DIR_O_JSONL] [--tope 0.02]

Cómo se poda (`model/bench/codificadores.py:elegir_vocab`, `model/l2/exportar.py:podar`): se conservan los ids que usa el corpus de poda
(massive-es train+val, MASSIVE en/pt test y HateCheck-es), los especiales, y el resto hasta `OBJETIVO_PODA` = 55 000 con piezas de
alfabeto latino en orden de id (la tabla de sentencepiece va de más a menos frecuente). Lo que no está en `ids_originales` pasa a
<unk> (id `unk` del JSON) dentro del grafo ONNX; `exportar.fraccion_unk` mide lo mismo sobre las filas de paridad.

Entrada: un `.jsonl`/`.jsonl.gz` o un directorio (recursivo); cada fichero es un corpus. Se leen los campos de `--campos` (por defecto `texto`).
Se tokeniza con el prefijo y el tope de tokens del mensaje de L2 (`query: `, 128 por defecto) y se cuenta, por corpus: textos, tokens,
tokens fuera del vocabulario podado (= <unk> tras la poda), tokens que el tokenizador ya marcaba <unk>, textos con algún token podado y
tipos distintos podados. NO se imprime ni se guarda ningún texto, token ni id: con `--anonimizar` ni siquiera los nombres de fichero.

`--barrido`: para cada tamaño N reconstruye un vocabulario con el criterio de la poda (ids del corpus de poda por frecuencia, luego
piezas latinas por id hasta N; sin `--textos-poda` el corpus de poda es el propio corpus evaluado y el resultado es optimista: se
marca `en_muestra`) y da la fracción de <unk> de cada corpus, el tamaño de la tabla de embeddings (N × 384) y el menor N cuyo
peor corpus queda ≤ `--tope` (2 % por defecto). Si ningún N de la rejilla llega, la recomendación es ampliar el vocabulario con los
tokens de los textos del propio cliente (`exportar.ampliar_vocab`, `--vocab-ampliar`) en lugar de subir N.
"""
import argparse
import gzip
import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TOPE = 0.02
DIM = 384
OBJETIVO_PODA = 55_000
BYTES = {"fp32": 4, "fp16": 2, "int8": 1}


# ---------------------------------------------------------------- lectura
def ficheros(ruta):
    p = Path(ruta)
    if p.is_file():
        return [p]
    fs = sorted(x for x in p.rglob("*") if x.is_file() and (x.name.endswith(".jsonl") or x.name.endswith(".jsonl.gz")))
    if not fs:
        raise SystemExit("tasa_unk: %s no contiene ficheros .jsonl ni .jsonl.gz" % ruta)
    return fs


def leer_textos(fichero, campos=("texto",)):
    """Textos de un jsonl (o .gz) en streaming: los campos pedidos de cada fila, los que sean cadena no vacía. Líneas rotas se cuentan, no se imprimen."""
    abrir = gzip.open if str(fichero).endswith(".gz") else open
    rotas = 0
    with abrir(fichero, "rt", encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            try:
                r = json.loads(linea)
            except json.JSONDecodeError:
                rotas += 1
                continue
            if not isinstance(r, dict):
                continue
            for c in campos:
                v = r.get(c)
                if isinstance(v, str) and v.strip():
                    yield v
    if rotas:
        print("tasa_unk: %d líneas no son JSON en un fichero (omitidas)" % rotas, file=sys.stderr)


def cargar_tokenizador(ruta):
    """`tokenizers.Tokenizer` desde un tokenizer.json (o un directorio que lo contenga o un snapshot de HF). Sin red."""
    from tokenizers import Tokenizer
    p = Path(ruta)
    if p.is_dir():
        p = p / "tokenizer.json"
    return Tokenizer.from_file(str(p))


def ids_de(tok, textos, prefijo="query: ", max_tok=128, lote=512):
    """Generador de listas de ids (con especiales, truncadas a `max_tok`) para `textos`."""
    tok.no_truncation()
    tok.enable_truncation(max_length=max_tok)
    buf = []
    for t in textos:
        buf.append(prefijo + t)
        if len(buf) >= lote:
            for e in tok.encode_batch(buf):
                yield e.ids
            buf = []
    if buf:
        for e in tok.encode_batch(buf):
            yield e.ids


# ---------------------------------------------------------------- medida
def medir_corpus(listas, conservados, unk):
    """Agregados de un corpus: `listas` son las listas de ids por texto; `conservados` el conjunto de ids del vocabulario podado."""
    n_textos = n_tok = fuera = unk_nativo = textos_con = 0
    tipos = set()
    for ids in listas:
        n_textos += 1
        n_tok += len(ids)
        f = [i for i in ids if i not in conservados]
        fuera += len(f)
        unk_nativo += sum(1 for i in ids if i == unk)
        textos_con += bool(f)
        tipos.update(f)
    return {"textos": n_textos, "tokens": n_tok, "tokens_podados": fuera, "frac_unk": fuera / max(n_tok, 1),
            "tokens_unk_nativo": unk_nativo, "frac_unk_nativo": unk_nativo / max(n_tok, 1),
            "textos_con_unk": textos_con, "frac_textos_con_unk": textos_con / max(n_textos, 1), "tipos_podados": len(tipos)}


def es_latino(pieza):
    """Copia de `model/bench/codificadores.py:es_latino`: la pieza solo tiene caracteres latinos o combinantes."""
    for c in pieza.replace("▁", "").replace("Ġ", ""):
        if ord(c) < 128:
            continue
        try:
            if not unicodedata.name(c).startswith(("LATIN", "COMBINING")):
                return False
        except ValueError:
            return False
    return True


def ids_por_frecuencia(listas_corpus):
    """Contador de ids sobre las listas de ids del corpus de poda."""
    c = Counter()
    for ids in listas_corpus:
        c.update(ids)
    return c


def vocab_de_tamano(tok, n, frec, especiales, n_vocab):
    """Conjunto de ids de un vocabulario podado de tamaño `n`: especiales, ids del corpus de poda por frecuencia descendente (empate: id menor)
    y el resto de piezas latinas por id hasta `n`. Con `n` menor que lo que usa el corpus, los menos frecuentes quedan fuera."""
    usados = list(dict.fromkeys(list(especiales) + [i for i, _ in sorted(frec.items(), key=lambda kv: (-kv[1], kv[0])) if i not in especiales]))
    if len(usados) >= n:
        return set(usados[:n])
    out = set(usados)
    for i in range(n_vocab):
        if len(out) >= n:
            break
        if i not in out and es_latino(tok.id_to_token(i) or "\u0000x"):
            out.add(i)
    return out


def recomendar(barrido, tope=TOPE):
    """barrido: lista de {n, peor_frac_unk, …} → (n elegido o None, texto)."""
    for b in sorted(barrido, key=lambda x: x["n"]):
        if b["peor_frac_unk"] <= tope:
            return b["n"], "el menor N de la rejilla con el peor corpus ≤ %.1f %% de <unk> es %d" % (100 * tope, b["n"])
    return None, ("ningún N de la rejilla deja el peor corpus ≤ %.1f %%: subir N no basta; ampliar el vocabulario con los tokens de los textos del "
                  "propio cliente (`exportar.ampliar_vocab`) y repetir" % (100 * tope))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--textos", required=True, help="jsonl, jsonl.gz o directorio (cada fichero es un corpus)")
    ap.add_argument("--vocab", default=str(Path(__file__).resolve().parents[1] / "bench" / "vocab-podado-e5-small.json"),
                    help="JSON de la poda (`ids_originales`, `unk`, `original`); por defecto el de e5-small de 55 000")
    ap.add_argument("--tokenizador", required=True, help="tokenizer.json de e5 (o su directorio)")
    ap.add_argument("--campos", default="texto", help="campos de texto, separados por comas")
    ap.add_argument("--prefijo", default="query: ")
    ap.add_argument("--max-tok", type=int, default=128)
    ap.add_argument("--anonimizar", action="store_true", help="llama corpus_1, corpus_2… en vez de usar el nombre del fichero")
    ap.add_argument("--barrido", type=int, nargs="*", default=[], help="tamaños de vocabulario a evaluar")
    ap.add_argument("--textos-poda", default=None, help="corpus con el que se construiría cada vocabulario del barrido (por defecto el evaluado: optimista)")
    ap.add_argument("--tope", type=float, default=TOPE, help="fracción de <unk> aceptable (0,02)")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args(argv)
    vocab = json.load(open(a.vocab, encoding="utf-8"))
    conservados, unk = set(vocab["ids_originales"]), vocab["unk"]
    tok = cargar_tokenizador(a.tokenizador)
    campos = [c for c in a.campos.split(",") if c]
    res = {"vocab": {"original": vocab["original"], "conservados": len(conservados), "unk": unk}, "max_tok": a.max_tok, "prefijo_vacio": not a.prefijo,
           "corpus": {}}
    ids_corpus = {}
    for i, f in enumerate(ficheros(a.textos), 1):
        nombre = "corpus_%d" % i if a.anonimizar else f.name
        listas = list(ids_de(tok, leer_textos(f, campos), a.prefijo, a.max_tok))
        ids_corpus[nombre] = listas
        res["corpus"][nombre] = medir_corpus(listas, conservados, unk)
    tot = {"textos": 0, "tokens": 0, "tokens_podados": 0}
    for r in res["corpus"].values():
        for k in tot:
            tot[k] += r[k]
    tot["frac_unk"] = tot["tokens_podados"] / max(tot["tokens"], 1)
    res["total"] = tot
    res["tope"] = a.tope
    res["por_encima_del_tope"] = sorted(n for n, r in res["corpus"].items() if r["frac_unk"] > a.tope)
    if a.barrido:
        poda = None
        if a.textos_poda:
            poda = ids_por_frecuencia(l for f in ficheros(a.textos_poda) for l in ids_de(tok, leer_textos(f, campos), a.prefijo, a.max_tok))
        n_vocab = vocab["original"]
        espec = sorted({unk, 0, 1, 2, 3})
        res["barrido"], res["barrido_en_muestra"] = [], poda is None
        for n in sorted(a.barrido):
            fr = poda if poda is not None else ids_por_frecuencia(l for ls in ids_corpus.values() for l in ls)
            keep = vocab_de_tamano(tok, n, fr, espec, n_vocab)
            por = {nombre: medir_corpus(ls, keep, unk)["frac_unk"] for nombre, ls in ids_corpus.items()}
            res["barrido"].append({"n": n, "frac_unk": por, "peor_frac_unk": max(por.values(), default=0.0),
                                   "tabla_embeddings_mb": {k: n * DIM * b / 1e6 for k, b in BYTES.items()}})
        n_ok, txt = recomendar(res["barrido"], a.tope)
        res["recomendacion"] = {"n": n_ok, "texto": txt + ("; OJO: sin --textos-poda la cifra es en muestra y optimista" if poda is None else "")}
    for nombre, r in res["corpus"].items():
        print("%-24s textos %7d  tokens %9d  <unk> por poda %6.2f %%  (nativo %.2f %%, textos afectados %.1f %%, tipos %d)" % (
            nombre, r["textos"], r["tokens"], 100 * r["frac_unk"], 100 * r["frac_unk_nativo"], 100 * r["frac_textos_con_unk"], r["tipos_podados"]))
    print("TOTAL: %d tokens, %.2f %% a <unk>%s" % (tot["tokens"], 100 * tot["frac_unk"],
                                                   "  → por encima de %.0f %%: %d corpus" % (100 * a.tope, len(res["por_encima_del_tope"])) if res["por_encima_del_tope"] else ""))
    for b in res.get("barrido", []):
        print("N=%7d  peor corpus %.2f %%  tabla fp32 %.0f MB / fp16 %.0f MB" % (b["n"], 100 * b["peor_frac_unk"], b["tabla_embeddings_mb"]["fp32"], b["tabla_embeddings_mb"]["fp16"]))
    if "recomendacion" in res:
        print(res["recomendacion"]["texto"])
    if a.salida:
        Path(a.salida).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
