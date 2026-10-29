#!/usr/bin/env python3
"""Exporta el tokenizador BPE byte-level de Qwen2/Qwen3 (el EFECTIVO de transformers, con su plantilla de especiales) al `.chispa-bpe` que lee bpe.go.

  python runtime/tok/exportar_bpe.py DIR_MODELO SALIDA.chispa-bpe

DIR_MODELO es un directorio con tokenizer.json (o vocab.json + merges.txt) y tokenizer_config.json (p. ej. Qwen3-Embedding-0.6B en la revisión
fijada). Se carga con AutoTokenizer(padding_side="left", local_files_only=True) y se lee el tokenizador de Rust que usa transformers, de modo que
lo que se exporta es lo que transformers aplica (incluido el `<|endoftext|>` final si lo añade). El formato es JSON compacto de claves ordenadas:
determinista (misma entrada → mismo sha256). Falla si el tokenizador no es el esperado (normalizador, regex, ignore_merges, añadidos con lstrip…).
"""
import hashlib
import json
import sys

REGEX_QWEN2 = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def bytes_a_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def exigir(c, msg):
    if not c:
        raise SystemExit("exportar_bpe: " + msg)


def plantilla(pp, nombre_a_id):
    """[{secuencia: true} | {id: n}] de `single` de un TemplateProcessing (o de uno dentro de un Sequence); None = sin plantilla."""
    if pp is None:
        return [{"secuencia": True}]
    tipo = pp["type"]
    if tipo == "ByteLevel":
        return [{"secuencia": True}]
    if tipo == "Sequence":
        usa = [p for p in pp["processors"] if p["type"] != "ByteLevel"]
        exigir(len(usa) <= 1, "Sequence de post-procesadores con más de un TemplateProcessing")
        return plantilla(usa[0] if usa else None, nombre_a_id)
    exigir(tipo == "TemplateProcessing", "post-procesador %r no soportado" % tipo)
    out, vio = [], False
    for it in pp["single"]:
        if "Sequence" in it:
            exigir(it["Sequence"]["id"] == "A" and not vio, "plantilla single con secuencia distinta de una A")
            vio = True
            out.append({"secuencia": True})
        else:
            nombre = it["SpecialToken"]["id"]
            exigir(nombre in pp["special_tokens"], "especial %r sin definir" % nombre)
            out.append({"id": int(pp["special_tokens"][nombre]["ids"][0])})
    exigir(vio, "la plantilla single no lleva la secuencia")
    return out


def exportar(dir_modelo, salida):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(dir_modelo, padding_side="left", local_files_only=True)
    j = json.loads(tk.backend_tokenizer.to_str())
    exigir(j["normalizer"] == {"type": "NFC"}, "normalizador %r (se espera NFC)" % (j["normalizer"],))
    pre = j["pre_tokenizer"]
    exigir(pre["type"] == "Sequence" and len(pre["pretokenizers"]) == 2, "pre_tokenizer no es Sequence[Split, ByteLevel]")
    sp, bl = pre["pretokenizers"]
    exigir(sp["type"] == "Split" and sp["pattern"] == {"Regex": REGEX_QWEN2} and sp["behavior"] == "Isolated" and not sp["invert"], "el Split no es el regex de Qwen2")
    exigir(bl["type"] == "ByteLevel" and not bl["add_prefix_space"] and not bl["use_regex"], "ByteLevel con add_prefix_space o use_regex")
    m = j["model"]
    exigir(m["type"] == "BPE" and m.get("dropout") is None and m.get("unk_token") is None and not m.get("continuing_subword_prefix") and not m.get("end_of_word_suffix")
           and not m.get("fuse_unk") and not m.get("byte_fallback") and not m.get("ignore_merges"), "modelo BPE con opciones no soportadas: %r" % {k: v for k, v in m.items() if k not in ("vocab", "merges")})
    u2b = {c: b for b, c in bytes_a_unicode().items()}
    vocab = m["vocab"]
    n = max(vocab.values()) + 1
    lista = [""] * n
    for t, i in vocab.items():
        exigir(all(c in u2b for c in t), "token %r con caracteres fuera del alfabeto byte-level" % t)
        exigir(lista[i] == "", "id %d repetido" % i)
        lista[i] = bytes(u2b[c] for c in t).hex()
    exigir(len(set(x for x in lista if x)) == len([x for x in lista if x]), "dos ids con los mismos bytes")
    fus = []
    for mg in m["merges"]:
        a, b = mg.split(" ") if isinstance(mg, str) else mg
        fus += [vocab[a], vocab[b]]
    anad = []
    for a in j["added_tokens"]:
        exigir(not (a["lstrip"] or a["rstrip"] or a["single_word"]), "añadido %r con lstrip/rstrip/single_word" % a["content"])
        anad.append({"id": a["id"], "contenido": a["content"], "especial": bool(a["special"])})
    anad.sort(key=lambda a: a["id"])
    nombre_a_id = {a["contenido"]: a["id"] for a in anad}
    pad = tk.pad_token_id
    exigir(pad is not None, "el tokenizador no tiene pad_token_id")
    out = {"formato": "chispa-bpe-1", "normalizador": "NFC", "regex": "qwen2", "vocab": lista, "fusiones": fus, "anadidos": anad,
           "plantilla": plantilla(j["post_processor"], nombre_a_id), "pad": int(pad)}
    data = (json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    open(salida, "wb").write(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "vocab": len([x for x in lista if x]), "fusiones": len(fus) // 2, "anadidos": len(anad),
            "plantilla": out["plantilla"], "pad": out["pad"]}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(exportar(sys.argv[1], sys.argv[2])))
