"""Etiquetas blandas de Qwen3-4B (vLLM) para el train de la alternativa (b): solo filas de los ficheros del manifiesto.

Mismo prompt y misma puntuación que `model/maestro/blandas.py` (logprob del primer token de cada letra, softmax sobre las letras
de las opciones), pero con vLLM para que quepan decenas de miles de filas en una hora de g5. vLLM devuelve las `--top-logprobs`
mejores: una letra fuera de ellas recibe el logprob mínimo devuelto menos 5 (masa ≈ 0). Se etiquetan solo filas train con ≤ 26
opciones que `entrenar.tope_por_tarea` va a escoger (mismo orden de carga y semilla), una muestra de `--max-filas`; el resto
entrena solo con la etiqueta dura. Reescribe en su sitio los ficheros del manifiesto añadiendo `blanda` (el sellado viene después,
en `entrenar --sellar`). Las PQRS y lo que no admita el manifiesto no pasan nunca por aquí. `--guardar F` deja las etiquetas en F
({id, blanda} por línea) para subirlas; `--aplicar F` las reescribe sin vLLM (otra g5 con el mismo manifiesto no repite la destilación).

    python -m model.alt.destilar --datos RAIZ --modelo Qwen/Qwen3-4B-Instruct-2507 --max-filas 150000 --minutos 55 --resumen R.json [--guardar F]
    python -m model.alt.destilar --datos RAIZ --aplicar F --resumen R.json"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from model.l2.datos import iterar
from model.maestro.blandas import LETRAS, construir_prompt, ids_letras

from .manifiesto import cargar, verificar_filas

MODELO = "Qwen/Qwen3-4B-Instruct-2507"   # Apache-2.0
MAX_CHARS_TEXTO = 3000
MAX_LEN = 4096
MAX_TOKENS_PROMPT = MAX_LEN - 8   # margen para el token generado


def tope_por_tarea(filas, tope, semilla=0):
    """Copia de `entrenar.tope_por_tarea` (aquel módulo importa torch y este corre en el venv de vLLM)."""
    if not tope:
        return filas
    rng, por = random.Random(semilla), {}
    for r in filas:
        por.setdefault(r["tarea"], []).append(r)
    out = []
    for t in sorted(por):
        g = por[t]
        rng.shuffle(g)
        out += g[:tope]
    return out


def elegir(manifiesto, raiz, max_filas, tope_tarea, semilla=0):
    """Ids a etiquetar: train escogido por el tope por tarea, con 2..26 opciones; muestra determinista de `max_filas`."""
    train = [r for c in manifiesto.conjuntos for r in iterar([str(Path(raiz) / c["fichero"])]) if r["split"] == "train"]
    verificar_filas(train, manifiesto)
    aptas = [r for r in tope_por_tarea(train, tope_tarea, semilla) if 2 <= len(r["opciones"]) <= len(LETRAS)]
    random.Random(semilla + 1).shuffle(aptas)
    return aptas[:max_filas] if max_filas else aptas


def distribucion(top, opciones, letras_id):
    """{opcion: prob} desde el dict token_id → logprob de vLLM (softmax solo sobre las letras de las opciones)."""
    piso = (min(top.values()) if top else 0.0) - 5.0
    lps = [top.get(letras_id[i], piso) for i in range(len(opciones))]
    m = max(lps)
    es = [math.exp(x - m) for x in lps]
    z = sum(es)
    return {o: e / z for o, e in zip(opciones, es)}


def recortar(fila):
    return dict(fila, texto=fila["texto"][:MAX_CHARS_TEXTO])


def etiquetar(filas, modelo, minutos, top_logprobs=20, trozo=4096, log=print):
    # La DLAMI no trae nvcc: el muestreador de FlashInfer se compila con JIT y tumbaba el motor; modo eager sin grafos CUDA.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    llm = LLM(model=modelo, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.85, enable_prefix_caching=True,
              enforce_eager=True)
    tok = llm.get_tokenizer()
    letras_id = ids_letras(tok)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=top_logprobs)
    t0, out, largas = time.time(), {}, 0
    for i in range(0, len(filas), trozo):
        if (time.time() - t0) / 60 > minutos:
            log("tope de %.0f min: %d de %d filas etiquetadas" % (minutos, len(out), len(filas)))
            break
        lote, prompts = [], []
        for f in filas[i:i + trozo]:   # opciones y descripciones largas pasan de 4096 aunque el texto vaya recortado: esas se saltan
            pr = construir_prompt(tok, recortar(f), f["opciones"])
            if len(tok.encode(pr, add_special_tokens=False)) < MAX_TOKENS_PROMPT:
                lote.append(f)
                prompts.append(pr)
            else:
                largas += 1
        res = llm.generate(prompts, sp, use_tqdm=False) if prompts else []
        for f, r in zip(lote, res):
            lp = r.outputs[0].logprobs[0] if r.outputs and r.outputs[0].logprobs else {}
            out[f["id"]] = distribucion({k: v.logprob for k, v in lp.items()}, f["opciones"], letras_id)
        log("%d/%d filas, %d saltadas por largas, %.1f filas/s" % (len(out), len(filas), largas, len(out) / max(time.time() - t0, 1e-9)))
    return out, time.time() - t0, largas


def reescribir(manifiesto, raiz, blandas):
    """Añade `blanda` a las filas etiquetadas de cada fichero del manifiesto (gzip sin fecha: el sha256 solo depende de las filas)."""
    n = 0
    for c in manifiesto.conjuntos:
        p = Path(raiz) / c["fichero"]
        filas = list(iterar([str(p)]))
        for r in filas:
            if r["id"] in blandas and r["split"] == "train":
                r["blanda"] = blandas[r["id"]]
                n += 1
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fb, gzip.GzipFile(fileobj=fb, mode="wb", mtime=0) as gz:
            for r in filas:
                gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
        tmp.replace(p)
    return n


def guardar(blandas, ruta):
    with open(ruta, "wb") as fb, gzip.GzipFile(fileobj=fb, mode="wb", mtime=0) as gz:
        for i, b in blandas.items():
            gz.write((json.dumps({"id": i, "blanda": b}, ensure_ascii=False) + "\n").encode("utf-8"))


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return {r["id"]: r["blanda"] for r in map(json.loads, f)}


def aplicar(m, raiz, ruta, resumen, minimo=0.99):
    """Reescribe con etiquetas ya destiladas. Falla (1) si menos del 99 % encuentra su fila train: los datos no son los de entonces."""
    blandas = leer(ruta)
    escritas = reescribir(m, raiz, blandas)
    res = {"reutilizadas": True, "en_fichero": len(blandas), "escritas": escritas}
    Path(resumen).write_text(json.dumps(res, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0 if blandas and escritas >= minimo * len(blandas) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datos", required=True, help="raíz con los ficheros del manifiesto")
    ap.add_argument("--manifiesto")
    ap.add_argument("--modelo", default=MODELO)
    ap.add_argument("--max-filas", type=int, default=150000)
    ap.add_argument("--tope-tarea", type=int, default=1000)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--minutos", type=float, default=55)
    ap.add_argument("--resumen", required=True)
    ap.add_argument("--guardar", help="fichero .jsonl.gz donde dejar las etiquetas")
    ap.add_argument("--minimo-escritas", type=float, default=0.99, help="con --aplicar: fracción mínima de las etiquetas que debe encontrar su fila (0 = no exigir)")
    ap.add_argument("--aplicar", help="fichero .jsonl.gz con etiquetas ya destiladas: no se usa vLLM")
    a = ap.parse_args(argv)
    m = cargar(a.manifiesto) if a.manifiesto else cargar()
    if a.aplicar:
        return aplicar(m, a.datos, a.aplicar, a.resumen, a.minimo_escritas)
    filas = elegir(m, a.datos, a.max_filas, a.tope_tarea, a.semilla)
    print("a etiquetar: %d filas" % len(filas), flush=True)
    blandas, seg, largas = etiquetar(filas, a.modelo, a.minutos, log=lambda s: print(s, flush=True))
    escritas = reescribir(m, a.datos, blandas)
    if a.guardar and blandas:
        guardar(blandas, a.guardar)
    por_id = {f["id"]: f for f in filas}
    acuerdo = [max(b, key=b.get) == por_id[i]["etiqueta"] for i, b in blandas.items() if isinstance(por_id[i].get("etiqueta"), str)]
    pct = round(100 * largas / len(filas), 2) if filas else 0.0
    res = {"elegidas": len(filas), "etiquetadas": len(blandas), "saltadas_largas": largas, "pct_saltadas": pct,
           "aviso_saltadas": pct > 5, "completa": len(blandas) + largas == len(filas),
           "modelo": a.modelo, "licencia_modelo": "apache-2.0", "escritas": escritas, "segundos": round(seg, 1), "filas_por_s": round(len(blandas) / max(seg, 1e-9), 2),
           "acuerdo_top1": round(sum(acuerdo) / len(acuerdo), 4) if acuerdo else None,
           "por_tarea": dict(sorted(__import__("collections").Counter(por_id[i]["tarea"] for i in blandas).most_common(20)))}
    Path(a.resumen).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0 if blandas else 1


if __name__ == "__main__":
    sys.exit(main())
