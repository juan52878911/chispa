"""Techo local: cliente de logprobs para el servidor vLLM propio. Dos formatos, el mismo bucle de `llm_logprobs` (ejemplos, demos por fila, relleno por prefijo, salida).

    python -m model.alt.techo_local_cliente [--formato chat|harmony] <argumentos de llm_logprobs>

  chat     (Qwen3-4B y Qwen3-30B-A3B): es `llm_logprobs.main` tal cual; el 4B y el grande pasan por el MISMO código.
  harmony  (gpt-oss): el modelo no escribe la etiqueta como primer token de /v1/chat/completions (abre el canal `analysis` y razona), así que el primer token de
           la etiqueta NO es comparable. Sustituto: se renderiza a mano la conversación en formato harmony, se cierra con `<|start|>assistant<|channel|>final<|message|>`
           (se salta el razonamiento: razonamiento «apagado» a la fuerza) y se lee el top 20 del primer token con /v1/completions. Las etiquetas fuera del top 20 se
           puntúan por prefijo igual que en `llm_logprobs` (el prompt termina en la etiqueta y se lee el logprob de ese último token). Lo único que cambia respecto del
           camino chat es cómo se llega al primer token; probs, piso y salida son los de `llm_logprobs`. Limitación declarada: gpt-oss está entrenado para razonar antes
           de responder; forzar el canal final lo puede dejar por debajo de lo que daría con razonamiento (que aquí no se mide: no hay logprob de opción tras razonar).
"""
import argparse
import json
import re
import sys

from model.alt import llm_logprobs as ll

FECHA = "2025-06-28"   # fija: la plantilla oficial mete la fecha de hoy y eso haría el prompt dependiente del día
SISTEMA_H = ("You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: %s\n\nReasoning: low\n\n"
             "# Valid channels: analysis, commentary, final. Channel must be included for every message." % FECHA)
CANAL_FINAL = "<|start|>assistant<|channel|>final<|message|>"


def renderizar_harmony(msgs):
    """Texto harmony de [{role, content}]: mensaje de sistema fijo, el `system` de llm_logprobs como `developer`, turnos user/assistant y el prefijo del canal final.
    Si el último mensaje es del asistente (relleno por prefijo), se queda abierto: el prompt termina en su contenido, sin `<|end|>`."""
    out = ["<|start|>system<|message|>%s<|end|>" % SISTEMA_H]
    resto = list(msgs)
    if resto and resto[0]["role"] == "system":
        out.append("<|start|>developer<|message|># Instructions\n\n%s<|end|>" % resto.pop(0)["content"])
    abierto = bool(resto) and resto[-1]["role"] == "assistant"
    for i, m in enumerate(resto):
        if m["role"] == "user":
            out.append("<|start|>user<|message|>%s<|end|>" % m["content"])
        elif m["role"] == "assistant":
            fin = "" if (abierto and i == len(resto) - 1) else "<|end|>"
            out.append("%s%s%s" % (CANAL_FINAL, m["content"], fin))
        else:
            raise ValueError("rol %r no soportado en harmony" % m["role"])
    if not abierto:
        out.append(CANAL_FINAL)
    return "".join(out)


def post_comp(url, cuerpo, timeout):
    import urllib.request
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(cuerpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def primer_token_h(url, modelo, msgs, timeout, sin_pensar):
    """([(token, logprob)] del top 20 del primer token del canal final, token generado)."""
    r = post_comp(url, {"model": modelo, "prompt": renderizar_harmony(msgs), "temperature": 0.0, "max_tokens": 1, "logprobs": ll.TOP, "stream": False}, timeout)["choices"][0]
    lp = r["logprobs"]
    return [(t, float(v)) for t, v in lp["top_logprobs"][0].items()], (lp.get("tokens") or [""])[0]


def prefijo_h(url, modelo, msgs, cod, timeout, sin_pensar):
    """logprob de la etiqueta `cod` como primer token del canal final: el prompt termina en la etiqueta y se lee el último token del prompt."""
    cuerpo = {"model": modelo, "prompt": renderizar_harmony(msgs + [{"role": "assistant", "content": cod}]), "temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 0, "stream": False}
    r = post_comp(url, cuerpo, timeout)
    pl = [e for e in (r.get("prompt_logprobs") or r["choices"][0].get("prompt_logprobs") or []) if e]
    for v in pl[-1].values():
        if v.get("decoded_token", "").strip() == cod:
            return float(v["logprob"])
    raise ValueError("prompt_logprobs no trae el token de la etiqueta %r" % cod)


MUESTRA = [{"role": "system", "content": "Instrucción de prueba."}, {"role": "user", "content": "caso uno"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "caso dos"}]


def verificar(aplicar_plantilla):
    """¿Mi render harmony coincide con la plantilla de la tarjeta del modelo? `aplicar_plantilla(msgs)` devuelve el texto de la plantilla con add_generation_prompt y reasoning low.
    Se normaliza la fecha del mensaje de sistema (la plantilla pone la de hoy). Solo informa: una diferencia no para la medida, pero queda en cifras y en el informe."""
    norm = lambda s: re.sub(r"Current date: [0-9-]+", "Current date: X", s)  # noqa: E731
    esperado = norm(aplicar_plantilla(MUESTRA)) + "<|channel|>final<|message|>"
    mio = norm(renderizar_harmony(MUESTRA))
    i = next((k for k, (x, y) in enumerate(zip(esperado, mio)) if x != y), min(len(esperado), len(mio)))
    return {"coincide": esperado == mio, "primera_diferencia": None if esperado == mio else {"posicion": i, "plantilla": esperado[max(0, i - 40):i + 60], "mio": mio[max(0, i - 40):i + 60]}}


def verificar_modelo(directorio, salida):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(directorio)
    r = verificar(lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, reasoning_effort="low"))
    with open(salida, "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1, ensure_ascii=False)
    print("plantilla harmony:", "coincide" if r["coincide"] else "DIFIERE %s" % r["primera_diferencia"])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--verificar-harmony"]:
        return verificar_modelo(argv[1], argv[2])   # DIR_MODELO SALIDA.json
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--formato", choices=["chat", "harmony"], default="chat")
    a, resto = ap.parse_known_args(argv)
    if a.formato == "harmony":
        ll.primer_token, ll.prefijo = primer_token_h, prefijo_h
    ll.main(resto)


if __name__ == "__main__":
    main()
