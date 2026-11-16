"""R5: el LLM de L3 puntuado por logprobs en vez de pedirle un JSON de probabilidades (alternativa a `llm_jevbench.py`).

Solo biblioteca estándar (corre en el anfitrión de las réplicas vLLM, igual que `llm_jevbench.py`).

    python3 llm_logprobs.py --filas filas.jsonl --url http://IP1:8000 [--url http://IP2:8000] --modelo qwen3-1.7b --salida l3.jsonl [--sin-pensar]

Por qué: el formato JSON con salida restringida por esquema (v2 de `llm_jevbench.py`) parece castigar a los modelos pequeños (el 0.6B dice
`false` en todo hatecheck; con 60 opciones en massive-es dejan inválidas casi todas). Aquí el modelo no escribe números: se le pide una
sola etiqueta y se lee la distribución del PRIMER token que habría generado.

Método elegido (token de etiqueta + logprobs, con relleno por puntuación del prefijo):
  1. Cada opción recibe una etiqueta de UN carácter: A-Z, a-z, 0-9 (62; en Qwen cada carácter ASCII suelto es un solo token, así que el
     primer token ES la etiqueta entera y no hay que sumar secuencias). En tipo `score` con niveles de una cifra la etiqueta es el propio
     nivel. Más de 62 opciones: la fila sale inválida (la batería llega a 60, massive-es).
  2. Una llamada a /v1/chat/completions con `max_tokens=1`, `logprobs=true`, `top_logprobs=20`: la lista de los 20 tokens más probables
     del primer token de la respuesta. Las etiquetas que salen ahí dan su logprob (si salen " A" y "A", se suman las probabilidades).
  3. Las etiquetas que NO salen en el top 20 se puntúan exactas con `prompt_logprobs` (sobre la misma conversación con el mensaje final del
     asistente = la etiqueta, `add_generation_prompt=false` y `continue_final_message=true`; se lee el logprob de ese último token del
     prompt). Como cada una es una llamada completa, solo se hacen las primeras `--max-extra` (por defecto 4); las demás reciben un piso
     (el menor logprob conocido menos 1 nat), que es una cota superior razonable: por no estar en el top 20 pesan menos que el vigésimo.
     Así ninguna opción queda a 0 y el orden entre las del top nunca depende del relleno.
  4. probs = softmax de esos logprobs sobre las opciones (se descarta la masa que el modelo ponga fuera de las etiquetas).
     `valida` solo es falso si ninguna opción se pudo puntuar (error de red, o el modelo no puso ninguna etiqueta en su top 20 y el relleno falló).
  Se descartó: puntuar cada opción entera con `/v1/completions` + `echo` (n llamadas por fila, 60 en massive-es) y etiquetas de dos cifras
  (dos tokens: el primero no basta y habría que encadenar llamadas).

Parámetros de la API OpenAI de vLLM que se usan (documentados en la página «OpenAI-compatible server»): `logprobs`, `top_logprobs` (chat),
`prompt_logprobs`, `add_generation_prompt`, `continue_final_message` y `chat_template_kwargs` (extras de chat). El servidor debe arrancar con
`--max-logprobs 20` como mínimo (es el valor por defecto de vLLM; el .sh lo pasa explícito).

Misma entrada que `llm_jevbench.py` (filas.jsonl: id, tipo, pregunta, opciones, descripciones, texto) y MISMA salida por fila:
{id, probs, valida, raw, intentos, ms, error}; `llm_bateria medir/pequeno` la lee sin cambios. `raw` aquí es un JSON con el token generado,
cuántas etiquetas salieron del top, cuántas se puntuaron por prefijo y cuántas recibieron el piso. Reanuda: salta los ids ya escritos.
El prompt es el de `llm_jevbench` (mismo aviso sobre el texto como dato) cambiando solo la instrucción de formato de respuesta.
Las respuestas de Jev en JevBench solo sirven para medir (MCA 2.3(b)): este script no escribe nada que sirva de entrenamiento.
"""
import argparse
import json
import math
import random
import string
import threading
import time
import urllib.request

try:
    from model.alt import llm_jevbench as jb
except ImportError:   # en el anfitrión corre suelto, junto a llm_jevbench.py
    import llm_jevbench as jb

ETIQUETAS = string.ascii_uppercase + string.ascii_lowercase + string.digits
TOP = 20
INSTRUCCION = ('Cada opción lleva una "etiqueta" de un solo carácter. Responde SOLO con la etiqueta de la opción elegida: un único carácter, '
               'sin comillas, sin explicación y sin JSON. tipo "choice": una opción correcta. tipo "noul": la pregunta es de sí/no; las opciones son "true" y "false". '
               'tipo "score": las opciones son los niveles 1..K, del menor (1) al mayor (K); elige el nivel que mejor corresponda. '
               'tipo "multi": elige la opción más probable.')


def preparar(f):
    """(mensajes, {etiqueta: id de opción}). Reutiliza `mensajes` de llm_jevbench y le añade la etiqueta a cada opción."""
    msgs, n = jb.mensajes(f)
    entrada = json.loads(msgs[1]["content"])
    ids = [o["id"] for o in entrada["opciones"]]
    if f["tipo"] == "score" and all(len(i) == 1 and i.isdigit() for i in ids):
        cods = list(ids)
    elif n <= len(ETIQUETAS):
        cods = list(ETIQUETAS[:n])
    else:
        raise ValueError("%d opciones: más que etiquetas de un carácter (%d)" % (n, len(ETIQUETAS)))
    for o, c in zip(entrada["opciones"], cods):
        o["etiqueta"] = c
    sistema = jb.PROMPT[:jb.PROMPT.index("Responde SOLO")] + INSTRUCCION
    return [{"role": "system", "content": sistema}, {"role": "user", "content": json.dumps(entrada, ensure_ascii=False)}], dict(zip(cods, ids))


NOTA_EJEMPLOS = (' Los turnos previos de usuario/asistente son ejemplos resueltos de la misma tarea: el asistente respondió con la etiqueta correcta de cada texto; '
                 'el último mensaje de usuario es el caso nuevo y trae sus opciones con etiqueta completas.')


def elegir_ejemplos(soporte, k, semilla, excluir=()):
    """k ejemplos del soporte de oro (split train), estratificados por clase y deterministas: las clases se barajan con `semilla` y se toma
    uno por clase, al azar dentro de la clase, vuelta tras vuelta hasta k; el orden final también se baraja con la misma semilla. Nunca filas de test."""
    rng = random.Random(semilla)
    por = {}
    for r in soporte:
        if r.get("split", "train") != "train":
            continue
        if r["id"] in excluir:
            raise ValueError("el ejemplo %s también está en las filas a puntuar" % r["id"])
        por.setdefault(r["etiqueta"], []).append(r)
    cl = sorted(por)
    rng.shuffle(cl)
    for c in cl:
        por[c] = sorted(por[c], key=lambda r: r["id"])
        rng.shuffle(por[c])
    out, i = [], 0
    while len(out) < k and any(por.values()):
        c = cl[i % len(cl)]
        i += 1
        if por[c]:
            out.append(por[c].pop())
    rng.shuffle(out)
    return out


def turnos_ejemplos(ejemplos, max_chars):
    """[mensajes usuario/asistente]: cada ejemplo con las opciones solo como id→etiqueta (sin descripciones: con 59 opciones no cabrían 8) y la etiqueta correcta."""
    out = []
    for e in ejemplos:
        msgs, etq = preparar(e)
        entrada = json.loads(msgs[1]["content"])
        entrada["opciones"] = [{"id": o["id"], "etiqueta": o["etiqueta"]} for o in entrada["opciones"]]
        entrada["texto"] = "‹dato›" + e["texto"][:max_chars].translate(jb.ESCAPE) + "‹/dato›"
        cod = next(c for c, i in etq.items() if i == e["etiqueta"])
        out += [{"role": "user", "content": json.dumps(entrada, ensure_ascii=False)}, {"role": "assistant", "content": cod}]
    return out


def con_ejemplos(msgs, turnos):
    """Mismo sistema (+ la nota), los turnos de ejemplo y el caso nuevo tal cual lo prepara `preparar`."""
    return [{"role": "system", "content": msgs[0]["content"] + NOTA_EJEMPLOS}] + turnos + [msgs[1]]


def post(url, cuerpo, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(cuerpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cuerpo_base(modelo, msgs, sin_pensar):
    c = {"model": modelo, "messages": msgs, "temperature": 0.0, "max_tokens": 1, "stream": False}
    if sin_pensar:   # modelos híbridos (Qwen3-1.7B/0.6B): modo sin pensamiento de su plantilla; el prompt no cambia
        c["chat_template_kwargs"] = {"enable_thinking": False}
    return c


def primer_token(url, modelo, msgs, timeout, sin_pensar):
    """([(token, logprob)] del top 20 del primer token, token generado)."""
    c = cuerpo_base(modelo, msgs, sin_pensar)
    c.update({"logprobs": True, "top_logprobs": TOP})
    r = post(url, c, timeout)["choices"][0]
    cont = r["logprobs"]["content"][0]
    return [(t["token"], float(t["logprob"])) for t in cont["top_logprobs"]], cont.get("token", "")


def prefijo(url, modelo, msgs, cod, timeout, sin_pensar):
    """logprob de la etiqueta `cod` como inicio de la respuesta, por `prompt_logprobs` (la conversación termina en el mensaje del asistente)."""
    c = cuerpo_base(modelo, msgs + [{"role": "assistant", "content": cod}], sin_pensar)
    c.update({"add_generation_prompt": False, "continue_final_message": True, "prompt_logprobs": 0})
    pl = [e for e in post(url, c, timeout)["prompt_logprobs"] if e]
    for v in pl[-1].values():   # el último token del prompt es la etiqueta
        if v.get("decoded_token", "").strip() == cod:
            return float(v["logprob"])
    raise ValueError("prompt_logprobs no trae el token de la etiqueta %r" % cod)


def lse(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def softmax(lp):
    m = max(lp.values())
    e = {k: math.exp(v - m) for k, v in lp.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def puntuar(url, modelo, msgs, etq, timeout, sin_pensar, max_extra):
    """({id: prob} o None, raw, aviso). Ver el docstring del módulo."""
    top, gen = primer_token(url, modelo, msgs, timeout, sin_pensar)
    lp = {}
    for tok, l in top:
        c = tok.strip()
        if c in etq:
            lp[c] = lse(lp[c], l) if c in lp else l
    del_top = len(lp)
    aviso, extra = "", 0
    for c in [c for c in etq if c not in lp][:max_extra]:
        try:
            lp[c] = prefijo(url, modelo, msgs, c, timeout, sin_pensar)
            extra += 1
        except Exception as e:  # noqa: BLE001 - el relleno es opcional: se anota y se sigue
            aviso = "prefijo: " + str(e)[:100]
            break
    raw = {"token": gen, "del_top": del_top, "por_prefijo": extra}
    if not lp:
        return None, json.dumps(raw, ensure_ascii=False), aviso
    piso = min([l for _, l in top] + list(lp.values())) - 1.0
    raw["con_piso"] = sum(1 for c in etq if c not in lp)
    p = softmax({c: lp.get(c, piso) for c in etq})
    return {etq[c]: p[c] for c in etq}, json.dumps(raw, ensure_ascii=False), aviso


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--filas", required=True)
    ap.add_argument("--url", action="append", required=True, help="una por réplica; cada una atiende un hilo")
    ap.add_argument("--modelo", default="qwen3-4b-instruct-2507-q4_k_m")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--reintentos", type=int, default=1, help="reintentos si falla la llamada principal")
    ap.add_argument("--max-extra", type=int, default=4, help="etiquetas fuera del top 20 que se puntúan exactas por prefijo; el resto recibe el piso")
    ap.add_argument("--max", type=int, help="solo los primeros N casos (pruebas)")
    ap.add_argument("--ejemplos", help="few-shot: .jsonl con el soporte de oro de la tarea (split train); por defecto apagado (camino zero-shot intacto)")
    ap.add_argument("--demos", help="D (apagado por defecto; con --ejemplos): .jsonl {id, ejemplos: [ids del soporte]} con las demostraciones de CADA fila (p. ej. las k más cercanas por kNN, calculadas aparte); sustituye a la elección fija")
    ap.add_argument("--k", type=int, default=8, help="nº de ejemplos con --ejemplos")
    ap.add_argument("--semilla", type=int, default=20261115, help="semilla de la elección de ejemplos")
    ap.add_argument("--ejemplo-max-chars", type=int, default=1200, help="tope de caracteres del texto de cada ejemplo")
    ap.add_argument("--sin-pensar", action="store_true", help="chat_template_kwargs enable_thinking=false (Qwen3 híbridos)")
    a = ap.parse_args(argv)
    filas = [json.loads(l) for l in open(a.filas, encoding="utf-8") if l.strip()][:a.max]
    try:
        hechos = {json.loads(l)["id"] for l in open(a.salida, encoding="utf-8") if l.strip()}
    except FileNotFoundError:
        hechos = set()
    cola = [f for f in filas if f["id"] not in hechos]
    turnos = None
    if a.ejemplos:
        sop = [json.loads(l) for l in open(a.ejemplos, encoding="utf-8") if l.strip()]
        if a.demos:
            por_id = {e["id"]: e for e in sop if e.get("split", "train") == "train"}
            demos = {d["id"]: d["ejemplos"] for d in (json.loads(l) for l in open(a.demos, encoding="utf-8") if l.strip())}
            falta = [f["id"] for f in filas if f["id"] not in demos]
            if falta or any(i not in por_id for v in demos.values() for i in v) or any(i in por_id for i in demos):
                raise SystemExit("--demos: faltan filas (%d), hay ids que no son del soporte de oro o una fila a puntuar es del soporte" % len(falta))
            turnos = {k: turnos_ejemplos([por_id[i] for i in v], a.ejemplo_max_chars) for k, v in demos.items()}   # por fila
        else:
            turnos = turnos_ejemplos(elegir_ejemplos(sop, a.k, a.semilla, {f["id"] for f in filas}), a.ejemplo_max_chars)
    cerrojo, out = threading.Lock(), open(a.salida, "a", encoding="utf-8")

    def trabajo(url):
        while True:
            with cerrojo:
                if not cola:
                    return
                f = cola.pop(0)
            t0, err, probs, raw, intentos = time.perf_counter(), "", None, "", 0
            try:
                msgs, etq = preparar(f)
                if turnos is not None:
                    msgs = con_ejemplos(msgs, turnos[f["id"]] if isinstance(turnos, dict) else turnos)
            except ValueError as e:
                msgs, etq, err, intentos = None, None, str(e)[:120], 1
            while msgs is not None and intentos <= a.reintentos and probs is None:
                intentos += 1
                try:
                    probs, raw, aviso = puntuar(url, a.modelo, msgs, etq, a.timeout, a.sin_pensar, a.max_extra)
                    err = aviso if probs is not None else (aviso or "ninguna etiqueta en el top %d" % TOP)
                except Exception as e:  # noqa: BLE001 - se anota y se sigue
                    err = str(e)[:120]
            r = {"id": f["id"], "probs": probs, "valida": probs is not None, "raw": raw, "intentos": intentos, "ms": round((time.perf_counter() - t0) * 1000), "error": err}
            with cerrojo:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()

    hilos = [threading.Thread(target=trabajo, args=(u,)) for u in a.url]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    out.close()


if __name__ == "__main__":
    main()
