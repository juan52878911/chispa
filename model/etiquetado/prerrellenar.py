"""Sugerencias de un LLM local (VON de kindling, API OpenAI) para el banco de PQRS: solo para comparar con las humanas al final.

Lee las filas del banco por stdin (jsonl con id y texto) y el esquema de `--esquema`; escribe por stdout una línea
{id, tipo, area, plazo, modelo} por fila, sin texto. Pensado para correr en el host de kindling, en tubería entre las dos microVM,
sin que los textos toquen el disco:

    kling exec banco-pqrs -- cat /home/worker/banco/datos/muestra.jsonl \\
      | python3 prerrellenar.py --url http://IP_VON:8000 --esquema esquema.json \\
      | kling cp - banco-pqrs:/home/worker/banco/datos/sugerencias.jsonl

Solo biblioteca estándar. Una respuesta que no se puede leer, o con un valor fuera del esquema, deja ese campo vacío (null).
El servidor de etiquetado no sirve nunca estas sugerencias.
"""
import argparse
import json
import re
import sys
import urllib.request


def prompt(esquema, fila):
    partes = ["Clasifica esta solicitud ciudadana (PQRS) dirigida a una entidad pública colombiana. Responde SOLO con un objeto JSON "
              "con las claves %s y como valor el id de una opción." % ", ".join('"%s"' % c["campo"] for c in esquema["campos"])]
    for c in esquema["campos"]:
        partes.append("\n%s (%s):" % (c["campo"], c["pregunta"]))
        partes += ["- %s: %s" % (o["id"], o["desc"]) for o in c["opciones"]]
    partes.append("\nEntidad: %s\nTexto: %s" % (fila.get("entidad", ""), fila["texto"]))
    return "\n".join(partes)


def leer_respuesta(texto, esquema):
    m = re.search(r"\{.*\}", texto, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        d = {}
    out = {}
    for c in esquema["campos"]:
        v = str(d.get(c["campo"], "")).strip()
        out[c["campo"]] = v if v in {o["id"] for o in c["opciones"]} else None
    return out


def preguntar(url, texto, max_tokens=60, timeout=300):
    cuerpo = {"messages": [{"role": "user", "content": texto}], "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(cuerpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--esquema", required=True)
    ap.add_argument("--modelo", default="qwen3-4b-instruct-2507-q4_k_m")
    a = ap.parse_args(argv)
    with open(a.esquema, encoding="utf-8") as f:
        esquema = json.load(f)
    n = vacias = 0
    for linea in sys.stdin:
        if not linea.strip():
            continue
        fila = json.loads(linea)
        try:
            r = leer_respuesta(preguntar(a.url, prompt(esquema, fila)), esquema)
        except OSError:
            r = {c["campo"]: None for c in esquema["campos"]}
        vacias += sum(v is None for v in r.values())
        n += 1
        print(json.dumps(dict(id=fila["id"], modelo=a.modelo, **r), ensure_ascii=False), flush=True)
        if n % 25 == 0:
            print("%d filas, %d campos vacíos" % (n, vacias), file=sys.stderr, flush=True)
    print("fin: %d filas, %d campos vacíos" % (n, vacias), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
