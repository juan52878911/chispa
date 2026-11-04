"""Banco colombiano de PQRS (hito 3): muestra, servidor de etiquetado para dos personas a ciegas y κ por campo.

Todo corre DENTRO de kindling: las PQRS reales (`~/datos/datosgov`, ver `model/datos/pqrs_comun.py`) no salen de la microVM ni
del volumen del banco. Ninguna orden imprime textos: `contar` solo da cifras agregadas y `kappa` solo cifras de acuerdo.

    python -m model.etiquetado.banco_pqrs contar
    python -m model.etiquetado.banco_pqrs muestra --dir /banco --n 450 --semilla 7
    python -m model.etiquetado.banco_pqrs servir --dir /banco --puerto 8080
    python -m model.etiquetado.banco_pqrs kappa --dir /banco

`muestra` reparte `--n` filas a partes iguales entre entidades y, dentro de cada una, por turnos entre sus etiquetas originales; lo
que sobra de una entidad pequeña va a las demás; una sola fila por `plantilla` (los textos idénticos tras anonimizar y normalizar) y nada por debajo de
`--min-chars`. El texto ya viene anonimizado por los conversores (`anonimizar.anonimizar`). Escribe `DIR/muestra.jsonl` (id, entidad,
texto, etiqueta_original) y, si no existe, `DIR/esquema.json` (el propuesto en `ESQUEMA`).
`servir` lee `DIR/etiquetadores.json` ({"nombre": "token"}, dos personas) y guarda cada respuesta en
`DIR/etiquetas/<nombre>.jsonl` (la última por id manda). Cada persona solo ve sus propias etiquetas; ni la etiqueta original ni la
sugerencia del modelo (`DIR/sugerencias.jsonl`) se sirven nunca. Escucha en 0.0.0.0 dentro de la microVM (sin salida a la red);
desde fuera se llega por túnel SSH al host de kindling.
`kappa` da, por campo, κ de Cohen entre las dos personas (filas que ambas etiquetaron), el acuerdo bruto y, si hay sugerencias,
κ de cada persona frente al modelo (informativo: el modelo no cuenta como etiquetador).
"""
import argparse
import collections
import hashlib
import html
import json
import random
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from model.datos.pqrs_comun import DESCRIPCIONES_TIPO, TIPOS

# (módulo conversor, nombre de la entidad). Las cuatro fuentes de datos.gov.co que ya tiene la imagen dorada.
FUENTES = (("pqrs_gb9w", "Alcaldía de Flandes (Tolima)"), ("pqrs_e86p", "Instituto Social de Vivienda y Hábitat de Medellín"),
           ("pqrs_xtj2", "Concejo de Pereira"), ("pqrs_enrutamiento", "Dirección General de Sanidad Militar"))

# Esquema propuesto (se copia a DIR/esquema.json; lo que mande es ese fichero). Plazos: Ley 1755 de 2015, art. 14 y 20.
ESQUEMA = {
    "version": 1,
    "campos": [
        {"campo": "tipo", "pregunta": "¿Qué tipo de solicitud ciudadana es?",
         "opciones": [{"id": t, "desc": DESCRIPCIONES_TIPO[t]} for t in TIPOS]
         + [{"id": "no_es_pqrs", "desc": "no es una solicitud ciudadana (spam, prueba, texto sin sentido, trámite interno)"}]},
        {"campo": "area", "pregunta": "¿A qué área o buzón temático debe ir?",
         "opciones": [
             {"id": "salud", "desc": "salud: citas, medicamentos, autorizaciones, atención médica, EPS o sanidad"},
             {"id": "vivienda_obras", "desc": "vivienda, obras, vías, espacio público, servicios públicos domiciliarios"},
             {"id": "hacienda_pagos", "desc": "impuestos, cobros, pagos, facturas, devoluciones de dinero"},
             {"id": "tramites_documentos", "desc": "certificados, constancias, copias, estado de un trámite o radicado"},
             {"id": "convivencia_seguridad", "desc": "ruido, riñas, seguridad, animales, convivencia entre vecinos"},
             {"id": "atencion_funcionarios", "desc": "trato, demora o conducta de funcionarios y de la atención"},
             {"id": "talento_humano", "desc": "empleo, contratos, nómina, hojas de vida, pensiones de la entidad"},
             {"id": "otra", "desc": "cualquier otra área"}]},
        {"campo": "plazo", "pregunta": "¿Qué plazo legal de respuesta aplica (Ley 1755 de 2015)?",
         "opciones": [
             {"id": "prioritario", "desc": "atención prioritaria: riesgo para la vida, la salud o un derecho fundamental (art. 20)"},
             {"id": "10_dias", "desc": "10 días hábiles: pide documentos o información (art. 14.1)"},
             {"id": "15_dias", "desc": "15 días hábiles: petición general, queja, reclamo o denuncia (art. 14)"},
             {"id": "30_dias", "desc": "30 días hábiles: consulta sobre materias a cargo de la entidad (art. 14.2)"},
             {"id": "sin_plazo", "desc": "sin plazo legal: felicitación, sugerencia sin pedido, o no es PQRS"}]},
    ],
}


def _leer_jsonl(p):
    p = Path(p)
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _escribir_jsonl(p, filas):
    with open(p, "w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cargar_fuentes(raiz=None):
    """[(entidad, fila convertida)] de las fuentes disponibles; las que falten se avisan por stderr y se saltan."""
    import importlib
    from model.datos.comun import FuenteAusente
    out = []
    for mod, entidad in FUENTES:
        try:
            filas, _ = importlib.import_module("model.datos." + mod).convertir(raiz)
        except FuenteAusente as e:
            print("falta %s (%s)" % (mod, e), file=sys.stderr)
            continue
        out += [(entidad, r) for r in filas]
    return out


def muestra(pares, n, semilla=7, min_chars=20):
    """pares = [(entidad, fila)] -> filas del banco (una por plantilla, estratificadas por entidad × etiqueta original)."""
    vistas, grupos = set(), collections.defaultdict(list)
    for entidad, r in pares:
        if r["plantilla"] in vistas or len(r["texto"]) < min_chars:
            continue
        vistas.add(r["plantilla"])
        grupos[(entidad, r["etiqueta"])].append({"id": "banco-" + hashlib.sha1(r["id"].encode()).hexdigest()[:12],
                                                 "entidad": entidad, "texto": tapar_numeros(r["texto"]),
                                                 "etiqueta_original": r["etiqueta"], "tarea_original": r["tarea"]})
    # Dos niveles: el cupo se reparte por entidad y, dentro de cada una, por turnos entre sus etiquetas originales (así las 31
    # dependencias de Sanidad Militar no se llevan la mitad del banco).
    rng = random.Random(semilla)
    por_entidad = collections.defaultdict(list)
    for (entidad, _), fs in sorted(grupos.items()):
        fs = fs[:]
        rng.shuffle(fs)
        por_entidad[entidad].append(fs)
    turnos = {}
    for entidad, listas in por_entidad.items():
        rng.shuffle(listas)
        turnos[entidad] = [f for capa in zip_largo(listas) for f in capa]
    out = []
    pend = dict(turnos)
    while len(out) < n and any(pend.values()):
        vivos = [k for k, v in pend.items() if v]
        cupo = max(1, (n - len(out)) // len(vivos))
        for k in vivos:
            toma, pend[k] = pend[k][:cupo], pend[k][cupo:]
            out += toma[: n - len(out)]
            if len(out) >= n:
                break
    rng.shuffle(out)
    return out


def zip_largo(listas):
    """Capas por turno: la primera de cada lista, luego la segunda… (las listas cortas se agotan antes)."""
    for j in range(max((len(x) for x in listas), default=0)):
        yield [x[j] for x in listas if j < len(x)]


NUMEROS = re.compile(r"\d(?:[ .-]?\d){6,}")


def tapar_numeros(texto):
    """Segunda pasada sobre el anonimizado: toda tira de 7 dígitos o más (cédula, teléfono o radicado) pasa a [NUM]."""
    return NUMEROS.sub("[NUM]", texto)


def contar(pares):
    """Cifras agregadas, sin texto: filas y plantillas por entidad, filas por etiqueta original."""
    por = collections.defaultdict(lambda: {"filas": 0, "plantillas": set(), "etiquetas": collections.Counter(), "tarea": ""})
    for entidad, r in pares:
        d = por[entidad]
        d["filas"] += 1
        d["plantillas"].add(r["plantilla"])
        d["etiquetas"][r["etiqueta"]] += 1
        d["tarea"] = r["tarea"]
    # Los tipos de PQRS se cuentan por nombre; las dependencias de enrutamiento, solo cuántas hay.
    return {e: {"tarea": d["tarea"], "filas": d["filas"], "plantillas": len(d["plantillas"]),
                "etiquetas_distintas": len(d["etiquetas"]),
                **({"por_tipo": dict(d["etiquetas"].most_common())} if d["tarea"] == "pqrs" else {})}
            for e, d in sorted(por.items())}


# --- κ por campo

def kappa(dir_):
    from model.liston.acuerdo_humano import cohen
    d = Path(dir_)
    esquema = json.loads((d / "esquema.json").read_text(encoding="utf-8"))
    quien = sorted(json.loads((d / "etiquetadores.json").read_text(encoding="utf-8")))
    ult = {q: {r["id"]: r for r in _leer_jsonl(d / "etiquetas" / (q + ".jsonl"))} for q in quien}
    sug = {r["id"]: r for r in _leer_jsonl(d / "sugerencias.jsonl")}
    out = {"etiquetadores": quien, "filas": {q: len(ult[q]) for q in quien}, "campos": {}}
    if len(quien) != 2:
        return dict(out, error="hacen falta exactamente dos etiquetadores")
    a, b = quien
    for c in (x["campo"] for x in esquema["campos"]):
        par = [(ult[a][i][c], ult[b][i][c]) for i in ult[a] if i in ult[b] and ult[a][i].get(c) and ult[b][i].get(c)]
        res = {"n": len(par), "kappa": cohen([list(p) for p in par]) if par else None,
               "acuerdo": round(sum(x == y for x, y in par) / len(par), 4) if par else None}
        for q in quien:
            pm = [(ult[q][i][c], sug[i][c]) for i in ult[q] if i in sug and ult[q][i].get(c) and sug[i].get(c)]
            res["kappa_%s_vs_modelo" % q] = cohen([list(p) for p in pm]) if pm else None
        out["campos"][c] = res
    return out


# --- servidor

GUIA = Path(__file__).with_name("GUIA_BANCO_PQRS.md")

PAGINA = """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Banco PQRS</title><style>
body{font:16px/1.45 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px;background:#fafafa;color:#222}
#texto{white-space:pre-wrap;background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:10px 0}
.campo{margin:14px 0}.campo h3{margin:0 0 6px;font-size:15px}
button.op{display:block;width:100%;text-align:left;margin:3px 0;padding:7px 10px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
button.op.sel{background:#1f6feb;color:#fff;border-color:#1f6feb}.barra{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.meta{color:#666;font-size:14px}
</style><body><div class="barra"><b>Banco PQRS</b><span class="meta" id="prog"></span>
<button onclick="mover(-1)">← anterior</button><button onclick="mover(1)">siguiente →</button>
<button onclick="saltar()">primera sin etiquetar</button><a href="guia?t=" onclick="this.href='guia?t='+encodeURIComponent(T)" target="_blank">guía</a><label><input type="checkbox" id="duda"> dudosa (?)</label><label><input type="checkbox" id="dp"> dato personal visible (p)</label></div>
<div class="meta" id="entidad"></div><div id="texto"></div><div id="campos"></div>
<p class="meta">Atajos: 1-9 elige en el campo activo (Tab cambia de campo) · ← → navegan · ? marca dudosa · p marca un nombre, documento o dirección que el anonimizado no tapó. Se guarda al elegir.
Etiqueta por tu cuenta, sin consultar a la otra persona.</p>
<script>
const T=new URLSearchParams(location.search).get("t");let E,F,M={},i=0,activo=0;
async function api(p,b){const r=await fetch(p+(p.includes("?")?"&":"?")+"t="+encodeURIComponent(T),b?{method:"POST",body:JSON.stringify(b)}:{});if(!r.ok)throw new Error(await r.text());return r.json()}
async function ini(){const d=await api("api/datos");E=d.esquema;F=d.filas;M=d.mias;saltar()}
function pintar(){const f=F[i],m=M[f.id]||{};document.getElementById("prog").textContent=`${i+1}/${F.length} · hechas ${Object.keys(M).length}`;
document.getElementById("entidad").textContent=f.entidad;document.getElementById("texto").textContent=f.texto;
document.getElementById("duda").checked=!!m.dudosa;document.getElementById("dp").checked=!!m.dato_personal;const c=document.getElementById("campos");c.innerHTML="";
E.campos.forEach((k,j)=>{const d=document.createElement("div");d.className="campo";d.innerHTML=`<h3>${j===activo?"▶ ":""}${k.pregunta}</h3>`;
k.opciones.forEach((o,n)=>{const b=document.createElement("button");b.className="op"+(m[k.campo]===o.id?" sel":"");
b.textContent=`${n+1}. ${o.desc}`;b.onclick=()=>elegir(j,o.id);d.appendChild(b)});c.appendChild(d)})}
async function guardar(m){M[F[i].id]=m;await api("api/etiqueta",Object.assign({id:F[i].id},m))}
async function elegir(j,v){const m=Object.assign({},M[F[i].id]||{});m[E.campos[j].campo]=v;await guardar(m);
if(j<E.campos.length-1){activo=j+1}else{activo=0;if(E.campos.every(k=>m[k.campo]))mover(1)}pintar()}
function mover(d){i=Math.max(0,Math.min(F.length-1,i+d));activo=0;pintar()}
function saltar(){const k=F.findIndex(f=>!E.campos.every(c=>(M[f.id]||{})[c.campo]));i=k<0?0:k;activo=0;pintar()}
document.getElementById("duda").onchange=e=>{const m=Object.assign({},M[F[i].id]||{});m.dudosa=e.target.checked;guardar(m)};
document.getElementById("dp").onchange=e=>{const m=Object.assign({},M[F[i].id]||{});m.dato_personal=e.target.checked;guardar(m)};
document.onkeydown=e=>{if(e.key==="ArrowRight")mover(1);else if(e.key==="ArrowLeft")mover(-1);
else if(e.key==="Tab"){e.preventDefault();activo=(activo+1)%E.campos.length;pintar()}
else if(e.key==="?"||e.key==="p"){const d=document.getElementById(e.key==="p"?"dp":"duda");d.checked=!d.checked;d.onchange({target:d})}
else if(/^[1-9]$/.test(e.key)){const o=E.campos[activo].opciones[+e.key-1];if(o)elegir(activo,o.id)}};
ini().catch(e=>document.body.textContent="Error: "+e.message);
</script></body></html>"""


def manejador(dir_):
    d = Path(dir_)
    esquema = json.loads((d / "esquema.json").read_text(encoding="utf-8"))
    tokens = {v: k for k, v in json.loads((d / "etiquetadores.json").read_text(encoding="utf-8")).items()}
    filas = [{"id": r["id"], "entidad": r["entidad"], "texto": r["texto"]} for r in _leer_jsonl(d / "muestra.jsonl")]
    ids = {r["id"] for r in filas}
    campos = {c["campo"]: {o["id"] for o in c["opciones"]} for c in esquema["campos"]}
    (d / "etiquetas").mkdir(exist_ok=True)

    class H(BaseHTTPRequestHandler):
        def _quien(self):
            return tokens.get((parse_qs(urlparse(self.path).query).get("t") or [""])[0])

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            ruta, q = urlparse(self.path).path, self._quien()
            if q is None:
                return self._json({"error": "token desconocido"}, 403)
            if ruta in ("/", "/index.html"):
                b = PAGINA.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                return self.wfile.write(b)
            if ruta == "/guia":
                b = GUIA.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                return self.wfile.write(b)
            if ruta == "/api/datos":
                mias = {r["id"]: {k: v for k, v in r.items() if k not in ("id", "ts")}
                        for r in _leer_jsonl(d / "etiquetas" / (q + ".jsonl"))}
                return self._json({"esquema": esquema, "filas": filas, "mias": mias, "quien": html.escape(q)})
            self._json({"error": "no existe"}, 404)

        def do_POST(self):
            q = self._quien()
            if q is None or urlparse(self.path).path != "/api/etiqueta":
                return self._json({"error": "no"}, 403)
            try:
                r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError:
                return self._json({"error": "json"}, 400)
            if r.get("id") not in ids or any(k in campos and v not in campos[k] for k, v in r.items()):
                return self._json({"error": "fila o etiqueta fuera del esquema"}, 400)
            limpio = {k: v for k, v in r.items() if k in campos or k in ("id", "dudosa", "dato_personal")}
            import time
            with open(d / "etiquetas" / (q + ".jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(limpio, ts=round(time.time(), 1)), ensure_ascii=False) + "\n")
            self._json({"ok": True})

        def log_message(self, *a):   # sin registro de peticiones (las rutas no llevan texto, pero tampoco hace falta)
            pass

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="orden", required=True)
    s = sub.add_parser("contar")
    s.add_argument("--raiz")
    s = sub.add_parser("muestra")
    s.add_argument("--dir", required=True)
    s.add_argument("--raiz")
    s.add_argument("--n", type=int, default=450)
    s.add_argument("--semilla", type=int, default=7)
    s.add_argument("--min-chars", type=int, default=20)
    s = sub.add_parser("servir")
    s.add_argument("--dir", required=True)
    s.add_argument("--puerto", type=int, default=8080)
    s = sub.add_parser("kappa")
    s.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    if a.orden == "contar":
        print(json.dumps(contar(cargar_fuentes(a.raiz)), ensure_ascii=False, indent=1))
    elif a.orden == "muestra":
        d = Path(a.dir)
        d.mkdir(parents=True, exist_ok=True)
        filas = muestra(cargar_fuentes(a.raiz), a.n, a.semilla, a.min_chars)
        _escribir_jsonl(d / "muestra.jsonl", filas)
        if not (d / "esquema.json").exists():
            (d / "esquema.json").write_text(json.dumps(ESQUEMA, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        cuenta = collections.Counter((r["entidad"], r["etiqueta_original"]) for r in filas)
        print(json.dumps({"filas": len(filas), "estratos": len(cuenta),
                          "por_entidad": dict(collections.Counter(r["entidad"] for r in filas))}, ensure_ascii=False))
    elif a.orden == "servir":
        srv = ThreadingHTTPServer(("0.0.0.0", a.puerto), manejador(a.dir))
        print("sirviendo en :%d" % a.puerto, flush=True)
        srv.serve_forever()
    else:
        print(json.dumps(kappa(a.dir), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
