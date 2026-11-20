"""W1d (ud B2): termina la medida fs8 de la fase B. Solo re-mide en Python las filas que B dejó sin respuesta válida y las junta con las ya válidas.

    python3 -m model.alt.w1_juntar extraer --res-tgz resultados.tgz --fasea-tar faseA.tar --fasea-json faseA.json --salida DIR
    python3 -m model.alt.w1_juntar faltan --res DIR/res --ent DIR/ent --salida DIR_FALTAN
    python3 -m model.alt.w1_juntar juntar --res DIR/res --nuevo DIR_NUEVO --salida DIR_PY

extraer   (tras comprobar el sha256 de los dos ficheros) saca de resultados.tgz solo res/ y de faseA.tar solo ent/ (sus ficheros con el sha256 de faseA.json); nada más de A.
faltan    por tarea, las filas de res/test-demos sin fila válida en res/py (valida y probs; un aviso en `error` con probs no cuenta como fallo) con SUS demostraciones (res/demos),
          en <salida>/test-demos y <salida>/demos, y plan.json con el orden (finanzas primero, luego las demás). Comprueba que las demostraciones del tgz son las que se regeneran del registro
          del servidor (res/demos-log.jsonl) y las filas de test de A, y que Go tiene una fila 200 con probs por cada consulta.
juntar    una fila por id: la válida de B si la hay; si no, la nueva; si tampoco, la vieja no válida. Rechaza una nueva que choque con una vieja válida (no se re-mide lo ya válido).
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

from model.alt.w1_demos import escribir, leer, registro, sha_estado

PRIMERA = "finanzas-documentos"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def valida(r):
    return bool(r.get("valida")) and bool(r.get("probs"))


def leer_si(p):
    return leer(p) if Path(p).exists() else []


def _extraer_tar(tar, salida, ok, modo):
    salida = Path(salida)
    with tarfile.open(tar, modo) as t:
        ms = [m for m in t.getmembers() if ok(m.name.removeprefix("./"))]
        if any(m.issym() or m.islnk() or m.isdev() for m in ms):
            raise SystemExit("%s: enlaces o dispositivos en el tar" % tar)
        for m in ms:
            m.name = m.name.removeprefix("./")
        t.extractall(salida, members=ms, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    return len(ms)


def cmd_extraer(a):
    salida = Path(a.salida)
    nr = _extraer_tar(a.res_tgz, salida, lambda n: n == "res" or n.startswith("res/"), "r:gz")
    j = json.load(open(a.fasea_json, encoding="utf-8"))
    if j.get("fase") != "A" or j["tar_sha256"] != sha256(a.fasea_tar):
        raise SystemExit("faseA.json no es el de este faseA.tar")
    fa = salida / "faseA"
    _extraer_tar(a.fasea_tar, fa, lambda n: n.startswith("ent/"), "r:")
    for n, s in j["ficheros"].items():
        if n.startswith("ent/") and (not (fa / n).is_file() or sha256(fa / n) != s):
            raise SystemExit("fichero %s ausente o con otro sha256 que el de faseA.json" % n)
    hay = {str(p.relative_to(fa)) for p in (fa / "ent").rglob("*") if p.is_file()}
    if hay != {n for n in j["ficheros"] if n.startswith("ent/")}:
        raise SystemExit("ent/ de faseA.tar no es el de faseA.json")
    shutil.move(str(fa / "ent"), str(salida / "ent"))
    print("res: %d ficheros; ent: %d ficheros (sha256 de faseA.json)" % (nr, len(hay)))


def orden(tareas):
    return sorted(tareas, key=lambda t: (t != PRIMERA, t))


def cmd_faltan(a):
    res, ent, sal = Path(a.res), Path(a.ent), Path(a.salida)
    cons = leer(ent / "consultas.jsonl")
    go = {}
    for r in leer(res / "e2e-w1-filas.jsonl"):
        go[(r["tarea"], r["id"])] = r
    malos = [(r["tarea"], r["id"]) for r in cons if go.get((r["tarea"], r["id"]), {}).get("http") != 200 or not go[(r["tarea"], r["id"])].get("probs")]
    if malos or len(go) != len(cons):
        raise SystemExit("Go: %d consultas, %d filas, %d sin 200 con probs: no se re-mide Go y B2 no lo arregla" % (len(cons), len(go), len(malos)))
    tareas = sorted({r["tarea"] for r in cons})
    plan, tot = {}, 0
    for t in tareas:
        test = leer(res / "test-demos" / (t + ".jsonl"))
        demos = {d["id"]: d["ejemplos"] for d in leer(res / "demos" / (t + ".jsonl"))}
        if [r["id"] for r in test] != [r["id"] for r in cons if r["tarea"] == t] or set(demos) != {r["id"] for r in test}:
            raise SystemExit("%s: las filas de test-demos de B no son las consultas (o faltan demostraciones)" % t)
        viejas = {r["id"]: r for r in leer_si(res / "py" / (t + ".jsonl"))}
        pend = [r for r in test if not valida(viejas.get(r["id"], {}))]
        escribir(sal / "test-demos" / (t + ".jsonl"), pend)
        escribir(sal / "demos" / (t + ".jsonl"), [{"id": r["id"], "ejemplos": demos[r["id"]]} for r in pend])
        plan[t] = len(pend)
        tot += len(pend)
    # las demostraciones del tgz son las que el servidor registró (se regeneran del registro con el mismo código y las filas de test de A)
    reg = registro(res / "demos-log.jsonl")
    for t in tareas:
        dd = {x["id"]: x["ejemplos"] for x in leer(res / "demos" / (t + ".jsonl"))}
        for r in leer(res / "test-demos" / (t + ".jsonl")):
            if reg.get((t, sha_estado(r["texto"]))) != dd[r["id"]]:
                raise SystemExit("%s %s: las demostraciones de B no son las del registro del servidor" % (t, r["id"]))
    (sal / "plan.json").write_text(json.dumps({"orden": [t for t in orden(tareas) if plan[t]], "pendientes": plan, "total": tot}, indent=1) + "\n", encoding="utf-8")
    print("faltan %d filas: %s" % (tot, ", ".join("%s %d" % (t, plan[t]) for t in orden(tareas) if plan[t]) or "ninguna"))


def cmd_juntar(a):
    res, sal = Path(a.res), Path(a.salida)
    cuenta = {}
    for p in sorted((res / "test-demos").glob("*.jsonl")):
        t = p.stem
        ids = [r["id"] for r in leer(p)]
        viejas = {r["id"]: r for r in leer_si(res / "py" / (t + ".jsonl"))}
        nuevas = {r["id"]: r for r in leer_si(Path(a.nuevo) / (t + ".jsonl"))}
        choque = [i for i in nuevas if valida(viejas.get(i, {}))]
        if choque or set(nuevas) - set(ids):
            raise SystemExit("%s: filas nuevas que ya eran válidas o que no son de la medida: %s" % (t, (choque or sorted(set(nuevas) - set(ids)))[:3]))
        out, c = [], {"conservadas": 0, "sustituidas": 0, "sin_remedir": 0}
        for i in ids:
            v, n = viejas.get(i), nuevas.get(i)
            if v is not None and valida(v):
                out.append(v)
                c["conservadas"] += 1
            elif n is not None:
                out.append(n)
                c["sustituidas"] += 1
            elif v is not None:
                out.append(v)
                c["sin_remedir"] += 1
            else:
                c["sin_remedir"] += 1
        escribir(sal / (t + ".jsonl"), out)
        c.update(filas=len(out), validas=sum(valida(r) for r in out), de=len(ids))
        cuenta[t] = c
    (sal / "junta.json").write_text(json.dumps(cuenta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("junta: " + "; ".join("%s %d/%d válidas (%d conservadas, %d nuevas)" % (t, c["validas"], c["de"], c["conservadas"], c["sustituidas"]) for t, c in sorted(cuenta.items())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extraer")
    for k in ("res-tgz", "fasea-tar", "fasea-json", "salida"):
        e.add_argument("--" + k, required=True)
    f = sp.add_parser("faltan")
    for k in ("res", "ent", "salida"):
        f.add_argument("--" + k, required=True)
    j = sp.add_parser("juntar")
    for k in ("res", "nuevo", "salida"):
        j.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    {"extraer": cmd_extraer, "faltan": cmd_faltan, "juntar": cmd_juntar}[a.cmd](a)


if __name__ == "__main__":
    main()
