"""R10, piloto: evaluación EN INT8 de un checkpoint de L2 en las 6 tareas DEV de la batería v2, y veredicto del criterio congelado.

    python -m model.alt.eval_r10_piloto medir --ckpt C.pt --etiqueta base --dev DIR_DEV --salida R.json --crit crit-r10-piloto.json --sha-crit SHA
        [--vocab model/bench/vocab-podado-e5-small.json | --sin-poda] [--hilos 2] [--trabajo DIR]
    python -m model.alt.eval_r10_piloto comparar --base R_base.json --destilado R_dest.json --crit crit-r10-piloto.json --sha-crit SHA --salida DIR

Camino de producción (`model/l2/exportar.py`, perfil `movil-int8`): ONNX del codificador de consulta + ranuras con int8 dinámico solo en el
codificador (las ranuras en fp32), ONNX de la torre con int8 dinámico, vocabulario podado (aquí ampliado con los ids de las tareas DEV, como
`--vocab-ampliar`: lo que se mide es la cuantización y no la poda) y vectores de torre con una escala int8 por vector (como ESQUEMAS).
Las cabezas (T de choice, β, CORAL…) son las del checkpoint; la puntuación sale de `ModeloL2.forward` con `ranuras` y `torre` sustituidas
por las salidas del ONNX int8. Dos configuraciones por tarea: zero-shot (descripciones) y k = 8 (prototipos con los 8 ejemplos de oro de cada clase
= split train del fichero). Métrica: macro-F1 por tarea (`model.evaluacion.chispa_eval`). Solo lee las tareas DEV de la batería v2: aborta si el
directorio trae algo que no esté marcado `dev` en el manifiesto, y jamás abre una sellada. El criterio (crit-r10-piloto.json) se comprueba por sha256
antes de medir.
"""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.evaluacion import chispa_eval as CE
from model.l2 import datos as D
from model.l2 import exportar as E
from model.l2.modelo import Codificador, ModeloL2, SoporteEval, clave_cabeza, normalizar_mayusculas

AQUI = Path(__file__).parent
MANIFIESTO_V2 = AQUI / "bateria-v2-manifiesto.json"
CRITERIO = AQUI / "crit-r10-piloto.json"
VOCAB = Path(__file__).resolve().parents[1] / "bench" / "vocab-podado-e5-small.json"
K = 8


def sha256_fichero(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def comprobar_criterio(ruta, sha):
    """Falla si el fichero del criterio no tiene el sha256 con que se congeló. Devuelve el criterio."""
    real = sha256_fichero(ruta)
    if not sha or real != sha:
        raise SystemExit("el criterio %s tiene sha256 %s y se esperaba %s: no se mide" % (ruta, real, sha))
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------- datos
def tareas_dev_v2(manifiesto=MANIFIESTO_V2):
    """{nombre: {tipo, idioma}} de las tareas con partición `dev` de la batería v2."""
    m = json.loads(Path(manifiesto).read_text(encoding="utf-8"))
    return {t["nombre"]: {"tipo": t.get("tipo"), "idioma": t.get("idioma")}
            for t in m["tareas"] if t["particion"] == "dev"}


def leer_dev(directorio, manifiesto=MANIFIESTO_V2):
    """{tarea: (soporte, test)} de los .jsonl de `directorio`. Solo tareas DEV del manifiesto; exige las 6."""
    dev = tareas_dev_v2(manifiesto)
    d = Path(directorio)
    ficheros = sorted(d.glob("*.jsonl"))
    if "sellada" in str(d).lower() or any("sellada" in f.name.lower() for f in ficheros):
        raise SystemExit("la carpeta %s parece de selladas: no se abre" % d)
    extra = sorted(f.stem for f in ficheros if f.stem not in dev)
    if extra:
        raise SystemExit("hay tareas que no son DEV de la batería v2: %s" % extra)
    falta = sorted(set(dev) - {f.stem for f in ficheros})
    if falta:
        raise SystemExit("faltan tareas DEV: %s" % falta)
    out = {}
    for f in ficheros:
        filas = [r for r in D.iterar([str(f)]) if r.get("split") in ("train", "test")]
        sop, test = [r for r in filas if r["split"] == "train"], [r for r in filas if r["split"] == "test"]
        if not sop or not test:
            raise SystemExit("%s: sin soporte o sin test" % f.name)
        if len({r["tarea"] for r in filas}) != 1:
            raise SystemExit("%s: más de una `tarea`" % f.name)
        out[f.stem] = (sop, test)
    return out


# ---------------------------------------------------------------------------------------------------------------- int8
class RanurasPaso(nn.Module):
    """Sustituye a `Ranuras`: la «entrada» ya es S (B,K,d) salido del ONNX int8."""

    def __init__(self, k):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(k, 384), requires_grad=False)

    def forward(self, H, mascara):
        return H


class EncOnnx:
    """Hace de `Codificador` para `ModeloL2.forward`: `tokens` devuelve S del ONNX de consulta (un texto por llamada, como el runtime) y
    `pasaje` los vectores de la torre del ONNX, con una escala int8 por vector (ESQUEMAS)."""

    n_desc = 0

    def __init__(self, enc, sess_q, sess_t):
        self.tok, self.max_tok, self.sq, self.st = enc.tok, enc.max_tok, sess_q, sess_t
        self.cs, self.ct = {}, {}
        self.llamadas = 0

    def _ids(self, textos):
        out = []
        for t in textos:
            ids = self.tok([t], truncation=True, max_length=self.max_tok)["input_ids"][0]
            out.append([int(x) for x in ids])
        return out

    def tokens(self, textos):
        textos = [normalizar_mayusculas(t) for t in textos]
        falta = [t for t in dict.fromkeys(textos) if t not in self.cs]
        for t, ids in zip(falta, self._ids(["query: " + t for t in falta])):
            self.cs[t] = torch.from_numpy(E.correr(self.sq, np.array([ids]), np.ones((1, len(ids))))[0])
            self.llamadas += 1
        S = torch.stack([self.cs[t] for t in textos])
        return S, torch.ones(len(textos), 1, dtype=torch.bool)

    def pasaje(self, textos):
        falta = [t for t in dict.fromkeys(textos) if t not in self.ct]
        if falta:
            h = E.correr_torre(self.st, self._ids(falta))
            for t, v in zip(falta, h):
                q, esc = E.cuantizar_vec(v)
                self.ct[t] = torch.from_numpy(q.astype(np.float32) * np.float32(esc))
        return torch.stack([self.ct[t] for t in textos])


def modelo_int8(modelo):
    """Copia de `modelo` cuyas ranuras y torre son identidades: las cabezas, T, β y CORAL siguen siendo las del checkpoint."""
    m = copy.deepcopy(modelo)
    m.ranuras, m.torre = RanurasPaso(modelo.ranuras.q.shape[0]), nn.Identity()
    return m.eval()


def cargar(ckpt):
    """(modelo fp32, Codificador CPU con las capas del checkpoint, encoder HF eager con los mismos pesos, checkpoint)."""
    import os
    from transformers import AutoModel
    from model.l2.evaluar import cargar_modelo
    modelo, ck = cargar_modelo(ckpt, "cpu")
    enc = Codificador(device="cpu", max_tok=int(ck.get("max_tok") or 128))
    enc.cargar_enc(ck.get("enc"))
    ruta = os.environ.get("CHISPA_E5") or "intfloat/multilingual-e5-small"
    local = os.environ.get("HF_HUB_OFFLINE") == "1" or Path(ruta).exists() or Path("~/hf/hub").expanduser().exists()
    encoder = AutoModel.from_pretrained(ruta, attn_implementation="eager", local_files_only=local).eval()
    encoder.load_state_dict(enc.enc.state_dict())
    return modelo, enc, encoder, ck


def exportar_int8(modelo, enc, encoder, vocab, trabajo):
    """(ruta ONNX de consulta, ruta ONNX de torre, info) con el perfil movil-int8 de `exportar.py`."""
    trabajo = Path(trabajo)
    trabajo.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    q8, t8 = trabajo / "consulta-int8.onnx", trabajo / "torre-int8.onnx"
    qenc, qremap = E.podar(encoder, vocab) if vocab else (encoder, None)
    n = len(qremap) if vocab else encoder.get_input_embeddings().weight.shape[0]
    q32 = trabajo / "consulta-fp32.onnx"
    E.exportar_onnx(E.Envoltura(qenc, modelo.ranuras, qremap), q32, n)
    E.cuantizar(q32, q8, "codificador", True)
    E.meta_consulta(q8, enc.max_tok)
    q32.unlink()
    inf_t = E.exportar_torre(encoder, modelo.torre, t8, trabajo / "t", vocab, cuant="dinamico", per_channel=True, max_len=enc.max_tok,
                             texto_torre=modelo.texto_torre)
    inf = {"consulta_mb": round(q8.stat().st_size / 1e6, 1), "torre_mb": round(t8.stat().st_size / 1e6, 1), "vocab_n": inf_t["vocab_n"],
           "segundos_exportar": round(time.time() - t0, 1)}
    return q8, t8, inf


def evaluar(ckpt, dev, vocab=None, trabajo=None, hilos=2, rutas_vocab=(), log=print):
    """{tarea: {zeroshot: {...}, k8: {...}}, info} de un checkpoint, en int8. `dev` = salida de `leer_dev`."""
    modelo, enc, encoder, ck = cargar(ckpt)
    inf = {"ckpt": str(ckpt), "paso": ck.get("paso"), "max_tok": enc.max_tok, "cabezas_ckpt": sorted(modelo.coral.keys()),
           "capas_e5_en_ckpt": len(ck.get("enc") or {})}
    filas_t = [r for _, test in dev.values() for r in test]
    inf["cabezas_sin_entrenar"] = sorted({clave_cabeza(r) for r in filas_t if r["tipo"] == "score"} - set(modelo.coral.keys()))
    if vocab:
        vocab, ampl = E.ampliar_vocab(enc, vocab, list(rutas_vocab)) if rutas_vocab else (vocab, None)
        inf["vocab_ampliado"] = ampl
    with tempfile.TemporaryDirectory() as t:
        q8, t8, inf_e = exportar_int8(modelo, enc, encoder, vocab, trabajo or t)
        inf.update(inf_e)
        eo = EncOnnx(enc, E.sesion(q8, hilos), E.sesion(t8, hilos))
        m8 = modelo_int8(modelo)
        for r in filas_t:
            m8.asegurar_cabeza(clave_cabeza(r))
        from model.l2.evaluar import predecir
        sop = SoporteEval({rs[0]["tarea"]: rs for rs, _ in dev.values()}, K, 0, 0.0)
        res = {}
        t0 = time.time()
        for nombre, preds in (("zeroshot", predecir(m8, eo, filas_t)), ("k8", predecir(m8, eo, filas_t, soporte=sop))):
            ev = CE.evaluar(filas_t, preds)["por_tarea"]
            for tarea, m in ev.items():
                res.setdefault(tarea, {})[nombre] = {"macro_f1": m["macro_f1"], "exactitud": m["exactitud"], "n": m["n"]}
            log("%s: macro-F1 medio %.4f (%.0f s)" % (nombre, float(np.mean([m["macro_f1"] for m in ev.values()])), time.time() - t0))
        inf["segundos_evaluar"] = round(time.time() - t0, 1)
        inf["textos_consulta"], inf["textos_torre"] = len(eo.cs), len(eo.ct)
    return {"tareas": res, "info": inf}


def evaluar_e5(dev, vocab=None, trabajo=None, hilos=2, rutas_vocab=(), max_tok=256, log=print):
    """Base «e5 puro» (sin L2), en int8 con el mismo camino de export: zero-shot = coseno entre el vector e5 de «query: texto» y el de «passage:
    pregunta · opción · descripción» de cada opción; k = 8 = coseno con el prototipo de clase (media normalizada de los vectores e5 de los 8 ejemplos
    de oro). e5 va en ONNX int8 dinámico (media normalizada, sin ranuras ni torre)."""
    import os
    from transformers import AutoModel
    from model.l2.modelo import texto_opcion
    enc = Codificador(device="cpu", max_tok=max_tok)
    ruta = os.environ.get("CHISPA_E5") or "intfloat/multilingual-e5-small"
    local = os.environ.get("HF_HUB_OFFLINE") == "1" or Path(ruta).exists() or Path("~/hf/hub").expanduser().exists()
    encoder = AutoModel.from_pretrained(ruta, attn_implementation="eager", local_files_only=local).eval()
    inf = {"modelo": "e5 puro", "max_tok": max_tok}
    if vocab and rutas_vocab:
        vocab, inf["vocab_ampliado"] = E.ampliar_vocab(enc, vocab, list(rutas_vocab))
    with tempfile.TemporaryDirectory() as t:
        d = Path(trabajo or t)
        d.mkdir(parents=True, exist_ok=True)
        qenc, remap = E.podar(encoder, vocab) if vocab else (encoder, None)
        n = len(remap) if vocab else encoder.get_input_embeddings().weight.shape[0]
        f32, i8 = d / "e5-fp32.onnx", d / "e5-int8.onnx"
        E.exportar_onnx(E.EnvolturaTorre(qenc, nn.Identity(), remap), f32, n, "h")
        E.cuantizar(f32, i8, "todo", True)
        f32.unlink()
        inf["onnx_mb"] = round(i8.stat().st_size / 1e6, 1)
        sess, cache = E.sesion(i8, hilos), {}

        def vec(textos):
            falta = [x for x in dict.fromkeys(textos) if x not in cache]
            if falta:
                ids = E.ids_torre(enc, falta, max_tok)
                for x, v in zip(falta, E.correr_torre(sess, ids)):
                    cache[x] = torch.from_numpy(v.astype(np.float32))
            return torch.stack([cache[x] for x in textos])
        q = lambda r: "query: " + normalizar_mayusculas(r["texto"])
        res, filas_t, preds = {}, [r for _, test in dev.values() for r in test], {"zeroshot": [], "k8": []}
        protos = {}
        for sop, _ in dev.values():
            v = torch.nn.functional.normalize(vec([q(r) for r in sop]), dim=-1)
            cl = {}
            for r, x in zip(sop, v):
                cl.setdefault(r["etiqueta"], []).append(x)
            protos[sop[0]["tarea"]] = {c: torch.nn.functional.normalize(torch.stack(xs).mean(0), dim=-1) for c, xs in cl.items()}
        for r in filas_t:
            qv = torch.nn.functional.normalize(vec([q(r)])[0], dim=-1)
            dv = torch.nn.functional.normalize(vec([texto_opcion(r, o, "v1") for o in r["opciones"]]), dim=-1)
            zs = dv @ qv
            pr = protos[r["tarea"]]
            k8 = torch.stack([(pr[o] @ qv) if o in pr else torch.tensor(-1.0) for o in r["opciones"]])
            for nombre, sc in (("zeroshot", zs), ("k8", k8)):
                p = torch.softmax(sc / 0.02, -1).tolist()
                preds[nombre].append({"id": r["id"], "probs": dict(zip(r["opciones"], p))})
        for nombre in ("zeroshot", "k8"):
            ev = CE.evaluar(filas_t, preds[nombre])["por_tarea"]
            for tarea, m in ev.items():
                res.setdefault(tarea, {})[nombre] = {"macro_f1": m["macro_f1"], "exactitud": m["exactitud"], "n": m["n"]}
            log("%s: macro-F1 medio %.4f" % (nombre, float(np.mean([m["macro_f1"] for m in ev.values()]))))
    return {"tareas": res, "info": inf}


# ---------------------------------------------------------------------------------------------------------------- criterio
def macro(res, cfg, tareas):
    return 100.0 * float(np.mean([res["tareas"][t][cfg]["macro_f1"] for t in tareas]))


def veredicto(crit, dzs, dk8):
    """SIGUE / PARA / ENTRE_MEDIAS según el criterio congelado (diferencias en puntos de macro-F1)."""
    if dzs < crit["para"]["delta_zeroshot_menor_que"]:
        return "PARA"
    if dzs >= crit["sigue"]["delta_zeroshot_min"] and dk8 >= crit["sigue"]["delta_k8_min"]:
        return "SIGUE"
    return "ENTRE_MEDIAS"


def bootstrap_tareas(diferencias, remuestreos=10000, semilla=0):
    """IC95 percentil de la media de las diferencias por tarea, remuestreando tareas con reposición."""
    d = np.asarray(diferencias, dtype=float)
    rng = np.random.default_rng(semilla)
    medias = d[rng.integers(len(d), size=(remuestreos, len(d)))].mean(axis=1)
    return [float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))]


def comparar(base, dest, crit, sinteticas, sha_crit):
    tareas = sorted(base["tareas"])
    if tareas != sorted(dest["tareas"]):
        raise SystemExit("base y destilado no tienen las mismas tareas")
    sin = [t for t in tareas if t not in sinteticas]
    out = {"criterio_sha256": sha_crit, "tareas": tareas, "sinteticas": sorted(set(tareas) & set(sinteticas)), "por_tarea": {}}
    for t in tareas:
        out["por_tarea"][t] = {c: {"base": 100 * base["tareas"][t][c]["macro_f1"], "destilado": 100 * dest["tareas"][t][c]["macro_f1"],
                                   "delta": 100 * (dest["tareas"][t][c]["macro_f1"] - base["tareas"][t][c]["macro_f1"]),
                                   "exactitud_base": 100 * base["tareas"][t][c]["exactitud"], "exactitud_destilado": 100 * dest["tareas"][t][c]["exactitud"],
                                   "n": base["tareas"][t][c]["n"]} for c in ("zeroshot", "k8")}
    for nombre, ts in (("seis", tareas), ("sin_sinteticas", sin)):
        m = {}
        for c in ("zeroshot", "k8"):
            b, d = macro(base, c, ts), macro(dest, c, ts)
            dif = [100 * (dest["tareas"][t][c]["macro_f1"] - base["tareas"][t][c]["macro_f1"]) for t in ts]
            m[c] = {"base": b, "destilado": d, "delta": d - b, "ic95_delta_tareas": bootstrap_tareas(dif), "tareas": len(ts)}
        m["veredicto"] = veredicto(crit, m["zeroshot"]["delta"], m["k8"]["delta"])
        out[nombre] = m
    out["veredicto"] = out["seis"]["veredicto"]
    out["veredicto_cambia_sin_sinteticas"] = out["seis"]["veredicto"] != out["sin_sinteticas"]["veredicto"]
    out["info"] = {"base": base["info"], "destilado": dest["info"]}
    out["vs_l2_actual"] = {"medido_aqui": False, "fuente": "T3", "zeroshot_delta": None, "k8_delta": None}   # a rellenar con las cifras de T3
    return out


def markdown(r):
    f = lambda x: "%.1f" % x
    L = ["# R10 piloto: destilación 4B -> L2, int8, tareas DEV de la batería v2", "",
         "Criterio sha256 `%s`. Métrica: macro-F1 por tarea ×100, media simple. Veredicto (6 tareas): **%s**%s." % (
             r["criterio_sha256"], r["veredicto"], " (cambia sin las sintéticas: %s)" % r["sin_sinteticas"]["veredicto"] if r["veredicto_cambia_sin_sinteticas"] else ""),
         "", "| grupo | config | base | destilado | Δ | IC95 Δ (tareas) |", "|---|---|---|---|---|---|"]
    for g, et in (("seis", "6 tareas"), ("sin_sinteticas", "sin sintéticas (%d)" % r["sin_sinteticas"]["zeroshot"]["tareas"])):
        for c in ("zeroshot", "k8"):
            m = r[g][c]
            L.append("| %s | %s | %s | %s | %+.1f | [%+.1f, %+.1f] |" % (et, c, f(m["base"]), f(m["destilado"]), m["delta"], *m["ic95_delta_tareas"]))
    L += ["", "| tarea | config | base | destilado | Δ |", "|---|---|---|---|---|"]
    for t in r["tareas"]:
        for c in ("zeroshot", "k8"):
            m = r["por_tarea"][t][c]
            L.append("| %s%s | %s | %s | %s | %+.1f |" % (t, " (sintética)" if t in r["sinteticas"] else "", c, f(m["base"]), f(m["destilado"]), m["delta"]))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("medir")
    m.add_argument("--ckpt", help="checkpoint de L2 (no con --e5-puro)")
    m.add_argument("--e5-puro", action="store_true", help="la base: e5 sin L2 (coseno de pasaje; prototipos con k = 8)")
    m.add_argument("--max-tok", type=int, default=256, help="solo con --e5-puro")
    m.add_argument("--etiqueta", default="modelo")
    m.add_argument("--dev", required=True)
    m.add_argument("--salida", required=True)
    m.add_argument("--crit", default=str(CRITERIO))
    m.add_argument("--sha-crit", required=True)
    m.add_argument("--vocab", default=str(VOCAB))
    m.add_argument("--sin-poda", action="store_true")
    m.add_argument("--hilos", type=int, default=2)
    m.add_argument("--trabajo")
    c = sub.add_parser("comparar")
    c.add_argument("--base", required=True)
    c.add_argument("--destilado", required=True)
    c.add_argument("--crit", default=str(CRITERIO))
    c.add_argument("--sha-crit", required=True)
    c.add_argument("--salida", required=True)
    a = ap.parse_args(argv)
    crit = comprobar_criterio(a.crit, a.sha_crit)
    if a.cmd == "medir":
        dev = leer_dev(a.dev)
        vocab = None if a.sin_poda else json.loads(Path(a.vocab).read_text(encoding="utf-8"))
        log = lambda s: print("[%s] %s" % (a.etiqueta, s), flush=True)
        rutas = sorted(str(p) for p in Path(a.dev).glob("*.jsonl"))
        if a.e5_puro:
            r = evaluar_e5(dev, vocab, a.trabajo, a.hilos, rutas, a.max_tok, log=log)
        elif a.ckpt:
            r = evaluar(a.ckpt, dev, vocab, a.trabajo, a.hilos, rutas, log=log)
        else:
            raise SystemExit("falta --ckpt o --e5-puro")
        r["etiqueta"], r["criterio_sha256"] = a.etiqueta, a.sha_crit
        Path(a.salida).write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return 0
    base, dest = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.base, a.destilado))
    for x in (base, dest):
        if x.get("criterio_sha256") != a.sha_crit:
            raise SystemExit("una de las mediciones se hizo con otro criterio")
    r = comparar(base, dest, crit, set(crit["sinteticas"]), a.sha_crit)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "r10-piloto.json").write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (sal / "r10-piloto.md").write_text(markdown(r), encoding="utf-8")
    print(markdown(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
