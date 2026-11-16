"""R10, piloto de destilación 4B -> biencoder de Chispa (L2). Ajuste fino de un checkpoint de L2 con las etiquetas blandas de Qwen3-4B.

    python -m model.alt.destilar_biencoder entrenar --datos RAIZ --manifiesto M --partida CKPT.pt --salida DIR [--dev-dir DIR_DEV_V2] ...

Qué es: la MISMA puntuación de producción (MaxSim texto <-> torre de cada opción con e5-small y ranuras de `model/l2/modelo.py`), sin
cabeza nueva, entrenada con
    L = 0,6·KL(maestro, T=2)·T² + 0,2·CE(oro) + 0,2·contrastivo en lote + peso_ref·(1 − cos(e5 del modelo, e5 congelado)).
- KL: el maestro son las blandas de Qwen3-4B-Instruct-2507 (Apache-2.0) de `model/alt/destilar.py`. Si oro y maestro discrepan manda el oro
  (`--desacuerdo intercambiar`, por defecto: se intercambian en el vector del maestro la masa del oro y la de su argmax, así que el oro
  gana y el resto de la forma —las confusiones que el 4B ve plausibles— se conserva; `ignorar` quita la fila del KL; `oro` la sustituye
  por el one-hot del oro).
- Contrastivo (solo `choice`): positivo = la torre de la opción de oro; negativos = las demás opciones de la fila (las 2..k del maestro) y las
  `--minadas` torres más parecidas, de las otras tareas del lote.
- 20 % de los pasos (`--frac-proto`) son episodios: una tarea de clases fijas, soporte de 8 ejemplos por clase y logit = β·simproto(S, P_c)
  (la media de las ranuras de los 8 ejemplos) EN LUGAR de la descripción de la opción.
- Protección del kNN: la penalidad por alejar de e5 original el vector de pasaje de las opciones y el vector medio del mensaje (coseno
  frente a un e5 congelado). Solo hay penalidad si `--descongelar` > 0 (si e5 no se mueve no hay nada que proteger).
- `score` ordinal: en lugar de KL y CE se usa la distancia entre las CDF (EMD normalizada por K−1) al maestro y al oro (`--score-perdida
  emd`), o CE con el oro suavizado por distancia (`suavizado`).
Solo entran filas con `blanda` (`--con-sin-blanda` las admite) y fuera de las exclusiones de `EXCLUSIONES`/`huellas_exclusion`: batería v2
(todas, dev y selladas, por repo y familia), batería v1, tareas nuevas y lo que Juan no ha decidido. Casi-duplicados: se quitan del train las
filas cuyo texto tiene coseno e5 >= 0,95 con algún texto de las tareas DEV de v2 (nunca se abre una sellada). Las salidas de JevBench no
entran: aquí solo hay las blandas del 4B y las etiquetas de las fuentes del manifiesto.

    python -m model.alt.destilar_biencoder sintetico --salida DIR     # datos, checkpoint y e5 diminutos para la prueba en seco
"""
import argparse
import json
import math
import random
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.l2 import datos as D
from model.l2.modelo import (Codificador, ModeloL2, blanda_vector, clave_cabeza, niveles_score, normalizar_mayusculas, prototipos, simproto,
                             textos_torre_fila)

from .manifiesto import cargar, verificar_filas

AQUI = Path(__file__).parent
PESOS = {"kl": 0.6, "ce": 0.2, "contrastivo": 0.2}
T_KL = 2.0
# Tareas de tsjev que son de la misma familia que algo que solo se mide (v1: massive-es/-esc, tuits-es, jailbreak, inyecciones, hatecheck;
# manifiesto-l2x.json `medicion_solo`: clinc150). Se comparan con el campo `tarea` (sin mayúsculas).
EXCLUSIONES = (
    (r"^multilingual/massive$", "massive-es y massive-esc (batería v1: mteb/amazon_massive_intent y _scenario)"),
    (r"^multilingual/tweet_sentiment_multilingual$", "tuits-es (batería v1: mteb/tweet_sentiment_multilingual)"),
    (r"^clinc_oos", "clinc150 (solo medición)"),
    (r"^(prompt-injection|prompt-injections|promptshield)", "inyecciones y jailbreak (batería v1: deepset/prompt-injections, jackhhao)"),
    (r"hatecheck", "hatecheck-es (batería v1: Paul/hatecheck-spanish)"),
)
# Conjuntos del manifiesto que NO entran al piloto aunque su licencia declarada sea comercial: los textos son de plataformas (X/Twitter, Reddit)
# cuyos términos prohíben redistribuirlos. Se excluyen por esta lista (nombre de conjunto del manifiesto), sin tocar el manifiesto; el ud tampoco los baja.
EXCLUIDOS_PLATAFORMA = {
    "ciberacoso": "tuits (FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_1 y _2)",
    "toxicidad_protestas": "tuits (bgonzalezbustamante/toxicity-protests-ES)",
    "toxicidad_constitucion": "tuits de la Convención Constitucional de Chile (model/datos/toxicidad_constitucion.py)",
    "goemociones_es": "comentarios de Reddit (mrm8488/go_emotions-es-mt)",
}
FUENTES_SINTETICAS = {"spam_tanaos": "sintético (tanaos/Artifex, MIT)"}
RE_PLATAFORMA_TSJEV = re.compile(r"twitter|tweet|reddit|tuit", re.I)   # tareas de tsjev cuyo nombre sugiere X/Twitter o Reddit: solo se informan
MEDICION_SOLO = {"massive", "clinc150", "hatecheck", "inyecciones"}   # D.conjunto de las filas por prefijo de id
MANIFIESTOS_MEDICION = (("bateria-v2-manifiesto.json", "v2"), ("manifiesto-tareas-nuevas.json", "nuevas"),
                        ("resultados-20261113-p0-p2/bateria-manifiesto.json", "v1"))


def huellas_exclusion(base=AQUI):
    """{huella en minúsculas: de dónde sale}: el nombre del repo (sin el propietario), la familia y el nombre de cada tarea de la batería v2 (dev
    y selladas), de la v1 y de las tareas nuevas. Una fila se excluye si alguna huella (>= 6 caracteres) aparece en su tarea, fuente o id."""
    out = {}

    def poner(x, de):
        x = (x or "").lower().strip()
        if len(x) >= 6:
            out.setdefault(x, de)

    for nombre, de in MANIFIESTOS_MEDICION:
        ruta = Path(base) / nombre
        if not ruta.is_file():
            continue
        m = json.loads(ruta.read_text(encoding="utf-8"))
        if "tareas" in m:                                   # v2
            for t in m["tareas"]:
                poner(t.get("repo", "").split("/")[-1], de); poner(t.get("familia"), de); poner(t.get("nombre"), de)
        for t in m.get("propuestas", []):   # nuevas (las descartadas no se miden)
            poner(str(t.get("repo", "")).split("/")[-1], de); poner(t.get("nombre"), de)
        for k, t in m.items():                              # v1: {nombre: {repo, ...}}
            if isinstance(t, dict) and "repo" in t:
                poner(t["repo"].split("/")[-1], de); poner(k, de)
    return out


def motivo_exclusion(fila, huellas):
    """Motivo (str) por el que la fila no puede entrenar, o None."""
    if D.conjunto(fila) in MEDICION_SOLO:
        return "solo medición (%s)" % D.conjunto(fila)
    tarea = str(fila.get("tarea", "")).lower()
    for pat, mot in EXCLUSIONES:
        if re.search(pat, tarea):
            return mot
    cadena = " ".join(str(fila.get(k, "")) for k in ("tarea", "fuente", "id")).lower()
    for h, de in huellas.items():
        if h in cadena:
            return "huella %s (batería %s)" % (h, de)
    return None


# ---------------------------------------------------------------------------------------------------------------- datos
def cargar_filas(manifiesto, raiz, con_sin_blanda=False, excluidos_cuenta=None):
    """(train, dev) de los ficheros del manifiesto: train con `blanda` (todas con `con_sin_blanda`), dev todas. Verificadas contra el manifiesto."""
    train, dev, n_train = [], [], 0
    excluidos_cuenta = {} if excluidos_cuenta is None else excluidos_cuenta
    for c in manifiesto.conjuntos:
        if c["nombre"] in EXCLUIDOS_PLATAFORMA:   # lista explícita: se cuentan las filas con blanda que se pierden
            f = Path(raiz) / c["fichero"]
            perdidas = excluidos_cuenta.setdefault(c["nombre"], {"motivo": EXCLUIDOS_PLATAFORMA[c["nombre"]], "filas_train": 0, "con_blanda": 0})
            for r in (D.iterar([str(f)]) if f.is_file() else ()):
                if r["split"] == "train":
                    perdidas["filas_train"] += 1
                    perdidas["con_blanda"] += bool(r.get("blanda"))
            continue
        for r in D.iterar([str(Path(raiz) / c["fichero"])]):
            if r["split"] == "train":
                n_train += 1
                if r.get("blanda") or con_sin_blanda:
                    train.append(r)
            elif r["split"] == "dev":
                dev.append(r)
    verificar_filas(train + dev, manifiesto)
    return train, dev, n_train


def tope_por_tarea(filas, tope, semilla=0):
    if not tope:
        return list(filas)
    rng, por = random.Random(semilla), defaultdict(list)
    for r in filas:
        por[r["tarea"]].append(r)
    out = []
    for t in sorted(por):
        g = por[t]
        rng.shuffle(g)
        out += g[:tope]
    return out


def textos_dev_v2(directorio):
    """Textos (soporte y test) de las tareas DEV de v2 que hay en `directorio`. Nunca abre una carpeta o fichero de selladas."""
    d = Path(directorio)
    if "sellada" in str(d).lower():
        raise ValueError("una carpeta de selladas no se abre aquí: %s" % d)
    out = []
    for f in sorted(d.glob("*.jsonl")):
        if "sellada" in f.name.lower():
            raise ValueError("fichero de sellada: %s" % f)
        for r in D.iterar([str(f)]):
            if r.get("split") in ("train", "test"):
                out.append(r["texto"])
    return out


def casi_duplicados(ref, textos_train, textos_test, umbral=0.95, lote=256, max_tok=128):
    """Posiciones (en `textos_train`) cuyo coseno e5 (media de «query: texto») con algún texto de `textos_test` es >= umbral. `ref` es un
    Codificador sin gradiente."""
    if not textos_test or not textos_train:
        return set()
    viejo, ref.max_tok = ref.max_tok, min(ref.max_tok, max_tok)

    def vec(ts):
        with torch.no_grad():
            return ref._vectores(["query: " + normalizar_mayusculas(t) for t in ts]).float()
    try:
        T = torch.cat([vec(textos_test[i:i + lote]) for i in range(0, len(textos_test), lote)])
        dup = set()
        for i in range(0, len(textos_train), lote):
            sim = vec(textos_train[i:i + lote]) @ T.T.to(ref.device)
            dup.update(i + int(j) for j in (sim.max(dim=1).values >= umbral).nonzero().flatten().tolist())
    finally:
        ref.max_tok = viejo
    return dup


def preparar(a, ref, log=print):
    """(train, dev, informe): filas con blanda, sin exclusiones, con tope por tarea y sin casi-duplicados de DEV de v2."""
    man = cargar(a.manifiesto) if a.manifiesto else cargar()
    cuenta = {}
    train, dev, n_train = cargar_filas(man, a.datos, a.con_sin_blanda, cuenta)
    inf = {"filas_train_totales": n_train, "filas_con_blanda": sum(1 for r in train if r.get("blanda"))}
    inf["conjuntos_excluidos_por_plataforma"] = cuenta
    inf["fuentes_sinteticas"] = {k: v for k, v in FUENTES_SINTETICAS.items() if k in {c["nombre"] for c in man.conjuntos}}
    if a.blandas_resumen and Path(a.blandas_resumen).is_file():   # salida de `destilar --aplicar`: las blandas de los conjuntos no bajados no se escriben
        b = json.loads(Path(a.blandas_resumen).read_text(encoding="utf-8"))
        inf["blandas_en_fichero_origen"], inf["blandas_escritas"] = b.get("en_fichero"), b.get("escritas")
        inf["blandas_perdidas_por_exclusion_de_conjuntos"] = (b.get("en_fichero") or 0) - (b.get("escritas") or 0)
    huellas = huellas_exclusion()
    motivos, tareas = Counter(), defaultdict(Counter)
    keep = []
    for r in train:
        m = motivo_exclusion(r, huellas)
        if m:
            motivos[m] += 1
            tareas[m][r["tarea"]] += 1
        else:
            keep.append(r)
    dev_ok = [r for r in dev if not motivo_exclusion(r, huellas)]
    inf["excluidas"] = {"filas": sum(motivos.values()), "por_motivo": dict(motivos),
                        "tareas": {m: dict(c.most_common(8)) for m, c in tareas.items()}, "dev_quitadas": len(dev) - len(dev_ok)}
    plat = defaultdict(int)   # tsjev: tareas cuyo nombre sugiere Twitter/Reddit (se informan; decide Juan)
    for r in keep:
        if str(r.get("id", "")).startswith("tsjev:") and RE_PLATAFORMA_TSJEV.search(r["tarea"]):
            plat[r["tarea"]] += 1
    inf["tsjev_tareas_twitter_reddit"] = {"filas_con_blanda_tras_exclusion": sum(plat.values()), "por_tarea": dict(sorted(plat.items()))}
    train = tope_por_tarea(keep, a.tope_tarea, a.semilla)
    inf["tras_exclusion"], inf["tras_tope"] = len(keep), len(train)
    if len(train) > a.max_filas:
        train = [r for r in train if D._u(r["id"], a.semilla) < a.max_filas / len(train)]
    inf["tras_max_filas"] = len(train)
    if a.dev_dir and a.umbral_dup <= 1:
        tx = textos_dev_v2(a.dev_dir)
        dup = casi_duplicados(ref, [r["texto"] for r in train], tx, a.umbral_dup)
        inf["casi_duplicados"] = {"umbral": a.umbral_dup, "textos_dev_v2": len(tx), "quitadas": len(dup)}
        train = [r for i, r in enumerate(train) if i not in dup]
    inf["filas_entrenamiento"] = len(train)
    inf["tareas"] = len({r["tarea"] for r in train})
    inf["por_tipo"] = dict(Counter(r["tipo"] for r in train))
    inf["con_blanda_entrenamiento"] = sum(1 for r in train if r.get("blanda"))
    if len(dev_ok) > a.dev_max:
        dev_ok = [r for r in dev_ok if D._u(r["id"], 1) < a.dev_max / len(dev_ok)]
    inf["dev_interno"] = len(dev_ok)
    log("datos: " + json.dumps({k: v for k, v in inf.items() if k not in ("excluidas",)}, ensure_ascii=False))
    return train, dev_ok, inf


# ---------------------------------------------------------------------------------------------------------------- episodios
class EpisodiosProto:
    """Episodios de prototipos (función pura de (semilla, paso)): una tarea `choice` de clases fijas, k ejemplos por clase de soporte y `lote`
    consultas disjuntas, equilibradas por clase. Tareas elegibles: opciones idénticas en >= 95 % de sus filas, >= 2 clases con >= k+2 filas."""

    def __init__(self, filas, lote=32, k=8, vias=16, semilla=0, alfa=0.5):
        por = defaultdict(list)
        for r in filas:
            if r["tipo"] == "choice":
                por[r["tarea"]].append(r)
        self.por, self.k, self.vias, self.lote, self.semilla = {}, k, vias, lote, semilla
        for t, rs in por.items():
            comun = Counter(tuple(r["opciones"]) for r in rs).most_common(1)[0]
            if comun[1] < 0.95 * len(rs):
                continue
            cl = defaultdict(list)
            for r in rs:
                if tuple(r["opciones"]) == comun[0]:
                    cl[r["etiqueta"]].append(r)
            cl = {c: g for c, g in cl.items() if len(g) >= k + 2}
            if len(cl) >= 2:
                self.por[t] = cl
        self.tareas = sorted(self.por)
        n = np.array([sum(len(g) for g in self.por[t].values()) for t in self.tareas], dtype=float)
        self.p = n ** alfa / (n ** alfa).sum() if len(n) else n

    def paso(self, paso):
        rng = np.random.default_rng([self.semilla, paso, 11])
        t = self.tareas[rng.choice(len(self.tareas), p=self.p)]
        cl = sorted(self.por[t])
        if len(cl) > self.vias:
            cl = [cl[i] for i in sorted(rng.choice(len(cl), self.vias, replace=False))]
        soporte, pool = [], {}
        for c in cl:
            g = self.por[t][c]
            ix = set(rng.choice(len(g), self.k, replace=False).tolist())
            soporte += [g[i] for i in sorted(ix)]
            pool[c] = [g[i] for i in range(len(g)) if i not in ix]
        cs = rng.integers(len(cl), size=self.lote)
        filas = [pool[cl[c]][rng.integers(len(pool[cl[c]]))] for c in cs]
        if len(cl) < len(filas[0]["opciones"]):
            filas = [D.restringir(r, set(cl)) for r in filas]
        return {"tarea": t, "filas": filas, "soporte": soporte}


# ---------------------------------------------------------------------------------------------------------------- pérdidas
def tempering(p, T):
    """p^(1/T) renormalizada (p (n,) con p>0): el softmax de los logits/T cuando p es un softmax."""
    return F.softmax(torch.log(p.clamp(min=1e-8)) / T, -1)


def kl_temp(t, p, T=T_KL):
    """T²·KL(t_T || p_T) con t el objetivo del maestro y p la distribución del alumno, ambos (n,)."""
    tt, pt = tempering(t, T), tempering(p, T)
    return (T * T) * (tt * (torch.log(tt.clamp(min=1e-8)) - torch.log(pt.clamp(min=1e-8)))).sum()


def emd(p, q):
    """Distancia entre dos distribuciones sobre niveles ordenados: L1 entre CDF, dividida por K−1 (rango 0..1)."""
    return (torch.cumsum(p, -1) - torch.cumsum(q, -1)).abs()[:-1].sum() / max(len(p) - 1, 1)


def suavizar_oro(k, y, tau=1.0):
    """Distribución sobre k niveles ∝ exp(−|j−y|/tau): CE con ella castiga más lo lejano."""
    w = torch.exp(-(torch.arange(k, dtype=torch.float32) - y).abs() / tau)
    return w / w.sum()


def objetivo_maestro(fila, modo="intercambiar"):
    """(vector (n,) en el orden de `opciones` o None, discrepa). `discrepa`: argmax del maestro != oro. Ver el docstring del módulo."""
    bl = blanda_vector(fila)
    if bl is None:
        return None, False
    y = fila["opciones"].index(fila["etiqueta"])
    j = int(bl.argmax())
    if j == y:
        return bl, False
    if modo == "ignorar":
        return None, True
    if modo == "oro":
        return F.one_hot(torch.tensor(y), len(bl)).float(), True
    bl = bl.clone()
    bl[y], bl[j] = bl[j].item(), bl[y].item()
    return bl, True


def probs_fila(modelo, r, s, sp, T, solo_proto=False):
    """Distribución de la fila en el orden de `opciones`, como `ModeloL2.forward` (s: similitudes de torre de la fila; sp: simproto por opción
    o None). Con `solo_proto` (episodios, solo `choice`): β·simproto en lugar de la similitud con la torre."""
    tipo, ops = r["tipo"], r["opciones"]
    if tipo == "choice":
        if solo_proto:
            if sp is None:
                raise ValueError("episodio de prototipos sin prototipos")
            lg = modelo.beta["choice"] * sp / T
        else:
            lg = (s if sp is None else s + modelo.beta["choice"] * sp) / T
        return F.softmax(lg, -1)
    if solo_proto:
        raise ValueError("los episodios de prototipos son solo de tipo choice")
    if tipo == "noul":
        a, b = modelo.ab["noul"]
        if modelo.cabeza_noul != "una":
            raise ValueError("el piloto exige la cabeza noul `una` (la única que exporta el runtime)")
        pt = torch.sigmoid(a * (s[0] + modelo._ajuste(sp, "noul")) + b)
        return torch.stack([1 - pt, pt])
    niv = niveles_score(ops)
    if sp is not None:
        sp = sp[torch.tensor([ops.index(n) for n in niv], device=sp.device)]
    if modelo.cabeza_score == "coral1":
        pn = modelo.coral["score%d" % len(ops)](s[:1] + modelo._ajuste(sp, "score"))[0]
    else:
        pn = modelo.coral["score%d" % len(ops)]((s if sp is None else s + modelo.beta["score"] * sp).unsqueeze(0))[0]
    return pn[torch.tensor([niv.index(o) for o in ops], device=pn.device)]


def media_normalizada(H, m):
    mf = m.unsqueeze(-1).float()
    return F.normalize((H * mf).sum(1) / mf.sum(1).clamp(min=1), dim=-1)


def calcular(modelo, enc, ref, filas, cfg, protos=None, solo_proto=False):
    """Pérdidas del lote como tensores {total, kl, ce, contrastivo, ref_msg, ref_pas, ...} y las probabilidades por fila."""
    dev = modelo.ranuras.q.device
    for k in {clave_cabeza(r) for r in filas}:
        modelo.asegurar_cabeza(k, dev)
    H, m = enc.tokens([r["texto"] for r in filas])
    H, m = H.to(dev), m.to(dev)
    S = modelo.ranuras(H, m)
    textos, pos, fila_txt = [], {}, []
    for r in filas:
        ix = []
        for t in textos_torre_fila(r, modelo.texto_torre, modelo.cabeza_score, modelo.cabeza_noul):
            if t not in pos:
                pos[t] = len(textos)
                textos.append(t)
            ix.append(pos[t])
        fila_txt.append(ix)
    vp = enc.pasaje(textos).to(dev)
    h = modelo.torre(vp)
    sim = modelo.sim(S, h)
    sp_filas = modelo._simproto(S, filas, protos)
    T = modelo.log_t["choice"].exp().clamp(min=5e-3)
    dueños = defaultdict(set)
    for r, ix in zip(filas, fila_txt):
        for j in ix:
            dueños[j].add(r["tarea"])
    kl, ce, con, discrepan, probs = [], [], [], 0, []
    for i, r in enumerate(filas):
        ix = torch.tensor(fila_txt[i], device=dev)
        p = probs_fila(modelo, r, sim[i, ix], sp_filas[i], T, solo_proto)
        probs.append(p)
        ops, y = r["opciones"], r["opciones"].index(r["etiqueta"])
        t, disc = objetivo_maestro(r, cfg["desacuerdo"])
        discrepan += disc
        if r["tipo"] == "score":
            niv = niveles_score(ops)
            orden = torch.tensor([ops.index(n) for n in niv], device=dev)
            pn = p[orden]
            to = None if t is None else t.to(dev)[orden]
            if cfg["score_perdida"] == "suavizado":
                ce.append(-(suavizar_oro(len(niv), niv.index(r["etiqueta"])).to(dev) * torch.log(pn.clamp(min=1e-7))).sum())
                if to is not None:
                    kl.append(kl_temp(to, pn, 1.0))
            else:
                oro = F.one_hot(torch.tensor(niv.index(r["etiqueta"])), len(niv)).float().to(dev)
                ce.append(emd(pn, oro))
                if to is not None:
                    kl.append(emd(pn, to))
        else:
            ce.append(-torch.log(p[y].clamp(min=1e-7)))
            if t is not None:
                kl.append(kl_temp(t.to(dev), p, cfg["temp_kl"]))
        if r["tipo"] == "choice" and len(ops) >= 2 and cfg["peso_contr"] > 0 and not solo_proto:
            propias = fila_txt[i]
            ajenas = [j for j in range(len(textos)) if r["tarea"] not in dueños[j]]
            minadas = []
            if ajenas and cfg["minadas"] > 0:
                sj = sim[i, torch.tensor(ajenas, device=dev)].detach()
                minadas = [ajenas[int(q)] for q in sj.topk(min(cfg["minadas"], len(ajenas))).indices]
            cand = propias + minadas
            con.append(F.cross_entropy((sim[i, torch.tensor(cand, device=dev)] / T.detach()).unsqueeze(0), torch.tensor([y], device=dev)))
    cero = sim.sum() * 0
    media = lambda xs: torch.stack(xs).mean() if xs else cero
    ps = {"kl": media(kl), "ce": media(ce), "contrastivo": media(con)}
    ps["ref_msg"] = ps["ref_pas"] = cero
    if ref is not None and enc.n_desc > 0 and cfg["peso_ref"] > 0:
        with torch.no_grad():
            Hr, mr = ref.tokens([r["texto"] for r in filas])
            vr = ref.pasaje(textos).to(dev)
        ps["ref_msg"] = (1 - (media_normalizada(H, m) * media_normalizada(Hr.to(dev), mr.to(dev))).sum(-1)).mean()
        ps["ref_pas"] = (1 - (vp * vr).sum(-1)).mean()
    ps["total"] = (cfg["peso_kl"] * ps["kl"] + cfg["peso_ce"] * ps["ce"] + cfg["peso_contr"] * ps["contrastivo"]
                   + cfg["peso_ref"] * (ps["ref_msg"] + ps["ref_pas"]) / 2)
    ps["discrepan"] = torch.tensor(float(discrepan) / len(filas))
    return ps, probs


def cfg_de(a):
    return {"peso_kl": a.peso_kl, "peso_ce": a.peso_ce, "peso_contr": a.peso_contr, "peso_ref": a.peso_ref, "temp_kl": a.temp_kl,
            "minadas": a.minadas, "desacuerdo": a.desacuerdo, "score_perdida": a.score_perdida}


def recortar_lote(filas, modelo, max_textos):
    """Las primeras filas del lote cuyas torres suman <= `max_textos` textos distintos (al menos 4): las filas con 26 opciones agotan la memoria
    del retroceso con e5 descongelado. Determinista."""
    vistos, n = set(), 0
    for r in filas:
        nuevos = set(textos_torre_fila(r, modelo.texto_torre, modelo.cabeza_score, modelo.cabeza_noul)) - vistos
        if n >= 4 and len(vistos) + len(nuevos) > max_textos:
            break
        vistos |= nuevos
        n += 1
    return filas[:n]


# ---------------------------------------------------------------------------------------------------------------- entrenamiento
def lr_en(paso, total, base, warm=0.05):
    w = max(int(total * warm), 1)
    if paso < w:
        return base * (paso + 1) / w
    return base * 0.5 * (1 + math.cos(math.pi * (paso - w) / max(total - w, 1)))


def estado_ckpt(modelo, enc, paso, f1=None):
    return {"modelo": modelo.state_dict(), "cabezas": sorted(modelo.coral.keys()), "paso": paso, "f1": f1, "enc": enc.estado_enc(),
            "max_tok": enc.max_tok, "texto_torre": modelo.texto_torre, "cabeza_score": modelo.cabeza_score, "cabeza_noul": modelo.cabeza_noul,
            "sim": modelo.sim_modo, "k_ranuras": modelo.ranuras.q.shape[0]}


def guardar(ruta, estado):
    tmp = str(ruta) + ".tmp"
    torch.save(estado, tmp)
    Path(tmp).replace(ruta)


def cargar_partida(a, device):
    """(modelo, enc_estado, max_tok): el checkpoint de producción (`--partida RUTA.pt`) o ranuras y torre nuevas sobre e5 (`--partida e5`)."""
    if a.partida == "e5":
        return ModeloL2(k=a.k_ranuras, texto_torre="v1").to(device), None, a.max_tok or 256
    from model.l2.evaluar import cargar_modelo
    m, ck = cargar_modelo(a.partida, device)
    return m, ck.get("enc"), a.max_tok or int(ck.get("max_tok") or 256)


def evaluar_dev(modelo, enc, dev):
    from model.l2.evaluar import macro_f1_medio, predecir
    if not dev:
        return None
    modelo.eval()
    try:
        return macro_f1_medio(dev, predecir(modelo, enc, dev))[0]
    finally:
        modelo.train()


def coseno_e5(enc, ref, textos, lote=128):
    """Coseno medio entre los vectores de e5 del modelo y los del e5 original, para «query: texto» y «passage: texto»."""
    out = {}
    with torch.no_grad():
        for pre in ("query: ", "passage: "):
            cs = []
            for i in range(0, len(textos), lote):
                ts = [pre + normalizar_mayusculas(t) for t in textos[i:i + lote]]
                cs.append((enc._vectores(ts) * ref._vectores(ts)).sum(-1).float().cpu())
            out[pre.strip(": ")] = float(torch.cat(cs).mean())
    return out


def entrenar(a, enc=None, ref=None, log=print):
    torch.manual_seed(a.semilla)
    random.seed(a.semilla)
    np.random.seed(a.semilla)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    modelo, enc_estado, max_tok = cargar_partida(a, a.device)
    if enc is None:
        enc = Codificador(device=a.device, max_tok=max_tok, descongelar=a.descongelar)
        ref = Codificador(device=a.device, max_tok=max_tok, descongelar=0)
    enc.cargar_enc(enc_estado)
    train, dev, inf = preparar(a, ref, log)
    if not train:
        raise SystemExit("sin filas de entrenamiento")
    claves = sorted({clave_cabeza(r) for r in train + dev})
    for k in claves:
        modelo.asegurar_cabeza(k, a.device)
    grupos = modelo.grupos_optimizador(a.lr, a.lr_torre)
    if a.descongelar:
        grupos.append({"params": enc.parametros_entrenables(), "lr": a.lr_enc, "weight_decay": 0.0})
    opt = torch.optim.AdamW(grupos)
    bases = [g["lr"] for g in opt.param_groups]
    mues = D.Muestreador(train, a.lote, a.alfa, semilla=a.semilla)
    epi = EpisodiosProto(train, a.lote, a.k_proto, a.vias, a.semilla) if a.frac_proto > 0 else None
    inf["episodios_proto"] = {"tareas": epi.tareas if epi else [], "frac": a.frac_proto if epi and epi.tareas else 0.0}
    total = a.pasos or max(int(mues.pasos_por_epoca() * a.epocas), 1)
    cfg = cfg_de(a)
    parar = []
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: parar.append(1))
    hist = {"dev": [], "args": vars(a), "pasos_totales_inicial": total, "datos": inf}
    f1_0 = evaluar_dev(modelo, enc, dev[:a.dev_max])
    hist["dev"].append({"paso": 0, "macro_f1": f1_0})
    log("dev paso 0: macro-F1 medio %s" % (f1_0,))
    eval_cada = a.eval_cada or max(total // 3, 1)
    modelo.train()
    t0 = t_ck = time.time()
    paso, calibrado, acc, nepi, omitidos = 0, bool(a.pasos or not a.minutos), {}, 0, 0
    mejor = {"f1": -1.0 if f1_0 is None else f1_0, "paso": 0}
    while paso < total and not parar:
        for g, b in zip(opt.param_groups, bases):
            g["lr"] = lr_en(paso, total, b)
        r = np.random.default_rng([a.semilla, paso, 13]).random()
        if epi and epi.tareas and r < a.frac_proto:
            ep = epi.paso(paso)
            protos = {ep["tarea"]: prototipos(modelo, enc, ep["soporte"])}
            lote, solo = ep["filas"], True
            nepi += 1
        else:
            lote, protos, solo = mues.lote_paso(paso), None, False
        lote = recortar_lote(lote, modelo, a.max_textos) if not solo else lote
        opt.zero_grad(set_to_none=True)
        try:
            ps, _ = calcular(modelo, enc, ref, lote, cfg, protos, solo)
            ps["total"].backward()
        except torch.cuda.OutOfMemoryError:
            omitidos += 1
            log("sin memoria en el paso %d (%d filas): se omite (%d omitidos)" % (paso, len(lote), omitidos))
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if omitidos > max(5, 0.02 * (paso + 1)):
                raise SystemExit("demasiados pasos sin memoria: baja --lote o --max-textos")
            paso += 1
            continue
        torch.nn.utils.clip_grad_norm_([*modelo.parameters(), *enc.parametros_entrenables()], 1.0)
        opt.step()
        if a.descongelar:
            enc.invalidar()
        paso += 1
        for k, v in ps.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        if paso % a.log_cada == 0 or paso == 1:
            n = 1 if paso == 1 else a.log_cada
            log("paso %d/%d " % (paso, total) + " ".join("%s=%.4f" % (k, v / n) for k, v in acc.items())
                + " %.2f s/paso episodios=%d" % ((time.time() - t0) / paso, nepi))
            acc = {}
        if not calibrado and paso == min(30, total):
            por_paso = (time.time() - t0) / paso
            nuevo = paso + int(max(a.minutos * 60 - (time.time() - t0), 0) / por_paso * 0.95)
            if nuevo < total:
                log("calibrado: %.2f s/paso, el total pasa de %d a %d pasos" % (por_paso, total, nuevo))
                total = nuevo
                eval_cada = a.eval_cada or max(total // 3, 1)
            calibrado = True
        if paso % eval_cada == 0 and paso < total:
            f1 = evaluar_dev(modelo, enc, dev[:a.dev_max])
            hist["dev"].append({"paso": paso, "macro_f1": f1})
            log("dev paso %d: macro-F1 medio %s" % (paso, f1))
            if f1 is not None and f1 > mejor["f1"]:
                mejor.update(f1=f1, paso=paso)
                guardar(sal / "mejor.pt", estado_ckpt(modelo, enc, paso, f1))
        if time.time() - t_ck >= a.ckpt_cada:
            guardar(sal / "ultimo.pt", estado_ckpt(modelo, enc, paso))
            t_ck = time.time()
    f1 = evaluar_dev(modelo, enc, dev[:a.dev_max])
    hist["dev"].append({"paso": paso, "macro_f1": f1})
    log("dev final (paso %d): macro-F1 medio %s (paso 0: %s)" % (paso, f1, f1_0))
    guardar(sal / "final.pt", estado_ckpt(modelo, enc, paso, f1))
    if not (sal / "mejor.pt").exists():
        guardar(sal / "mejor.pt", estado_ckpt(modelo, enc, paso, f1))
    hist.update(pasos=paso, pasos_totales=total, episodios=nepi, omitidos_sin_memoria=omitidos, segundos=round(time.time() - t0, 1), interrumpido=bool(parar),
                mejor=mejor, cabezas=claves)
    if ref is not None and a.descongelar:
        hist["coseno_e5"] = coseno_e5(enc, ref, [r["texto"] for r in train[:200]])
        log("coseno con e5 original: %s" % hist["coseno_e5"])
    (sal / "historial.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    (sal / "datos.json").write_text(json.dumps(inf, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 143 if parar else 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entrenar")
    e.add_argument("--datos", required=True, help="raíz con los ficheros del manifiesto (con `blanda` ya aplicada: destilar --aplicar)")
    e.add_argument("--manifiesto")
    e.add_argument("--partida", required=True, help="checkpoint .pt de L2 de partida, o «e5» (ranuras y torre nuevas)")
    e.add_argument("--salida", required=True)
    e.add_argument("--dev-dir", help="carpeta con las tareas DEV de v2 (.jsonl): solo para quitar casi-duplicados del train")
    e.add_argument("--blandas-resumen", help="resumen de `destilar --aplicar` (para contar las blandas que se pierden por conjuntos excluidos)")
    e.add_argument("--umbral-dup", type=float, default=0.95, help="coseno e5 a partir del cual una fila de train se quita (> 1 = no filtrar)")
    e.add_argument("--con-sin-blanda", action="store_true", help="admite filas de train sin blanda (por defecto, solo con blanda)")
    e.add_argument("--tope-tarea", type=int, default=1000)
    e.add_argument("--max-filas", type=int, default=400_000)
    e.add_argument("--epocas", type=float, default=1.5)
    e.add_argument("--pasos", type=int, default=0, help="fija el total de pasos (pruebas)")
    e.add_argument("--minutos", type=float, default=0, help="tope de reloj: tras 30 pasos recalcula el total con la velocidad medida")
    e.add_argument("--lote", type=int, default=32)
    e.add_argument("--alfa", type=float, default=D.ALFA)
    e.add_argument("--max-tok", type=int, default=0, help="0 = el del checkpoint de partida")
    e.add_argument("--k-ranuras", type=int, default=16)
    e.add_argument("--descongelar", type=int, default=4, help="últimas N capas de e5 que se entrenan (0 = e5 congelado, sin penalidad)")
    e.add_argument("--lr", type=float, default=5e-5)
    e.add_argument("--lr-torre", type=float, default=2e-5)
    e.add_argument("--lr-enc", type=float, default=1e-5)
    e.add_argument("--peso-kl", type=float, default=PESOS["kl"])
    e.add_argument("--peso-ce", type=float, default=PESOS["ce"])
    e.add_argument("--peso-contr", type=float, default=PESOS["contrastivo"])
    e.add_argument("--peso-ref", type=float, default=2.0, help="peso de la penalidad al alejarse de e5 original")
    e.add_argument("--temp-kl", type=float, default=T_KL)
    e.add_argument("--minadas", type=int, default=8, help="negativos minados de otras tareas del lote por fila")
    e.add_argument("--desacuerdo", choices=("intercambiar", "ignorar", "oro"), default="intercambiar")
    e.add_argument("--score-perdida", choices=("emd", "suavizado"), default="emd")
    e.add_argument("--frac-proto", type=float, default=0.2)
    e.add_argument("--k-proto", type=int, default=8)
    e.add_argument("--vias", type=int, default=16)
    e.add_argument("--max-textos", type=int, default=500, help="tope de textos de torre distintos por lote (recorta filas del final)")
    e.add_argument("--dev-max", type=int, default=1200)
    e.add_argument("--eval-cada", type=int, default=0)
    e.add_argument("--log-cada", type=int, default=20)
    e.add_argument("--ckpt-cada", type=int, default=600)
    e.add_argument("--semilla", type=int, default=0)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s = sub.add_parser("sintetico")
    s.add_argument("--salida", required=True)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == "sintetico":
        from . import r10_sintetico
        r10_sintetico.generar(a.salida)
        return 0
    return entrenar(a, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main())
