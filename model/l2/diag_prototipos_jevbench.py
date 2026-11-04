"""Diagnóstico: ¿por qué los prototipos con k=8 por clase empeoran JevBench (0,230 frente a 0,489 sin soporte)?

    python -m model.l2.diag_prototipos_jevbench --ckpt mejor.pt --jevbench RUTA --salida DIR [--k 8] [--max-casos N] [--device cpu]

Dónde se construyen los prototipos: `model.l2.modelo.prototipos` (media normalizada, ranura a ranura, de las ranuras S de los ejemplos
de soporte de cada clase; con κ > 0 se mezcla con la torre de la descripción), `ModeloL2._simproto` (coseno medio de S con P por
opción, 0 si la opción no tiene prototipo) y `ModeloL2.forward`, que lo suma al MaxSim como `s + β·simproto` (choice) o con el ajuste
lineal `_ajuste` (noul, score). El soporte de JevBench sale de `model.evaluacion.jevbench.soporte` (k por opción, sin el caso ni
sus paráfrasis) y la evaluación de `model.evaluacion.jevbench_l2`.

Hipótesis que comprueba (cada una con la variante que la prueba; todas sobre las mismas filas, exactitud global y por tipo):
H1 cableado: β = 0 con soporte tiene que dar exactamente lo mismo que k = 0. Si no, el soporte cambia la predicción por otra vía.
H2 normalización y escala: el MaxSim (softplus, ≥ 0,69) y el coseno del prototipo (en [−1, 1]) se suman con β = 1 y T muy pequeña;
   se prueban β ∈ {0,1; 0,3; 3; 10} y el prototipo sin normalizar (media de ranuras, norma ≤ 1).
H3 qué torre/prefijo: los prototipos salen de las ranuras del mensaje («query: »), no de la torre de opción («passage: »). κ = ∞
   usa solo la torre de la descripción por la vía de los prototipos (debería reproducir k = 0); κ = 8 mezcla.
H4 muestreo del soporte con k=8: tamaño real del soporte por clase, fracción de casos con soporte vacío o que no cubre todas las
   opciones, y la variación entre tres semillas de muestreo.
H5 mapeo de etiquetas: fracción de pares (caso, opción) sin prototipo (valen 0 en `_simproto`: penalizan a las opciones no vistas
   frente a las vistas), casos cuyo oro no tiene prototipo, soporte con etiquetas fuera de las opciones y, como control, las
   etiquetas del soporte barajadas: si la exactitud no baja, el soporte no se está usando como etiqueta → clase.
Solo imprime y guarda agregados.
"""
import argparse
import contextlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from model.evaluacion import chispa_eval as CE
from model.evaluacion.jevbench import cargar_jevbench, cubre_opciones, pliegues
from model.evaluacion.jevbench_l2 import _con_tarea_esquema


# ---------------------------------------------------------------- piezas
@contextlib.contextmanager
def beta_fija(modelo, valor):
    """Pone todos los β a `valor` mientras dure el bloque (None = no toca)."""
    if valor is None:
        yield
        return
    antes = {k: p.detach().clone() for k, p in modelo.beta.items()}
    with torch.no_grad():
        for p in modelo.beta.values():
            p.fill_(valor)
    try:
        yield
    finally:
        with torch.no_grad():
            for k, p in modelo.beta.items():
                p.copy_(antes[k])


@torch.no_grad()
def prototipos_sin_normalizar(modelo, enc, filas_soporte, lote=256):
    """Como `modelo.prototipos` pero la media de las ranuras va sin renormalizar (norma ≤ 1)."""
    dev = modelo.ranuras.q.device
    etqs = sorted({r["etiqueta"] for r in filas_soporte})
    idx = {e: i for i, e in enumerate(etqs)}
    suma, n = None, torch.zeros(len(etqs), device=dev)
    for i in range(0, len(filas_soporte), lote):
        fs = filas_soporte[i:i + lote]
        H, m = enc.tokens([r["texto"] for r in fs])
        S = modelo.ranuras(H.to(dev), m.to(dev))
        ci = torch.tensor([idx[r["etiqueta"]] for r in fs], device=dev)
        parte = torch.zeros(len(etqs), *S.shape[1:], device=dev).index_add(0, ci, S)
        suma = parte if suma is None else suma + parte
        n = n.index_add(0, ci, torch.ones(len(fs), device=dev))
    return idx, suma / n.clamp(min=1).view(-1, 1, 1)


def mezclar_etiquetas(sop, semilla=0):
    """Control: las mismas etiquetas del soporte repartidas al azar entre sus ejemplos."""
    ys = [r["etiqueta"] for r in sop]
    random.Random(semilla).shuffle(ys)
    return [dict(r, etiqueta=y) for r, y in zip(sop, ys)]


@torch.no_grad()
def predecir(modelo, enc, filas, k, semilla=0, proto=None, beta=None, kappa=0.0, mezclar=False, lote=64):
    """{id: {opción: p}} con soporte k. `proto`: función (modelo, enc, soporte) → (idx, P) (por defecto `modelo.prototipos`)."""
    from .modelo import prototipos
    proto = proto or (lambda m, e, s: prototipos(m, e, s, kappa=kappa))
    out = {}
    with beta_fija(modelo, beta):
        for _, rs, sop in (pliegues(filas, k, semilla) if k else [(r["id"], [r], []) for r in filas]):
            if mezclar and sop:
                sop = mezclar_etiquetas(sop, semilla)
            protos = {rs[0]["tarea"]: proto(modelo, enc, sop)} if sop else None
            for i in range(0, len(rs), lote):
                fs = rs[i:i + lote]
                probs, _ = modelo(fs, enc, entrenar=False, protos=protos)
                for r, p in zip(fs, probs):
                    p = p.double().cpu()
                    out[r["id"]] = dict(zip(r["opciones"], (p / p.sum()).tolist()))
    return out


def exactitud(filas, preds):
    """Exactitud global y por tipo (score por mediana CORAL, como el arnés)."""
    res = {}
    for nombre, sub in [("global", filas)] + [(t, [r for r in filas if r["tipo"] == t]) for t in ("choice", "noul", "score")]:
        if sub:
            res[nombre] = {"n": len(sub), "exactitud": CE.evaluar(sub, [{"id": r["id"], "probs": preds[r["id"]]} for r in sub])["global"]["exactitud"]}
    return res


def estadisticas_soporte(filas_esq, k, semilla=0):
    """Hipótesis H4 y H5 sin el modelo: cobertura y tamaño del soporte, opciones sin prototipo, fugas."""
    n_casos = vacios = sin_cubrir = oro_sin = pares = pares_sin = fuera = fuga = 0
    por_clase = []
    for _, rs, sop in pliegues(filas_esq, k, semilla):
        etqs = Counter(r["etiqueta"] for r in sop)
        por_clase += list(etqs.values())
        for r in rs:
            n_casos += 1
            vacios += not sop
            sin_cubrir += not cubre_opciones(r, sop)
            oro_sin += r["etiqueta"] not in etqs
            pares += len(r["opciones"])
            pares_sin += sum(o not in etqs for o in r["opciones"])
            fuera += sum(e not in r["opciones"] for e in etqs)
            fuga += sum(1 for s in sop if s["id"] == r["id"] or (r.get("grupo") and s.get("grupo") == r["grupo"]))
    return {
        "casos": n_casos, "soporte_vacio": vacios / max(n_casos, 1), "no_cubre_todas_las_opciones": sin_cubrir / max(n_casos, 1),
        "pares_opcion_sin_prototipo": pares_sin / max(pares, 1), "oro_sin_prototipo": oro_sin / max(n_casos, 1),
        "etiquetas_de_soporte_fuera_de_opciones": fuera, "fugas_caso_o_parafrasis_en_soporte": fuga,
        "ejemplos_por_clase": {"min": min(por_clase, default=0), "media": sum(por_clase) / max(len(por_clase), 1), "max": max(por_clase, default=0)},
    }


@torch.no_grad()
def escalas(modelo, enc, filas, k, muestra=200):
    """H2: medias de MaxSim y del coseno de prototipo sobre una muestra de casos choice (rango y separación oro−resto)."""
    from .modelo import prototipos, simproto, texto_opcion
    dev = modelo.ranuras.q.device
    s_vals, p_vals, d_s, d_p = [], [], [], []
    for _, rs, sop in pliegues(filas, k)[:muestra]:
        r = rs[0]
        if r["tipo"] != "choice" or not sop:
            continue
        idx, P = prototipos(modelo, enc, sop)
        H, m = enc.tokens([r["texto"]])
        S = modelo.ranuras(H.to(dev), m.to(dev))
        h = modelo.torre(enc.pasaje([texto_opcion(r, o, modelo.texto_torre) for o in r["opciones"]]).to(dev))
        s = modelo.sim(S, h)[0].cpu()
        sp = simproto(S, P)[0].cpu()
        col = [idx.get(o, -1) for o in r["opciones"]]
        if min(col) < 0:
            continue
        sp = sp[col]
        oro = r["opciones"].index(r["etiqueta"])
        resto = [i for i in range(len(col)) if i != oro]
        s_vals += s.tolist()
        p_vals += sp.tolist()
        d_s.append(float(s[oro] - s[resto].mean()))
        d_p.append(float(sp[oro] - sp[resto].mean()))
    med = lambda v: sum(v) / len(v) if v else float("nan")
    return {"casos": len(d_s), "maxsim_media": med(s_vals), "maxsim_min": min(s_vals, default=float("nan")), "maxsim_max": max(s_vals, default=float("nan")),
            "simproto_media": med(p_vals), "simproto_min": min(p_vals, default=float("nan")), "simproto_max": max(p_vals, default=float("nan")),
            "separacion_oro_maxsim": med(d_s), "separacion_oro_simproto": med(d_p), "T_choice": float(modelo.log_t["choice"].exp()),
            "beta": {t: float(b) for t, b in modelo.beta.items()}}


def diagnosticar(modelo, enc, filas, k=8, semillas=(1, 2)):
    modelo.eval()
    f = _con_tarea_esquema(filas)
    v = {}
    v["k0_sin_soporte"] = exactitud(f, predecir(modelo, enc, f, 0))
    v["k_tal_cual"] = exactitud(f, predecir(modelo, enc, f, k))
    v["H1_beta0"] = exactitud(f, predecir(modelo, enc, f, k, beta=0.0))
    for b in (0.1, 0.3, 3.0, 10.0):
        v["H2_beta_%g" % b] = exactitud(f, predecir(modelo, enc, f, k, beta=b))
    v["H2_prototipo_sin_normalizar"] = exactitud(f, predecir(modelo, enc, f, k, proto=lambda m, e, s: prototipos_sin_normalizar(m, e, s)))
    v["H3_kappa_inf_solo_torre"] = exactitud(f, predecir(modelo, enc, f, k, kappa=float("inf")))
    v["H3_kappa_8_mezcla"] = exactitud(f, predecir(modelo, enc, f, k, kappa=8.0))
    for s in semillas:
        v["H4_semilla_%d" % s] = exactitud(f, predecir(modelo, enc, f, k, semilla=s))
    v["H5_etiquetas_barajadas"] = exactitud(f, predecir(modelo, enc, f, k, mezclar=True))
    res = {"k": k, "casos": len(f), "variantes": v, "soporte": estadisticas_soporte(f, k), "escalas": escalas(modelo, enc, f, k)}
    ex = lambda n: v[n]["global"]["exactitud"]
    res["lectura"] = {
        "H1_cableado_ok": abs(ex("H1_beta0") - ex("k0_sin_soporte")) < 1e-9,
        "H2_algun_beta_recupera": max(ex("H2_beta_0.1"), ex("H2_beta_0.3")) >= ex("k0_sin_soporte") - 0.01,
        "H3_torre_sola_reproduce_k0": abs(ex("H3_kappa_inf_solo_torre") - ex("k0_sin_soporte")) < 0.05,
        "H4_variacion_entre_semillas": max([ex("k_tal_cual")] + [ex("H4_semilla_%d" % s) for s in semillas]) - min([ex("k_tal_cual")] + [ex("H4_semilla_%d" % s) for s in semillas]),
        "H5_barajar_etiquetas_baja": ex("k_tal_cual") - ex("H5_etiquetas_barajadas"),
    }
    return res


def md(res):
    L = ["# Prototipos con k=%d en JevBench (%d casos)" % (res["k"], res["casos"]), "", "| variante | exactitud | choice | noul | score |", "|---|---|---|---|---|"]
    for n, r in res["variantes"].items():
        L.append("| %s | %s |" % (n, " | ".join("%.3f" % r[t]["exactitud"] if t in r else "n/d" for t in ("global", "choice", "noul", "score"))))
    L += ["", "Soporte: `%s`" % json.dumps(res["soporte"], ensure_ascii=False), "", "Escalas: `%s`" % json.dumps(res["escalas"], ensure_ascii=False), "",
          "Lectura: `%s`" % json.dumps(res["lectura"], ensure_ascii=False)]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-casos", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    from .evaluar import cargar_modelo
    from .modelo import MAX_TOK, Codificador
    filas = cargar_jevbench(a.jevbench)
    if a.max_casos:
        filas = filas[:a.max_casos]
    modelo, ck = cargar_modelo(a.ckpt, a.device)
    enc = Codificador(device=a.device, max_tok=ck.get("max_tok", MAX_TOK))
    enc.cargar_enc(ck.get("enc"))
    res = diagnosticar(modelo, enc, filas, a.k)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "diag_prototipos.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (sal / "diag_prototipos.md").write_text(md(res), encoding="utf-8")
    print(md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
