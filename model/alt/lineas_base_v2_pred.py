"""R10-T3: predictores de las líneas base que no son LLM por logprobs (torch/transformers; la medida está en `lineas_base_v2.py`).

    python -m model.alt.lineas_base_v2_pred nli --modelo DIR_MDEBERTA --dev DIR_DEV --salida DIR_PRED/mdeberta-nli [--max-filas N] [--dispositivo cuda]
    python -m model.alt.lineas_base_v2_pred l2  --m1 mejor.pt --dev DIR_DEV --salida DIR_PRED/l2-zs [--max-filas N] [--dispositivo cuda]
    python -m model.alt.lineas_base_v2_pred knn --dev DIR_DEV --vec DIR_VEC --salida DIR_PRED/knn-e5-k8 [--k 8] [--max-filas N]

Escriben DIR_SALIDA/<tarea>.jsonl con {id, probs, valida, ms}; `ms` es el tiempo por fila (NLI: todos sus pares de la fila; L2 y kNN: la media del lote).
`nli`: zero-shot por implicación. Premisa = el texto (800 caracteres, 192 tokens en total); hipótesis = la pregunta + «Respuesta:» + la descripción de la opción (o su id si no la tiene);
puntuación de cada opción = logit de implicación − logit de contradicción; probs = softmax sobre las opciones de la fila.
`knn`: kNN-e5 con k ejemplos de oro por clase (`fusion_knn_l2.muestrear`, semilla 0), 32 vecinos, temperatura 0,05 (`fusion_knn_l2.knn`), con los
vectores de `knn_soporte_llm vectores` (media de tokens de e5, «query: », 256 tokens, transformers fijado). Los vectores se calculan con `vectores`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HIPOTESIS_MAX = 400
PREMISA_MAX = 800
MAX_TOK_NLI = 192


def tareas(dev, max_filas):
    from model.alt.lineas_base_v2 import cargar_dev
    return {t: (sop, ev[:max_filas]) for t, (sop, ev) in cargar_dev(dev).items()}


def escribir(sal, t, filas):
    Path(sal).mkdir(parents=True, exist_ok=True)
    with open(Path(sal) / (t + ".jsonl"), "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas))


def hipotesis(r, o):
    d = (r.get("descripciones") or {}).get(o) or o
    return ("%s Respuesta: %s" % (r["pregunta"], d))[:HIPOTESIS_MAX]


def cmd_nli(a):
    import torch
    torch.set_num_threads(a.hilos)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.modelo)
    m = AutoModelForSequenceClassification.from_pretrained(a.modelo).to(a.dispositivo).eval()
    if a.dispositivo != "cpu":
        m = m.half()
    elif a.bf16:   # CPU con AMX/AVX512-bf16 (c7i): bf16 en vez de fp32
        m = m.to(torch.bfloat16)
    lab = {int(i): str(n).lower() for i, n in m.config.id2label.items()}
    ie = next(i for i, n in lab.items() if n.startswith("entail"))
    ic = next(i for i, n in lab.items() if n.startswith("contra"))
    print("nli etiquetas", lab, "entail", ie, "contra", ic, flush=True)
    for t, (_, ev) in sorted(tareas(a.dev, a.max_filas).items(), key=lambda kv: len(kv[1][1]) * len(kv[1][1][0]["opciones"])):   # lo barato primero: un corte deja lo hecho
        out = []
        t_ini = time.time()
        for r in ev:
            if (a.seg_tarea and time.time() - t_ini > a.seg_tarea) or (a.hasta and time.time() > a.hasta):
                print("nli corte por tiempo en", t, len(out), "de", len(ev), flush=True)   # tope por tarea: lo hecho queda; medir cuenta el resto como fallo y da la cobertura
                break
            ops = r["opciones"]
            t0 = time.perf_counter()
            s = []
            with torch.no_grad():
                for i in range(0, len(ops), a.lote):
                    h = [hipotesis(r, o) for o in ops[i:i + a.lote]]
                    x = tok([r["texto"][:PREMISA_MAX]] * len(h), h, truncation="only_first", max_length=MAX_TOK_NLI, padding=True, return_tensors="pt").to(a.dispositivo)
                    lg = m(**x).logits.float()
                    s += (lg[:, ie] - lg[:, ic]).cpu().tolist()
            e = np.exp(np.array(s) - max(s))
            p = e / e.sum()
            out.append({"id": r["id"], "probs": dict(zip(ops, map(float, p))), "valida": True, "ms": round((time.perf_counter() - t0) * 1000, 3)})
        escribir(a.salida, t, out)
        print("nli", t, len(out), flush=True)


def cmd_l2(a):
    import torch
    torch.set_num_threads(a.hilos)
    from model.l2.evaluar import cargar_modelo, predecir
    from model.l2.modelo import MAX_TOK, Codificador
    modelo, ck = cargar_modelo(a.m1, a.dispositivo)
    enc = Codificador(device=a.dispositivo, max_tok=ck.get("max_tok", MAX_TOK))   # como model.alt.p0_medir.pred_l2, sin soporte
    for t, (_, ev) in tareas(a.dev, a.max_filas).items():
        ps = predecir(modelo, enc, ev, lote=64)
        escribir(a.salida, t, [{"id": p["id"], "probs": p["probs"], "valida": True, "ms": p["ms"]} for p in ps])
        print("l2", t, len(ps), flush=True)


def cmd_vectores(a):
    """Los vectores de `knn_soporte_llm vectores` (mismo código, mismo e5, transformers fijado), sin fondo."""
    from model.alt import knn_soporte_llm as K
    vacio = Path(a.vec) / "vacio.jsonl"
    Path(a.vec).mkdir(parents=True, exist_ok=True)
    vacio.write_text("", encoding="utf-8")
    from model.alt.lineas_base_v2 import cargar_dev
    if a.solo:   # solo esas tareas (carpeta aparte con sus jsonl)
        solo = Path(a.vec) / "dev-solo"
        solo.mkdir(parents=True, exist_ok=True)
        for t in a.solo.split(","):
            (solo / (t + ".jsonl")).write_bytes((Path(a.dev) / (t + ".jsonl")).read_bytes())
        a.dev = str(solo)
    n = sum(len(s) + len(e) for s, e in cargar_dev(a.dev).values())
    t0 = time.perf_counter()
    K.cmd_vectores(argparse.Namespace(bateria=a.dev, fondo=str(vacio), salida=a.vec, hilos=a.hilos, max_fondo=0, sin_bateria=False, transformers=a.transformers))
    (Path(a.vec) / "tiempo.json").write_text(json.dumps({"segundos": round(time.perf_counter() - t0, 2), "textos": n}), encoding="utf-8")


def cmd_topk(a):
    """Preselección top-k (R10-T3b, V4) de una tarea con muchas opciones: escribe SAL/test/<tarea>.jsonl y SAL/soporte/<tarea>.jsonl con `opciones`
    y `descripciones` recortadas a las k candidatas (orden alfabético) y SAL/preseleccion.json con las cifras (recall@k, kNN top-1), sin textos."""
    from model.alt.fusion_knn_l2 import knn, muestrear
    t = a.tarea
    sop, ev = tareas(a.dev, None)[t]
    z = np.load(Path(a.vec) / (t + ".npz"))
    if list(z["idtr"]) != [r["id"] for r in sop] or list(z["idte"]) != [r["id"] for r in ev]:
        raise SystemExit("los vectores de %s no corresponden a las filas de dev" % t)
    ys = [r["etiqueta"] for r in sop]
    idx = muestrear(ys, a.k_oro, 0)
    ops = ev[0]["opciones"]

    def candidatas(x, rel, forzar=None):
        P, _, C = knn(x[None, :], z["Xtr"][rel], [ys[i] for i in rel], [ops])
        orden = np.lexsort((-C[0, :len(ops)], -P[0, :len(ops)]))   # P descendente; desempate por el centroide
        top = [ops[i] for i in orden[:a.k]]
        if forzar is not None and forzar not in top:
            top[-1] = forzar
        return top, ops[int(np.argmax(P[0, :len(ops)]))]

    def recortar(r, top):
        g = dict(r)
        g["opciones"] = sorted(top)
        g["descripciones"] = {o: d for o, d in (r.get("descripciones") or {}).items() if o in top}
        return g

    sal = Path(a.salida)
    nt = nk = rec = 0
    filas = []
    for r, x in zip(ev, z["Xte"]):
        top, k1 = candidatas(x, idx)
        nt, nk, rec = nt + 1, nk + (k1 == r["etiqueta"]), rec + (r["etiqueta"] in top)
        filas.append(recortar(r, top))
    escribir(sal / "test", t, filas)
    forzadas, filas = 0, []
    for i, (r, x) in enumerate(zip(sop, z["Xtr"])):
        rel = np.array([j for j in idx if j != i])   # kNN dejando fuera la propia fila
        forzadas += r["etiqueta"] not in candidatas(x, rel)[0]
        filas.append(recortar(r, candidatas(x, rel, forzar=r["etiqueta"])[0]))
    escribir(sal / "soporte", t, filas)
    cif = {"tarea": t, "k": a.k, "k_oro": a.k_oro, "n_test": nt, "n_soporte": len(sop), "recall_at_k": round(rec / nt, 4), "knn_top1": round(nk / nt, 4),
           "soporte_oro_forzado": int(forzadas), "transformers": str(z["transformers"])}
    (sal / "preseleccion.json").write_text(json.dumps(cif, indent=1) + "\n", encoding="utf-8")
    print("topk", json.dumps(cif), flush=True)


def cmd_knn(a):
    from model.alt.fusion_knn_l2 import knn, muestrear
    tv = json.loads((Path(a.vec) / "tiempo.json").read_text(encoding="utf-8"))
    ms_enc = 1000.0 * tv["segundos"] / max(tv["textos"], 1)
    for t, (sop, ev) in tareas(a.dev, None).items():
        z = np.load(Path(a.vec) / (t + ".npz"))
        if list(z["idtr"]) != [r["id"] for r in sop] or list(z["idte"]) != [r["id"] for r in ev]:
            raise SystemExit("los vectores de %s no corresponden a las filas de dev" % t)
        ev = ev[:a.max_filas]
        ys = [r["etiqueta"] for r in sop]
        idx = muestrear(ys, a.k, 0)
        t0 = time.perf_counter()
        P, _, _ = knn(z["Xte"][:len(ev)], z["Xtr"][idx], [ys[i] for i in idx], [r["opciones"] for r in ev])
        ms = round((time.perf_counter() - t0) * 1000 / max(len(ev), 1) + ms_enc, 3)   # kNN + la codificación de la consulta con e5
        escribir(a.salida, t, [{"id": r["id"], "probs": dict(zip(r["opciones"], map(float, P[i, :len(r["opciones"])]))), "valida": True, "ms": ms} for i, r in enumerate(ev)])
        print("knn", t, len(ev), "soporte", len(idx), "transformers", str(z["transformers"]), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ("nli", "l2", "knn", "vectores", "topk"):
        p = sub.add_parser(nombre)
        p.add_argument("--dev", required=True)
        p.add_argument("--max-filas", type=int)
        p.add_argument("--hilos", type=int, default=4)
        if nombre not in ("vectores",):
            p.add_argument("--salida", required=True)
        if nombre in ("nli", "l2"):
            p.add_argument("--dispositivo", default="cpu")
        if nombre == "nli":
            p.add_argument("--modelo", required=True)
            p.add_argument("--lote", type=int, default=32)
            p.add_argument("--bf16", action="store_true")
            p.add_argument("--seg-tarea", type=float, default=0, help="tope de segundos por tarea (0: sin tope)")
            p.add_argument("--hasta", type=float, default=0, help="fecha límite (epoch) de todo el NLI (0: sin tope)")
        if nombre == "l2":
            p.add_argument("--m1", required=True)
        if nombre in ("knn", "vectores", "topk"):
            p.add_argument("--vec", required=True)
        if nombre == "topk":
            p.add_argument("--tarea", required=True)
            p.add_argument("--k", type=int, default=10)
            p.add_argument("--k-oro", type=int, default=8)
        if nombre == "knn":
            p.add_argument("--k", type=int, default=8)
        if nombre == "vectores":
            p.add_argument("--solo", help="solo estas tareas (separadas por comas)")
            p.add_argument("--transformers", required=True, help="versión de transformers exigida (la instalada)")
    a = ap.parse_args(argv)
    {"nli": cmd_nli, "l2": cmd_l2, "knn": cmd_knn, "vectores": cmd_vectores, "topk": cmd_topk}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
