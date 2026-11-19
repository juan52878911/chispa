"""R11: tres alternativas nuevas en las 6 tareas de DESARROLLO de la batería v2 (criterio congelado en `crit-r11.json`).

    python -m model.alt.alternativas_r11 vec-qwen --modelo DIR --dev DIR_DEV --salida DIR_VECQ [--dispositivo cuda] [--max-filas N]
    python -m model.alt.alternativas_r11 op-e5    --dev DIR_DEV --vec DIR_VEC_E5                      (descripciones de etiqueta con e5, CPU)
    python -m model.alt.alternativas_r11 knn      --dev DIR_DEV --vec DIR_VEC --vecinos 32 --salida DIR_PRED/x
    python -m model.alt.alternativas_r11 zs-emb   --dev DIR_DEV --vec DIR_VEC --salida DIR_PRED/x
    python -m model.alt.alternativas_r11 rerank   --modelo DIR --dev DIR_DEV --salida DIR_PRED/rerank [--dispositivo cuda] [--max-filas N] [--hasta EPOCH]
    python -m model.alt.alternativas_r11 medir    --dev DIR_DEV --pred DIR_PRED --ref DIR_REF --crit crit-r11.json --salida cifras.json

A: Qwen3-Embedding-0.6B en lugar de e5-small (kNN con 32 y 8 vecinos, y zero-shot por descripción). Los vectores van en el mismo formato que los de
   `lineas_base_v2_pred vectores` (<tarea>.npz con Xtr, Xte, idtr, idte, transformers), así el kNN, la cascada de E y la preselección los leen igual;
   las descripciones de etiqueta van en op-<tarea>.npz (Xop, ops). lat-emb.json: ms por consulta (las 15 primeras de test de cada tarea, de una en una).
B: Qwen3-Reranker-0.6B como zero-shot: una pregunta de sí o no por etiqueta; <tarea>.jsonl con {id, probs (softmax de los logits de «yes»), yes, lsm
   (log p(yes) frente a «no», la puntuación de la tarjeta, informativa), valida, ms}; lat-rerank.json como lat-emb.json.
C: no se mide (licencia; ver el criterio).
`medir` no necesita torch: lee las predicciones y las referencias (REF/{4b-zs,zs-topk,l2-zs,knn-e5-k8-t3a}) y aplica el criterio.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

INSTR_EMB = "Classify the text to answer this question: %s"
MAX_TOK_EMB = 320
MAX_TOK_TEXTO_RR = 384
MAX_TOK_RR = 1024
PREFIJO_RR = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
              "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFIJO_RR = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
N_LAT = 15
TEMP = 0.05
REMUESTREOS = 2000
SINTETICAS = ("finanzas-documentos", "horeca-resenas")


def tareas(dev, max_filas=None):
    from model.alt.lineas_base_v2 import cargar_dev
    return {t: (sop, ev[:max_filas]) for t, (sop, ev) in cargar_dev(dev).items()}


def escribir(sal, t, filas):
    Path(sal).mkdir(parents=True, exist_ok=True)
    with open(Path(sal) / (t + ".jsonl"), "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas))


def desc(r, o):
    return (r.get("descripciones") or {}).get(o) or o


def consulta_emb(r):
    return "Instruct: %s\nQuery:%s" % (INSTR_EMB % r["pregunta"], r["texto"])


def cuerpo_rr(r, texto, o):
    return "<Instruct>: %s\n<Query>: %s\n<Document>: %s" % (r["pregunta"], texto, desc(r, o))


def sincronizar(dispositivo):
    if str(dispositivo).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


# ---------------------------------------------------------------- A: Qwen3-Embedding
class EmbQwen:
    def __init__(self, ruta, dispositivo):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.dev = torch, dispositivo
        self.tok = AutoTokenizer.from_pretrained(ruta, padding_side="left", local_files_only=True)
        dt = torch.bfloat16 if str(dispositivo).startswith("cuda") else torch.float32
        self.m = AutoModel.from_pretrained(ruta, dtype=dt, local_files_only=True).to(dispositivo).eval()

    def __call__(self, textos, lote=32):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(textos), lote):
                b = self.tok(textos[i:i + lote], padding=True, truncation=True, max_length=MAX_TOK_EMB, return_tensors="pt").to(self.dev)
                h = self.m(**b).last_hidden_state[:, -1]          # relleno a la izquierda: la última posición es el último token de todas
                out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, self.m.config.hidden_size), np.float32)


def cmd_vec_qwen(a):
    import transformers
    enc = EmbQwen(a.modelo, a.dispositivo)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    tv = np.array(transformers.__version__)
    lat, t0, n = {}, time.perf_counter(), 0
    for t, (sop, ev) in tareas(a.dev, a.max_filas).items():
        Xtr, Xte = enc([consulta_emb(r) for r in sop]), enc([consulta_emb(r) for r in ev])
        ops = ev[0]["opciones"]
        Xop = enc([desc(ev[0], o) for o in ops])
        n += len(sop) + len(ev) + len(ops)
        np.savez(sal / (t + ".npz"), Xtr=Xtr, Xte=Xte, idtr=np.array([r["id"] for r in sop]), idte=np.array([r["id"] for r in ev]), transformers=tv)
        np.savez(sal / ("op-" + t + ".npz"), Xop=Xop, ops=np.array(ops))
        ms = []
        for r in ev[:N_LAT]:   # latencia de una consulta: codificarla sola + el producto con el soporte
            sincronizar(a.dispositivo)
            s = time.perf_counter()
            q = enc([consulta_emb(r)], lote=1)
            _ = q @ Xtr.T
            sincronizar(a.dispositivo)
            ms.append(1000 * (time.perf_counter() - s))
        lat[t] = [round(x, 3) for x in ms]
        print("vec-qwen", t, Xtr.shape, Xte.shape, flush=True)
    (sal / "tiempo.json").write_text(json.dumps({"segundos": round(time.perf_counter() - t0, 2), "textos": n}), encoding="utf-8")
    (sal / "lat-emb.json").write_text(json.dumps(lat), encoding="utf-8")


def cmd_op_e5(a):
    """Vectores e5 de las descripciones de etiqueta (el mismo Codificador y la misma función que los textos de T3a: «query: », media, 256 tokens)."""
    from model.l2.linea_base_knn_e5 import vectores
    from model.l2.modelo import Codificador
    enc = Codificador(device="cpu", max_tok=256)
    for t, (_, ev) in tareas(a.dev).items():
        ops = ev[0]["opciones"]
        np.savez(Path(a.vec) / ("op-" + t + ".npz"), Xop=vectores(enc, [desc(ev[0], o) for o in ops]).numpy(), ops=np.array(ops))
        print("op-e5", t, len(ops), flush=True)


def cargar_vec(vec, t, sop, ev):
    z = np.load(Path(vec) / (t + ".npz"))
    if list(z["idtr"]) != [r["id"] for r in sop] or list(z["idte"][:len(ev)]) != [r["id"] for r in ev]:
        raise SystemExit("los vectores de %s no corresponden a las filas de dev" % t)
    return z


def cmd_knn(a):
    from model.alt.fusion_knn_l2 import knn, muestrear
    for t, (sop, ev) in tareas(a.dev, a.max_filas).items():
        z = cargar_vec(a.vec, t, sop, ev)
        ys = [r["etiqueta"] for r in sop]
        idx = muestrear(ys, 8, 0)
        P, _, _ = knn(z["Xte"][:len(ev)], z["Xtr"][idx], [ys[i] for i in idx], [r["opciones"] for r in ev], vecinos=a.vecinos, temp=TEMP)
        escribir(a.salida, t, [{"id": r["id"], "probs": dict(zip(r["opciones"], map(float, P[i, :len(r["opciones"])]))), "valida": True} for i, r in enumerate(ev)])
        print("knn", t, len(ev), "vecinos", a.vecinos, flush=True)


def softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def cmd_zs_emb(a):
    for t, (sop, ev) in tareas(a.dev, a.max_filas).items():
        z = cargar_vec(a.vec, t, sop, ev)
        o = np.load(Path(a.vec) / ("op-" + t + ".npz"))
        ops = list(o["ops"])
        S = z["Xte"][:len(ev)] @ o["Xop"].T
        filas = []
        for i, r in enumerate(ev):
            j = [ops.index(x) for x in r["opciones"]]
            filas.append({"id": r["id"], "probs": dict(zip(r["opciones"], map(float, softmax(S[i, j] / TEMP)))), "valida": True})
        escribir(a.salida, t, filas)
        print("zs-emb", t, len(ev), flush=True)


# ---------------------------------------------------------------- B: Qwen3-Reranker
class Reranker:
    def __init__(self, ruta, dispositivo):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.dev = torch, dispositivo
        self.tok = AutoTokenizer.from_pretrained(ruta, padding_side="left", local_files_only=True)
        dt = torch.bfloat16 if str(dispositivo).startswith("cuda") else torch.float32
        self.m = AutoModelForCausalLM.from_pretrained(ruta, dtype=dt, local_files_only=True).to(dispositivo).eval()
        self.si, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(PREFIJO_RR, add_special_tokens=False)
        self.suf = self.tok.encode(SUFIJO_RR, add_special_tokens=False)

    def recortar(self, texto):
        ids = self.tok.encode(texto, add_special_tokens=False)
        return texto if len(ids) <= MAX_TOK_TEXTO_RR else self.tok.decode(ids[:MAX_TOK_TEXTO_RR])

    def ids(self, r):
        texto = self.recortar(r["texto"])
        return [(self.pre + self.tok.encode(cuerpo_rr(r, texto, o), add_special_tokens=False)[:MAX_TOK_RR] + self.suf) for o in r["opciones"]]

    def puntuar(self, seqs, lote=64):
        """(logit de «yes», log p(yes) frente a «no») por secuencia."""
        torch = self.torch
        si, lsm = [], []
        with torch.inference_mode():
            for i in range(0, len(seqs), lote):
                b = self.tok.pad({"input_ids": seqs[i:i + lote]}, padding=True, return_tensors="pt").to(self.dev)
                lg = self.m(**b, logits_to_keep=1).logits[:, -1, :].float()   # solo la última posición: los logits de todas no caben (151k de vocabulario)
                si += lg[:, self.si].cpu().tolist()
                lsm += torch.log_softmax(torch.stack([lg[:, self.no], lg[:, self.si]], 1), 1)[:, 1].cpu().tolist()
        return si, lsm


def cmd_rerank(a):
    rr = Reranker(a.modelo, a.dispositivo)
    lat = {}
    todas = tareas(a.dev, a.max_filas)
    for t, (_, ev) in sorted(todas.items(), key=lambda kv: sum(len(r["opciones"]) for r in kv[1][1])):   # de la más barata a la más cara
        ms = []
        for r in ev[:N_LAT]:
            sincronizar(a.dispositivo)
            s = time.perf_counter()
            rr.puntuar(rr.ids(r), lote=len(r["opciones"]))
            sincronizar(a.dispositivo)
            ms.append(1000 * (time.perf_counter() - s))
        lat[t] = [round(x, 3) for x in ms]
        (Path(a.salida)).mkdir(parents=True, exist_ok=True)
        (Path(a.salida) / "lat-rerank.json").write_text(json.dumps(lat), encoding="utf-8")
        filas, s0 = [], time.perf_counter()
        for k in range(0, len(ev), 16):   # 16 filas por tanda; cada fila con todas sus opciones
            if a.hasta and time.time() > a.hasta:
                print("rerank", t, "corte por tiempo en", k, flush=True)
                break
            grupo = ev[k:k + 16]
            seqs, cortes = [], [0]
            for r in grupo:
                seqs += rr.ids(r)
                cortes.append(len(seqs))
            si, lsm = rr.puntuar(seqs, lote=a.lote)
            for j, r in enumerate(grupo):
                y, l = si[cortes[j]:cortes[j + 1]], lsm[cortes[j]:cortes[j + 1]]
                filas.append({"id": r["id"], "probs": dict(zip(r["opciones"], map(float, softmax(y)))), "yes": dict(zip(r["opciones"], y)),
                              "lsm": dict(zip(r["opciones"], l)), "valida": True})
        escribir(a.salida, t, filas)
        print("rerank", t, len(filas), "de", len(ev), "en %.1f s" % (time.perf_counter() - s0), flush=True)


# ---------------------------------------------------------------- medida
def leer_preds(d, t):
    from model.alt.lineas_base_v2 import leer
    f = Path(d) / (t + ".jsonl")
    return {r["id"]: r for r in leer(f)} if f.exists() else {}


def oks(dev, preds_por_tarea, esp=False):
    from model.alt.lineas_base_v2 import evaluar_tarea
    out = {}
    for t, (_, ev) in dev.items():
        e = esp and ev[0]["tipo"] == "score"
        out[t] = evaluar_tarea(ev, preds_por_tarea.get(t, {}), e)[0]
    return out


def cobertura(dev, preds_por_tarea):
    return {t: round(sum(1 for r in ev if r["id"] in preds_por_tarea.get(t, {})) / len(ev), 4) for t, (_, ev) in dev.items()}


def macro(o, tareas_=None):
    ts = tareas_ or sorted(o)
    return float(np.mean([o[t].mean() for t in ts]))


def ic_dif_macro(oa, ob, semilla=0, n=REMUESTREOS):
    """IC95 de macro(a) − macro(b): bootstrap pareado, filas remuestreadas dentro de cada tarea con los mismos índices para los dos."""
    rs = np.random.RandomState(semilla)
    ts = sorted(oa)
    d = np.zeros(n)
    for t in ts:
        m = len(oa[t])
        idx = rs.randint(0, m, (n, m))
        d += oa[t][idx].mean(1) - ob[t][idx].mean(1)
    d /= len(ts)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def por_tarea(o):
    return {t: round(float(v.mean()), 4) for t, v in sorted(o.items())}


def resumen(dev, nombre, o, cub):
    return {"sistema": nombre, "macro": round(macro(o), 4), "macro_sin_sinteticas": round(macro(o, [t for t in sorted(o) if t not in SINTETICAS]), 4),
            "por_tarea": por_tarea(o), "cobertura": cub, "cobertura_total": all(v == 1.0 for v in cub.values())}


def p50(lat):
    v = [x for xs in lat.values() for x in xs]
    return round(float(np.percentile(v, 50)), 2) if v else None


def restringir(preds, cands):
    """Las probabilidades de B en finanzas recortadas a las candidatas de la preselección del 4B (las claves de zs-topk), renormalizadas."""
    out = {}
    for i, p in preds.items():
        c = cands.get(i)
        if not c:
            continue
        y = [p["yes"][o] for o in c]
        out[i] = {"id": i, "probs": dict(zip(c, map(float, softmax(y)))), "valida": True}
    return out


def cmd_medir(a):
    from model.alt.lineas_base_v2 import cargar_dev
    crit_bytes = Path(a.crit).read_bytes()
    crit = json.loads(crit_bytes)
    dev = cargar_dev(a.dev)
    P, R = Path(a.pred), Path(a.ref)
    ts = sorted(dev)
    out = {"criterio_sha256": hashlib.sha256(crit_bytes).hexdigest(), "n_test": {t: len(dev[t][1]) for t in ts}}
    sis = {}

    def sistema(nombre, fuente, esp=False):
        pp = {t: fuente(t) for t in ts}
        sis[nombre] = (oks(dev, pp, esp), cobertura(dev, pp))
        return sis[nombre]

    ref4 = lambda t: leer_preds(R / ("zs-topk" if t == "finanzas-documentos" else "4b-zs"), t)   # noqa: E731
    sistema("4b-zs", ref4)
    sistema("l2-zs", lambda t: leer_preds(R / "l2-zs", t))
    sistema("knn-e5-k8-t3a", lambda t: leer_preds(R / "knn-e5-k8-t3a", t))
    for n in ("knn-e5-v32", "knn-e5-v8", "knn-qwen-v32", "knn-qwen-v8", "zs-e5", "zs-qwen"):
        sistema(n, lambda t, n=n: leer_preds(P / n, t))
    sistema("rerank", lambda t: leer_preds(P / "rerank", t), esp=True)
    sistema("rerank-lsm", lambda t: {i: dict(p, probs=dict(zip(p["lsm"], map(float, softmax(list(p["lsm"].values()))))))
                                     for i, p in leer_preds(P / "rerank", t).items()}, esp=True)
    sistema("rerank-mediana", lambda t: leer_preds(P / "rerank", t))
    cands = {i: sorted(p["probs"]) for i, p in leer_preds(R / "zs-topk", "finanzas-documentos").items()}
    sistema("rerank-topk-finanzas", lambda t: restringir(leer_preds(P / "rerank", t), cands) if t == "finanzas-documentos" else leer_preds(P / "rerank", t), esp=True)
    out["sistemas"] = {n: resumen(dev, n, o, c) for n, (o, c) in sis.items()}
    ok = lambda n: bool(sis[n][1]) and all(v == 1.0 for v in sis[n][1].values())   # noqa: E731
    if ok("knn-e5-v32") and ok("knn-e5-k8-t3a"):   # comprobación: el kNN-e5 de esta máquina reproduce T3a (acuerdo de aciertos fila a fila)
        out["reproduce_t3a"] = {t: round(float((sis["knn-e5-v32"][0][t] == sis["knn-e5-k8-t3a"][0][t]).mean()), 4) for t in ts}
    # A
    va = {"estado": "no medido", "motivo": "cobertura < 100 %"}
    if ok("knn-qwen-v32") and ok("knn-e5-v32"):
        d = macro(sis["knn-qwen-v32"][0]) - macro(sis["knn-e5-v32"][0])
        ic = ic_dif_macro(sis["knn-qwen-v32"][0], sis["knn-e5-v32"][0])
        va = {"dif_macro": round(d, 4), "ic95": ic, "estado": "PASA" if d >= 0.02 - 1e-9 and ic[0] > 0 else "NO PASA"}
        if ok("knn-qwen-v8") and ok("knn-e5-v8"):
            va["informativo_v8"] = {"dif_macro": round(macro(sis["knn-qwen-v8"][0]) - macro(sis["knn-e5-v8"][0]), 4),
                                    "ic95": ic_dif_macro(sis["knn-qwen-v8"][0], sis["knn-e5-v8"][0])}
        if ok("zs-qwen") and ok("zs-e5"):
            va["informativo_zs"] = {"dif_macro": round(macro(sis["zs-qwen"][0]) - macro(sis["zs-e5"][0]), 4), "ic95": ic_dif_macro(sis["zs-qwen"][0], sis["zs-e5"][0])}
    le = P / "vec-qwen" / "lat-emb.json"
    va["lat_emb_ms_p50"] = p50(json.loads(le.read_text())) if le.exists() else None
    # B
    vb = {"estado": "no medido", "motivo": "cobertura < 100 %"}
    lr = P / "rerank" / "lat-rerank.json"
    lat_b = p50(json.loads(lr.read_text())) if lr.exists() else None
    if ok("rerank") and ok("4b-zs"):
        d = macro(sis["rerank"][0]) - macro(sis["4b-zs"][0])
        ratio = round(crit["B"]["lat_ref_4b_ms_p50"] / lat_b, 2) if lat_b else None
        pasa = d > 0 or (d >= -0.02 - 1e-9 and ratio is not None and ratio >= 5)
        vb = {"dif_macro_vs_4b": round(d, 4), "ic95": ic_dif_macro(sis["rerank"][0], sis["4b-zs"][0]), "lat_ms_p50": lat_b,
              "lat_4b_ms_p50": crit["B"]["lat_ref_4b_ms_p50"], "veces_mas_rapido": ratio, "estado": "PASA" if pasa else "NO PASA",
              "dif_macro_vs_l2zs": round(macro(sis["rerank"][0]) - macro(sis["l2-zs"][0]), 4)}
        if ok("rerank-topk-finanzas"):
            vb["informativo_topk_finanzas"] = {"dif_macro_vs_4b": round(macro(sis["rerank-topk-finanzas"][0]) - macro(sis["4b-zs"][0]), 4)}
    else:
        vb["lat_ms_p50"] = lat_b
    out["veredicto"] = {"A": va, "B": vb, "C": {"estado": crit["C"]["estado"], "motivo": crit["C"]["motivo"]}}
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["veredicto"], ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ("vec-qwen", "op-e5", "knn", "zs-emb", "rerank", "medir"):
        p = sub.add_parser(nombre)
        p.add_argument("--dev", required=True)
        p.add_argument("--max-filas", type=int)
        if nombre in ("vec-qwen", "rerank"):
            p.add_argument("--modelo", required=True)
            p.add_argument("--dispositivo", default="cuda")
        if nombre in ("op-e5", "knn", "zs-emb"):
            p.add_argument("--vec", required=True)
        if nombre != "op-e5":
            p.add_argument("--salida", required=True)
        if nombre == "knn":
            p.add_argument("--vecinos", type=int, required=True)
        if nombre == "rerank":
            p.add_argument("--lote", type=int, default=64)
            p.add_argument("--hasta", type=float, default=0)
        if nombre == "medir":
            for k in ("--pred", "--ref", "--crit"):
                p.add_argument(k, required=True)
    a = ap.parse_args(argv)
    {"vec-qwen": cmd_vec_qwen, "op-e5": cmd_op_e5, "knn": cmd_knn, "zs-emb": cmd_zs_emb, "rerank": cmd_rerank, "medir": cmd_medir}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
