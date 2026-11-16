"""Genera el fixture sintético del kNN de L2 (solo biblioteca estándar): bancos .npy/.jsonl, consultas .npy/.jsonl y la decisión esperada,
calculada con una reimplementación literal de `model/alt/escalado.py:l2` (argsort estable, 32 vecinos, T 0,05, umbral de oro 0,90) y la
regla de escalado de `exportar_banco.py`. Uso: python3 generar.py   (escribe en el directorio del script). Semilla fija: no cambia."""
import json
import math
import random
import struct
from pathlib import Path

AQUI = Path(__file__).parent
DIM, K, T, S_ORO, U = 8, 32, 0.05, 0.90, 0.50
r = random.Random(7)


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def f32(v):
    return list(struct.unpack("<%df" % len(v), struct.pack("<%df" % len(v), *v)))


def vec(c=None, ruido=1.0):
    base = c if c is not None else [0.0] * DIM
    return f32(norm([b + ruido * r.gauss(0, 1) for b in base]))


def npy(ruta, filas):
    cab = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (len(filas), DIM)
    cab += " " * ((64 - (10 + len(cab) + 1) % 64) % 64) + "\n"
    with open(ruta, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(cab)) + cab.encode("latin1"))
        for v in filas:
            f.write(struct.pack("<%df" % DIM, *v))


def l2(q, ops, X, y, G):
    s = [sum(a * b for a, b in zip(q, x)) for x in X]
    so = max(s[:G]) if G else -1.0
    cand = list(range(G)) if (G and so >= S_ORO) else list(range(len(y)))
    k = sorted(cand, key=lambda j: (-s[j], j))[:K]
    p = dict.fromkeys(ops, 0.0)
    if k:
        mx = max(s[j] for j in k)
        for j in k:
            if y[j] in p:
                p[y[j]] += math.exp((s[j] - mx) / T)
    tot = sum(p.values())
    v = [p[o] / tot for o in ops] if tot > 0 else [1.0 / len(ops)] * len(ops)
    q2 = sorted(v, reverse=True)
    return v, q2[0] - (q2[1] if len(q2) > 1 else 0.0), so


# tres tareas: a (con oro; 8 oro + 40 fondo, así K=32 recorta), b (sin oro), c (con oro, «siempre»)
cent = {t: {o: f32(norm([r.gauss(0, 1) for _ in range(DIM)])) for o in ops} for t, ops in
        {"a": ["x", "y", "z"], "b": ["p", "q"], "c": ["m", "n"]}.items()}
OPS = {"a": ["x", "y", "z"], "b": ["p", "q"], "c": ["m", "n"]}
banco = {}
for t, nOro, nFondo in (("a", 8, 40), ("b", 0, 24), ("c", 6, 20)):
    X, y, oro = [], [], []
    for i in range(nOro + nFondo):
        o = OPS[t][i % len(OPS[t])] if i < nOro else r.choice(OPS[t])
        X.append(vec(cent[t][o], 0.6 if i < nOro else 1.2))
        y.append(o)
        oro.append(i < nOro)
    banco[t] = (X, y, nOro)
    npy(AQUI / ("banco-%s.npy" % t), X)
    with open(AQUI / ("banco-%s.jsonl" % t), "w") as f:
        for l, o in zip(y, oro):
            f.write(json.dumps({"etiqueta": l, "es_oro": o}) + "\n")
Q, filas = [], []
for t in "abc":
    X, y, G = banco[t]
    for i in range(14):
        # algunas consultas pegadas a un vector de oro (activa la regla de umbral) y otras sueltas
        q = f32(norm([a + 0.02 * r.gauss(0, 1) for a in X[r.randrange(G)]])) if (G and i % 3 == 0) else vec(cent[t][r.choice(OPS[t])], r.choice([0.5, 1.0, 2.0]))
        ops = OPS[t] if i % 4 else list(reversed(OPS[t]))
        v, m, so = l2(q, ops, X, y, G)
        motivo = "sin_oro" if G == 0 else ("siempre" if t == "c" else ("margen" if m < U else ""))
        Q.append(q)
        filas.append({"id": "%s%d" % (t, i), "tarea": t, "opciones": ops, "l2": ops[max(range(len(v)), key=lambda j: (v[j], -j))],
                      "l2_probs": v, "margen": m, "sim_oro": so, "escala": bool(motivo), "motivo": motivo})
npy(AQUI / "consultas.npy", Q)
with open(AQUI / "consultas.jsonl", "w") as f:
    for x in filas:
        f.write(json.dumps(x) + "\n")
(AQUI / "referencia.json").write_text(json.dumps({"u": U, "siempre": ["c"]}) + "\n")
print(sum(x["escala"] for x in filas), "de", len(filas), "escalan;", sum(1 for x in filas if x["motivo"] == "margen"), "por margen")
