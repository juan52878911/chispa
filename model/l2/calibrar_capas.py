"""Calibración por capa, partición por hash y curvas riesgo–cobertura (solo numpy: corre en el Mac sin torch).

Piezas que usa `model.l2.calibrar` (L2) y que sirven igual para L0 y L3, que no pasan por MaxSim:

- `particion_hash`: reparte las filas en «ajuste» y «τ» por hash de (tarea, texto), con sal fija. Las filas que ajustan T o
  Platt nunca eligen τ; un duplicado cae siempre en el mismo lado.
- `curva_riesgo_cobertura` y `aurc`: riesgo selectivo (1 − precisión de lo aceptado) frente a cobertura al bajar el umbral.
- `platt` e `isotonica`: calibran la confianza propia de una capa (P(acierto | conf)).
- CLI: calibra una o varias capas a partir de sus predicciones (`{id, probs}`) y las etiquetas.

Fuentes de la confianza de cada capa (código del servidor, solo lectura):
- L0: probabilidad máxima de la regresión logística sobre TF-IDF (`pred.Prob` en `server/decide/l0.go`); en `model.cascada.capas`
  son los `probs` de `model.lineas_base.correr.correr_l0`.
- L1: softmax(coseno / T) (choice) o σ(a·s + b) (noul) en `server/decide/l1.go`; score no opina.
- L3: probabilidades que el LLM declara en su JSON (`interpretarL3`, `server/decide/l3.go`): autodeclaradas, sin calibrar.

    python -m model.l2.calibrar_capas --capa L0=pred/L0/massive.jsonl --capa L3=pred/L3/massive.jsonl \
        --etiquetas test.jsonl --salida calib_capas.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SAL = "tau-fuera-de-pliegue"
PRECISION = 0.95
MIN_AJUSTE = 20


# ---------------------------------------------------------------- partición
def lado_hash(fila, frac_tau=0.5, sal=SAL):
    """«tau» o «ajuste» para una fila, por hash de (sal, tarea, texto). Determinista y sin estado."""
    h = hashlib.sha256(("%s\x1f%s\x1f%s" % (sal, fila.get("tarea", ""), fila.get("texto", fila.get("id", "")))).encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / 2.0 ** 64
    return "tau" if u < frac_tau else "ajuste"


def particion_hash(filas, frac_tau=0.5, sal=SAL):
    """-> (idx_ajuste, idx_tau): índices de `filas`. Disjuntos y con unión = todas."""
    aj, ta = [], []
    for i, r in enumerate(filas):
        (ta if lado_hash(r, frac_tau, sal) == "tau" else aj).append(i)
    return aj, ta


# ---------------------------------------------------------------- riesgo–cobertura
def curva_riesgo_cobertura(conf, ok):
    """Ordena por confianza descendente (los empates se aceptan juntos) y devuelve (cobertura, riesgo) en cada corte posible."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    n = len(conf)
    if n == 0:
        return np.array([]), np.array([])
    o = np.argsort(-conf, kind="stable")
    c, a = conf[o], ok[o]
    acum = np.cumsum(a)
    k = np.arange(1, n + 1)
    corte = np.append(c[1:] < c[:-1], True)  # solo entre valores distintos
    return k[corte] / n, 1.0 - acum[corte] / k[corte]


def aurc(conf, ok):
    """Área bajo la curva riesgo–cobertura, integrada en cobertura (regla trapezoidal desde el primer corte). Menor es mejor;
    una confianza sin información da aproximadamente el riesgo total. nan sin filas."""
    cob, rie = curva_riesgo_cobertura(conf, ok)
    if len(cob) == 0:
        return float("nan")
    if len(cob) == 1:
        return float(rie[0])
    x = np.concatenate([[0.0], cob])
    y = np.concatenate([[rie[0]], rie])
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def resumen_rc(conf, ok, puntos=10):
    """Resumen JSON de la curva: AURC, riesgo total y el riesgo en `puntos` coberturas regulares."""
    cob, rie = curva_riesgo_cobertura(conf, ok)
    if len(cob) == 0:
        return {"n": 0, "aurc": float("nan"), "riesgo": float("nan"), "curva": []}
    curva = []
    for q in np.linspace(1 / puntos, 1, puntos):
        j = int(np.searchsorted(cob, q - 1e-12))
        j = min(j, len(cob) - 1)
        curva.append({"cobertura": float(cob[j]), "riesgo": float(rie[j])})
    return {"n": int(len(np.asarray(ok))), "aurc": aurc(conf, ok),
            "riesgo": float(1 - np.mean(np.asarray(ok, float))), "curva": curva}


# ---------------------------------------------------------------- calibradores de confianza
def platt(conf, ok, iters=100):
    """P(acierto | conf) = σ(a·logit(conf) + b), por Newton con una pequeña ridge. -> (a, b). Con una sola clase, (0, ±8)."""
    c = np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6)
    x = np.log(c / (1 - c))
    y = np.asarray(ok, float)
    if y.min() == y.max():
        return 0.0, (8.0 if y[0] else -8.0)
    w = np.array([1.0, 0.0])
    X = np.stack([x, np.ones_like(x)], 1)
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + 1e-6 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        paso = np.linalg.solve(H, g)
        w -= paso
        if np.abs(paso).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def aplicar_platt(conf, a, b):
    c = np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.clip(a * np.log(c / (1 - c)) + b, -30, 30)))


def isotonica(conf, ok):
    """Regresión isotónica creciente (PAV) de ok sobre conf. -> (xs, ys): nudos ordenados; se aplica con `aplicar_isotonica`."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    if len(conf) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    o = np.argsort(conf, kind="stable")
    x, y = conf[o], ok[o]
    bloques = []  # [suma, peso, x_min, x_max]
    for xi, yi in zip(x, y):
        bloques.append([yi, 1.0, xi, xi])
        while len(bloques) > 1 and bloques[-2][0] / bloques[-2][1] >= bloques[-1][0] / bloques[-1][1]:
            b2 = bloques.pop()
            b1 = bloques[-1]
            b1[0] += b2[0]
            b1[1] += b2[1]
            b1[3] = b2[3]
    xs, ys = [], []
    for s, p, lo, hi in bloques:
        xs += [lo, hi]
        ys += [s / p, s / p]
    return np.array(xs), np.array(ys)


def aplicar_isotonica(conf, xs, ys):
    return np.interp(np.asarray(conf, float), xs, ys)


def ajustar_confianza(metodo, conf, ok):
    """-> (función conf→conf calibrada, parámetros JSON). metodo: «platt» o «isotonica»."""
    if metodo == "platt":
        a, b = platt(conf, ok)
        return (lambda c: aplicar_platt(c, a, b)), {"metodo": "platt", "a": a, "b": b}
    if metodo == "isotonica":
        xs, ys = isotonica(conf, ok)
        return (lambda c: aplicar_isotonica(c, xs, ys)), {"metodo": "isotonica", "xs": [float(v) for v in xs], "ys": [float(v) for v in ys]}
    raise ValueError("metodo: platt o isotonica, no %r" % metodo)


def tau_precision(conf, ok, precision=PRECISION):
    """(τ, cobertura, precisión): el menor τ cuyas filas con conf ≥ τ aciertan ≥ `precision` (misma regla que `calibrar.tau_escalate`)."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    if len(conf) == 0:
        return 1.0, 0.0, float("nan")
    o = np.argsort(-conf, kind="stable")
    c, a = conf[o], ok[o]
    prec = np.cumsum(a) / np.arange(1, len(c) + 1)
    ultimo = np.flatnonzero((np.append(c[1:], -1.0) < c) & (prec >= precision))
    if len(ultimo) == 0:
        return 1.0, 0.0, float("nan")
    k = int(ultimo[-1])
    return float(c[k]), (k + 1) / len(c), float(prec[k])


def ece(conf, ok, cubos=10):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    if len(conf) == 0:
        return float("nan")
    idx = np.clip(np.ceil(conf * cubos - 1e-12).astype(int) - 1, 0, cubos - 1)
    return float(sum((idx == k).sum() / len(conf) * abs(ok[idx == k].mean() - conf[idx == k].mean()) for k in range(cubos) if (idx == k).any()))


def calibrar_capa(conf, ok, claves, metodo="isotonica", frac_tau=0.5, precision=PRECISION, sal=SAL):
    """Calibra la confianza de UNA capa con las filas de «ajuste» y elige τ con las de «τ» (sobre la confianza ya calibrada).
    `claves`: un dict por fila con `tarea`/`texto` o `id` para el hash. Sin ajuste suficiente no se calibra: se conserva la
    confianza cruda y se avisa en `ajustado=False`."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, bool)
    aj, ta = particion_hash(claves, frac_tau, sal)
    aj, ta = np.array(aj, int), np.array(ta, int)
    ajustado = len(aj) >= MIN_AJUSTE and len(set(ok[aj].tolist())) == 2
    f, par = ajustar_confianza(metodo, conf[aj], ok[aj]) if ajustado else ((lambda c: np.asarray(c, float)), {"metodo": "ninguno"})
    cal = f(conf[ta]) if len(ta) else np.array([])
    tau, cob, prec = tau_precision(cal, ok[ta], precision)
    tau_crudo, cob_crudo, prec_crudo = tau_precision(conf[ta], ok[ta], precision)
    return {
        "n": int(len(conf)), "n_ajuste": int(len(aj)), "n_tau": int(len(ta)), "ajustado": bool(ajustado), "calibrador": par,
        "exactitud_tau": float(np.mean(ok[ta])) if len(ta) else float("nan"),
        "ece_crudo": ece(conf[ta], ok[ta]), "ece_calibrado": ece(cal, ok[ta]),
        "tau": tau, "cobertura": cob, "precision_cubierta": prec,
        "tau_crudo": tau_crudo, "cobertura_cruda": cob_crudo, "precision_cruda": prec_crudo,
        "rc_crudo": resumen_rc(conf[ta], ok[ta]), "rc_calibrado": resumen_rc(cal, ok[ta]),
    }


# ---------------------------------------------------------------- CLI
def _leer_jsonl(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def conf_ok(preds, etiquetas):
    """Por id común: (claves, conf, ok). conf = prob. máxima (primer máximo); filas con `sin_opinion` o sin probs se omiten."""
    claves, conf, ok = [], [], []
    for r in etiquetas:
        p = preds.get(r["id"])
        if not p or p.get("sin_opinion") or not p.get("probs"):
            continue
        c, v = max(p["probs"].items(), key=lambda kv: kv[1])
        claves.append({"id": r["id"], "tarea": r.get("tarea", ""), "texto": r.get("texto", r["id"])})
        conf.append(float(v))
        ok.append(str(c) == str(r["etiqueta"]))
    return claves, conf, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capa", action="append", required=True, help="NOMBRE=pred.jsonl (filas {id, probs}); repetible")
    ap.add_argument("--etiquetas", required=True, help="jsonl con id, etiqueta y, si se tiene, tarea y texto (para el hash)")
    ap.add_argument("--metodo", choices=("isotonica", "platt"), default="isotonica")
    ap.add_argument("--frac-tau", type=float, default=0.5)
    ap.add_argument("--precision", type=float, default=PRECISION)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args(argv)
    etiquetas = _leer_jsonl(a.etiquetas)
    out = {}
    for spec in a.capa:
        nombre, _, ruta = spec.partition("=")
        preds = {r["id"]: r for r in _leer_jsonl(ruta)}
        claves, conf, ok = conf_ok(preds, etiquetas)
        out[nombre] = calibrar_capa(conf, ok, claves, a.metodo, a.frac_tau, a.precision)
        r = out[nombre]
        print("%s: n=%d (ajuste %d, τ %d)  ECE %.3f→%.3f  τ %.3f cobertura %.3f  AURC %.4f→%.4f" % (
            nombre, r["n"], r["n_ajuste"], r["n_tau"], r["ece_crudo"], r["ece_calibrado"], r["tau"], r["cobertura"],
            r["rc_crudo"]["aurc"], r["rc_calibrado"]["aurc"]))
    Path(a.salida).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
