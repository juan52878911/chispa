"""Reproduce la comparación «τ en validación vs τ en test» con el procedimiento antiguo y el nuevo (fuera de pliegue).

    python -m model.l2.tau_val_vs_test --ckpt DIR --dev dev.jsonl --test test.jsonl --salida tau_val_test.json [--no-vistas a,b]

Para cada procedimiento calibra con las filas de dev (`calibrar.calibrar`) y mide, sobre test, qué cobertura y precisión da el τ
elegido en dev:
- antiguo: T/Platt y τ con las MISMAS filas de dev (el τ sale optimista: 0,85 en validación frente a 0,777 en test);
- nuevo: T/Platt con un lado del hash de dev y τ con el otro.
`tau_test` es el τ que habría dado precisión 0,95 en el propio test (referencia, no usable). Un buen procedimiento deja
`tau_dev` cerca de `tau_test` y la precisión cubierta en test cerca de 0,95.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import calibrar as C
from . import datos as D


def comparar(modelo, enc, dev, test, no_vistas=(), frac_tau=0.5):
    """-> {procedimiento: {tau_dev_global, tareas: {tarea: {tau_dev, tau_test, precision_test, cobertura_test, …}}, medias}}."""
    out = {}
    for nombre, fuera in (("antiguo_mismas_filas", False), ("nuevo_fuera_de_pliegue", True)):
        inf = C.calibrar(modelo, enc, dev, tau_fuera_pliegue=fuera, frac_tau=frac_tau, tareas_no_vistas=no_vistas)
        ev = C.evaluar_tau(modelo, enc, test, inf)
        tareas = {t: {"tau_dev": inf["tareas"][t]["tau"], "cobertura_dev": inf["tareas"][t]["cobertura"], **e,
                      "dif_tau": e["tau"] - e["tau_test"]} for t, e in ev.items()}
        con_tau = [r for r in tareas.values() if r["tau_dev"] < 1.0 and not np.isnan(r["precision_cubierta"])]
        out[nombre] = {
            "tau_dev_global": inf["tau_escalate"], "tau_fuera_pliegue": inf["tau_fuera_pliegue"],
            "aurc_dev": inf["global"]["aurc"], "tareas": tareas,
            "media": {"tau_dev": float(np.mean([r["tau_dev"] for r in con_tau])) if con_tau else float("nan"),
                      "tau_test": float(np.mean([r["tau_test"] for r in con_tau])) if con_tau else float("nan"),
                      "precision_test": float(np.mean([r["precision_cubierta"] for r in con_tau])) if con_tau else float("nan"),
                      "cobertura_test": float(np.mean([r["cobertura"] for r in con_tau])) if con_tau else float("nan"),
                      "tareas_con_tau": len(con_tau)}}
    return out


def tabla(res):
    L = ["| procedimiento | τ dev (media) | τ test (media) | precisión en test | cobertura en test | tareas |", "|---|---|---|---|---|---|"]
    for n, r in res.items():
        m = r["media"]
        L.append("| %s | %.3f | %.3f | %.3f | %.3f | %d |" % (n, m["tau_dev"], m["tau_test"], m["precision_test"], m["cobertura_test"], m["tareas_con_tau"]))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dev", nargs="+", required=True, help="jsonl con filas split=dev")
    ap.add_argument("--test", nargs="+", required=True, help="jsonl con filas split=test (se filtra por split si lo traen)")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-vistas", default="")
    ap.add_argument("--frac-tau", type=float, default=0.5)
    ap.add_argument("--max-filas", type=int, default=0)
    a = ap.parse_args(argv)
    from .modelo import Codificador
    modelo, _ = C.cargar_modelo(a.ckpt, a.device)
    dev = C.filas_dev(a.dev, estricto=True)
    test = D.leer(a.test)
    test = [r for r in test if r.get("split", "test") == "test"]
    if a.max_filas:
        dev, test = dev[:a.max_filas], test[:a.max_filas]
    res = comparar(modelo, Codificador(device=a.device), dev, test, [t for t in a.no_vistas.split(",") if t], a.frac_tau)
    Path(a.salida).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(tabla(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
