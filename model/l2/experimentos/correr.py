"""Experimentos fuera de línea sobre L2 (E2 cabezas de score, E3 ruido de condicionamiento): ajuste fino desde M1 con solo tsjev y JevBench k=0.

    python -m model.l2.experimentos.correr --ckpt M1.pt --tsjev DIR --jevbench RUTA --salida DIR_SAL \\
        --pasos N --semilla S --variante {control,acumulados,grueso_fino,ruido} [--ruido-p 0.15] [--device cpu|cuda]
    python -m model.l2.experimentos.correr ... --solo-tiempo        # 20 pasos y extrapola a --pasos

Variantes (solo cambia lo indicado; todo lo demás es idéntico):
  control      cabeza `coral1` (el escalar de siempre), sin ruido;
  coral_vec    `--cabeza-score coral`: CORAL sobre el vector de las K similitudes (una por texto de torre de nivel);
  acumulados   `--cabeza-score acumulados` sobre el vector de K similitudes (el checkpoint coral1 se convierte sin cambiar la función);
  grueso_fino  `--cabeza-score grueso_fino` (la cabeza de score arranca con su init);
  ruido        `coral1` y `--ruido-cond P` (`--ruido-p`, 0,15 por defecto).
Datos y parámetros como el ajuste 1 (`aws/ajuste.sh` de 80ab8ef) donde aplican: partida de M1 con `--desde` (lr 3e-5 y torre 1e-5),
`--max-tok 256`, lote 64, `--tope-tarea 2000 --alfa 0.3 --excluir-fuera` sobre tsjev tratado como meta (mismo tope por tarea y mismas
tareas «fuera»), sin otro meta, sin aumento, sin etiquetas blandas, sin episodios (k=0). El orden de los lotes es una función pura de
(semilla, paso), así que las cuatro variantes de una semilla ven exactamente los mismos lotes; el ruido tiene su propio generador.
Se evalúa el checkpoint final (`ultimo.pt`, no el «mejor» del dev: los pesos del último paso son comparables entre variantes).
Salida en `DIR_SAL`: `entrenamiento/` (checkpoints y log), `jevbench/pred/l2_k0.jsonl` (+ `jevbench_filas.jsonl`) y `correr.json`.
"""
import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

VARIANTES = ("control", "coral_vec", "acumulados", "grueso_fino", "ruido")
RUIDO_P = 0.15
PASOS_TIEMPO = 20


def args_entrenar(a, salida, pasos=None):
    """argv de `model.l2.entrenar` para la variante de `a` (el mismo para todas salvo la cabeza o el ruido)."""
    arg = ["--salida", str(salida), "--meta", str(a.tsjev), "--desde", str(a.ckpt),
           "--tope-tarea", "2000", "--alfa", "0.3", "--max-tok", "256", "--excluir-fuera", "--max-filas", "300000",
           "--lote", str(a.lote), "--pasos", str(pasos or a.pasos), "--semilla", str(a.semilla), "--device", a.device,
           "--dev-max", str(a.dev_max), "--eval-cada", str(pasos or a.pasos), "--ckpt-cada", "100000000"]
    arg += ["--cabeza-score", {"coral_vec": "coral"}.get(a.variante, a.variante) if a.variante in ("coral_vec", "acumulados", "grueso_fino") else "coral1"]
    if a.variante == "ruido":
        arg += ["--ruido-cond", str(a.ruido_p)]
    return arg


def segundos_por_paso(log):
    """Último «X s/paso» (media desde el primer paso) del log de entrenar; None si no hay."""
    m = re.findall(r"([0-9.]+) s/paso", log)
    return float(m[-1]) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="M1: el mejor.pt del ajuste 1")
    ap.add_argument("--tsjev", required=True, help="directorio con el train de tsjev ya convertido (model.meta.convertir)")
    ap.add_argument("--jevbench", help="clon del repo, datasets/public o un .jsonl (obligatorio salvo con --solo-tiempo)")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--pasos", type=int, required=True)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--variante", choices=VARIANTES, required=True)
    ap.add_argument("--ruido-p", type=float, default=RUIDO_P, help="P de la variante ruido (por defecto %g)" % RUIDO_P)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--dev-max", type=int, default=500, help="filas de dev (solo para el log; no elige el checkpoint)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--solo-tiempo", action="store_true", help="entrena %d pasos, mide s/paso y extrapola a --pasos (no evalúa JevBench)" % PASOS_TIEMPO)
    a = ap.parse_args(argv)
    if not a.solo_tiempo and not a.jevbench:
        ap.error("falta --jevbench")
    from model.l2 import entrenar
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.solo_tiempo:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = entrenar.main(args_entrenar(a, sal / "tiempo", PASOS_TIEMPO))
        sp = segundos_por_paso(buf.getvalue())
        res = {"variante": a.variante, "pasos_medidos": PASOS_TIEMPO, "s_por_paso": sp, "pasos": a.pasos,
               "entrenamiento_estimado_s": None if sp is None else sp * a.pasos, "total_medido_s": time.time() - t0}
        (sal / "tiempo.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        if sp is not None:
            print("%d pasos × %.2f s/paso ≈ %.1f min de entrenamiento (sin la carga de datos ni JevBench)" % (a.pasos, sp, sp * a.pasos / 60))
        return rc
    rc = entrenar.main(args_entrenar(a, sal / "entrenamiento"))
    if rc:
        return rc
    t1 = time.time()
    from model.evaluacion import jevbench_l2
    rc = jevbench_l2.main(["--ckpt", str(sal / "entrenamiento" / "ultimo.pt"), "--jevbench", a.jevbench, "--salida", str(sal / "jevbench"),
                           "--k", "0", "--device", a.device])
    (sal / "correr.json").write_text(json.dumps({"variante": a.variante, "semilla": a.semilla, "pasos": a.pasos, "ruido_p": a.ruido_p if a.variante == "ruido" else 0.0,
                                                 "ckpt": a.ckpt, "pred": str(sal / "jevbench" / "pred" / "l2_k0.jsonl"),
                                                 "entrenar_s": t1 - t0, "evaluar_s": time.time() - t1}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
