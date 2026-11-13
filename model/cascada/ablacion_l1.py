"""Ablación de L1 en la cascada: ¿cuesta algo quitar L1? Criterio: quitar L1 no cuesta más de 0,5 puntos de macro-F1 en ninguna tarea.

    python -m model.cascada.ablacion_l1 --resultados DIR_O_JSON [DIR_O_JSON …] [--salida DIR] [--modo calidad] [--margen 0.005] [--estricto]
    python -m model.cascada.ablacion_l1 --rehacer --datos DIR --pred DIR [--l2b DIR --l2c DIR] [--fecha AAAA-MM-DD] --salida DIR

Cómo se evalúa hoy la cascada (el informe de referencia es `model/cascada/resultados/2026-10-28/informe.md`): `model.cascada.capas` deja las
predicciones por capa (`L0/`, `L1-zs/`, `L1-proto/`) y `model.cascada.informe` combina esas predicciones con las de L2 (`pred/` = sin
soporte, `pred_soporte/` = con soporte) en cascadas «L0 → L1-x → L2» y «L0 → L2», con τ por capa elegido en validación cruzada a 2
pliegues sobre test (`model.cascada.cascada.validar`). Ahí «sin datos» es la variante de L2 sin soporte (`L2b`, `L2c`) y «con pocos datos» la
que usa soporte (`L2b-sop`, `L2c-sop`). Este script solo compara dos entradas del mismo `resultados.json`, así que no cambia el método:

    coste de quitar L1  =  macro-F1(L0 → L1-x → L2v)  −  macro-F1(L0 → L2v)      (>0: L1 ayudaba; ≤ margen en TODAS las tareas = criterio)

para cada tarea, cada L1 (`L1-zs`, la del servidor, y `L1-proto`, la candidata con prototipos), cada variante de L2 y cada punto de
operación (`calidad` por defecto; `rapida` también se imprime). `--rehacer` vuelve a calcular el `resultados.json` llamando a
`model.cascada.informe` con esos mismos argumentos (necesita las predicciones; en AWS las deja `aws/arranque.sh`).

Quitar L1 del servidor se hace por configuración, no por el `.chispa`: el `.chispa` de `model/l2/exportar.py` solo lleva META, L2,
ESQUEMAS y CALIB (sin L1 ni SOPL1); las secciones L1 y SOPL1 las escribe `server/deploy/mkmodelo` y `chispa train`, fuera de `model/`.
`--config-sin-l1` escribe `config_sin_l1.json` (capas L0 → L2 → L3 con los τ por defecto) y `config_l1_mudo.json` (L1 con τ = 1,0:
solo contesta con confianza exactamente 1) para repetir la medida de extremo a extremo con `chispa-server -config`.
"""
import argparse
import json
import sys
from pathlib import Path

MARGEN = 0.005        # 0,5 puntos de macro-F1
MODOS = ("calidad", "rapida")
L1S = ("L1-zs", "L1-proto")
CAPAS_DEFECTO = [("L0", 0.80), ("L1", 0.80), ("L2", 0.70), ("L3", 0.60)]   # server/decide/config.go: ConfigPorDefecto


def cargar(rutas):
    """Une los `resultados.json` (o directorios que lo contengan): {tarea: {cascadas…}} con la primera aparición de cada cascada."""
    tareas = {}
    for r in rutas:
        p = Path(r)
        p = p / "resultados.json" if p.is_dir() else p
        for tarea, t in json.load(open(p, encoding="utf-8"))["tareas"].items():
            tareas.setdefault(tarea, {"cascadas": {}})["cascadas"].update({k: v for k, v in t["cascadas"].items()
                                                                           if k not in tareas[tarea]["cascadas"]})
    return tareas


def variantes_l2(tareas):
    """Variantes de L2 que aparecen tras «L0 → » en las cascadas de dos capas: «L2b», «L2b-sop», …"""
    out = []
    for t in tareas.values():
        for n in t["cascadas"]:
            partes = n.split(" → ")
            if len(partes) == 2 and partes[0] == "L0" and partes[1].startswith("L2") and partes[1] not in out:
                out.append(partes[1])
    return out


def regimen(v):
    return "con pocos datos (soporte)" if v.endswith("-sop") else "sin datos"


def comparar(tareas, modo="calidad", margen=MARGEN):
    """-> lista de filas {tarea, l2, regimen, l1, con, sin, coste, cumple} y el veredicto por (l2, l1)."""
    filas = []
    for l2 in variantes_l2(tareas):
        for l1 in L1S:
            for tarea, t in sorted(tareas.items()):
                con, sin = t["cascadas"].get("L0 → %s → %s" % (l1, l2)), t["cascadas"].get("L0 → %s" % l2)
                if not con or not sin:
                    continue
                a, b = con[modo]["macro_f1"], sin[modo]["macro_f1"]
                filas.append({"tarea": tarea, "l2": l2, "regimen": regimen(l2), "l1": l1, "con_l1": a, "sin_l1": b, "coste_de_quitar_l1": a - b,
                              "cumple": bool(a - b <= margen + 1e-12), "n": con["n"]})
    veredicto = {}
    for f in filas:
        v = veredicto.setdefault("%s / %s" % (f["l2"], f["l1"]), {"regimen": f["regimen"], "tareas": 0, "incumplen": [], "peor": None})
        v["tareas"] += 1
        if not f["cumple"]:
            v["incumplen"].append(f["tarea"])
        if v["peor"] is None or f["coste_de_quitar_l1"] > v["peor"]["coste_de_quitar_l1"]:
            v["peor"] = {"tarea": f["tarea"], "coste_de_quitar_l1": f["coste_de_quitar_l1"]}
    for v in veredicto.values():
        v["cumple"] = not v["incumplen"]
    return filas, veredicto


def pts(x):
    return ("%+.1f" % (100 * x)).replace(".", ",")


def informe_md(res, margen):
    L = ["# Ablación de L1: coste de quitarla de la cascada", "",
         "Coste = macro-F1 con L1 − macro-F1 sin L1, en puntos. Criterio: ≤ %s puntos en TODAS las tareas." % pts(margen).lstrip("+"), ""]
    for modo, (filas, ver) in res.items():
        L += ["## Punto de operación «%s»" % modo, "", "| L2 / L1 | régimen | tareas | peor tarea | peor coste (pts) | incumplen | criterio |", "|---|---|---|---|---|---|---|"]
        for k, v in ver.items():
            L.append("| %s | %s | %d | %s | %s | %s | %s |" % (k, v["regimen"], v["tareas"], v["peor"]["tarea"], pts(v["peor"]["coste_de_quitar_l1"]),
                                                           ", ".join(v["incumplen"]) or "—", "CUMPLE" if v["cumple"] else "NO CUMPLE"))
        L += ["", "| tarea | L2 | L1 | con L1 | sin L1 | coste (pts) |", "|---|---|---|---|---|---|"]
        for f in filas:
            L.append("| %s | %s | %s | %.3f | %.3f | %s |" % (f["tarea"], f["l2"], f["l1"], f["con_l1"], f["sin_l1"], pts(f["coste_de_quitar_l1"])))
        L.append("")
    return "\n".join(L)


def configs_sin_l1(salida):
    """Escribe las dos configuraciones del servidor para medir de extremo a extremo sin L1 (ver el docstring)."""
    sal = Path(salida)
    sal.mkdir(parents=True, exist_ok=True)
    sin = {"capas": [{"nombre": n, "tau": t} for n, t in CAPAS_DEFECTO if n != "L1"]}
    mudo = {"capas": [{"nombre": n, "tau": 1.0 if n == "L1" else t} for n, t in CAPAS_DEFECTO]}
    (sal / "config_sin_l1.json").write_text(json.dumps(sin, indent=1) + "\n")
    (sal / "config_l1_mudo.json").write_text(json.dumps(mudo, indent=1) + "\n")
    return sin, mudo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resultados", nargs="*", default=[], help="resultados.json de model.cascada.informe (o su directorio); varios se unen")
    ap.add_argument("--rehacer", action="store_true", help="recalcula primero con model.cascada.informe (usa --datos, --pred, --l2b, --l2c, --fecha, --tareas)")
    ap.add_argument("--datos")
    ap.add_argument("--pred")
    ap.add_argument("--l2b")
    ap.add_argument("--l2c")
    ap.add_argument("--tareas")
    ap.add_argument("--fecha")
    ap.add_argument("--base")
    ap.add_argument("--salida", default=".")
    ap.add_argument("--modo", choices=MODOS, default="calidad", help="punto de operación que decide el criterio")
    ap.add_argument("--margen", type=float, default=MARGEN)
    ap.add_argument("--estricto", action="store_true", help="código de salida 1 si el criterio no se cumple (en el modo elegido)")
    ap.add_argument("--config-sin-l1", action="store_true", help="escribe también las configuraciones del servidor sin L1")
    a = ap.parse_args(argv)
    rutas = list(a.resultados)
    if a.rehacer:
        if not a.datos:
            ap.error("--rehacer necesita --datos (y --pred y/o --l2b/--l2c)")
        from . import informe as I
        arg = ["--datos", a.datos, "--salida", str(Path(a.salida) / "informe")]
        for flag, v in (("--pred", a.pred), ("--l2b", a.l2b), ("--l2c", a.l2c), ("--tareas", a.tareas), ("--fecha", a.fecha), ("--base", a.base)):
            if v:
                arg += [flag, v]
        I.main(arg)
        rutas.append(str(Path(a.salida) / "informe" / (a.fecha or __import__("datetime").date.today().isoformat()) / "resultados.json"))
    if not rutas:
        ap.error("falta --resultados o --rehacer")
    tareas = cargar(rutas)
    res = {m: comparar(tareas, m, a.margen) for m in MODOS}
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "ablacion_l1.md").write_text(informe_md(res, a.margen), encoding="utf-8")
    (sal / "ablacion_l1.json").write_text(json.dumps({m: {"filas": f, "veredicto": v, "margen": a.margen} for m, (f, v) in res.items()},
                                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.config_sin_l1:
        configs_sin_l1(sal)
    filas, ver = res[a.modo]
    for k, v in ver.items():
        print("%-28s %-26s peor %s %s pts  %s" % (k, v["regimen"], v["peor"]["tarea"], pts(v["peor"]["coste_de_quitar_l1"]), "CUMPLE" if v["cumple"] else "NO CUMPLE (%s)" % ", ".join(v["incumplen"])))
    if not ver:
        print("sin cascadas «L0 → L1 → L2» y «L0 → L2» en los resultados: nada que comparar")
        return 2 if a.estricto else 0
    ok = all(v["cumple"] for v in ver.values())
    print("criterio «quitar L1 no cuesta más de %s puntos en ninguna tarea» (%s): %s" % (pts(a.margen).lstrip("+"), a.modo, "CUMPLE" if ok else "NO CUMPLE"))
    return 0 if ok or not a.estricto else 1


if __name__ == "__main__":
    sys.exit(main())
