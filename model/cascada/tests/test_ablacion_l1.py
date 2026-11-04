import json
import tempfile
import unittest
from pathlib import Path

from model.cascada import ablacion_l1 as A


def casc(f1):
    return {"n": 100, "calidad": {"macro_f1": f1}, "rapida": {"macro_f1": f1 - 0.1}}


def resultados(con, sin, l2="L2b", l1="L1-zs", tarea="t1"):
    return {"tareas": {tarea: {"cascadas": {"L0 → %s → %s" % (l1, l2): casc(con), "L0 → %s" % l2: casc(sin), "L1-proto → %s" % l2: casc(0.9)}}}}


class Comparar(unittest.TestCase):
    def test_coste_y_criterio(self):
        tareas = {"a": {"cascadas": resultados(0.80, 0.80)["tareas"]["t1"]["cascadas"]},
                  "b": {"cascadas": resultados(0.703, 0.70)["tareas"]["t1"]["cascadas"]}}
        filas, ver = A.comparar(tareas)
        self.assertEqual(len(filas), 2)
        self.assertTrue(ver["L2b / L1-zs"]["cumple"])                        # 0,3 puntos ≤ 0,5
        tareas["b"]["cascadas"]["L0 → L1-zs → L2b"] = casc(0.71)           # 1 punto
        filas, ver = A.comparar(tareas)
        self.assertEqual(ver["L2b / L1-zs"]["incumplen"], ["b"])
        self.assertAlmostEqual(ver["L2b / L1-zs"]["peor"]["coste_de_quitar_l1"], 0.01)
        self.assertEqual(ver["L2b / L1-zs"]["regimen"], "sin datos")

    def test_regimenes_y_negativos(self):
        r = resultados(0.60, 0.70, l2="L2c-sop")["tareas"]
        filas, ver = A.comparar(r)
        self.assertTrue(ver["L2c-sop / L1-zs"]["cumple"])                    # quitar L1 MEJORA: coste negativo
        self.assertLess(filas[0]["coste_de_quitar_l1"], 0)
        self.assertEqual(ver["L2c-sop / L1-zs"]["regimen"], "con pocos datos (soporte)")

    def test_cli_y_estricto(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "r.json").write_text(json.dumps(resultados(0.80, 0.70)))
            self.assertEqual(A.main(["--resultados", str(d / "r.json"), "--salida", str(d / "s"), "--config-sin-l1"]), 0)
            self.assertEqual(A.main(["--resultados", str(d / "r.json"), "--salida", str(d / "s"), "--estricto"]), 1)
            (d / "resultados.json").write_text(json.dumps(resultados(0.70, 0.70)))
            self.assertEqual(A.main(["--resultados", str(d), "--salida", str(d / "s"), "--estricto"]), 0)
            self.assertIn("CUMPLE", (d / "s" / "ablacion_l1.md").read_text())
            c = json.loads((d / "s" / "config_sin_l1.json").read_text())
            self.assertEqual([x["nombre"] for x in c["capas"]], ["L0", "L2", "L3"])
            m = json.loads((d / "s" / "config_l1_mudo.json").read_text())
            self.assertEqual({x["nombre"]: x["tau"] for x in m["capas"]}["L1"], 1.0)

    def test_con_el_informe_real_de_2026_10_28(self):
        ruta = Path(__file__).resolve().parents[1] / "resultados" / "2026-10-28"
        filas, ver = A.comparar(A.cargar([ruta]))
        self.assertEqual(set(ver), {"%s / %s" % (a, b) for a in ("L2b", "L2b-sop", "L2c", "L2c-sop") for b in A.L1S})
        self.assertTrue(all(len(filas) >= 8 for _ in [0]))


if __name__ == "__main__":
    unittest.main()
