"""Partición por hash, riesgo–cobertura, calibradores de confianza por capa y τ fuera de pliegue."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from model.l2 import calibrar_capas as CC


def claves(n, tarea="t"):
    return [{"tarea": tarea, "texto": "mensaje %d" % i} for i in range(n)]


def sintetico(n=3000, semilla=0, sobreconfianza=True):
    """Confianza cruda sobreconfiada: la probabilidad real de acierto es conf**3."""
    r = np.random.default_rng(semilla)
    c = r.uniform(0.5, 1.0, n)
    return c, r.uniform(size=n) < (c ** 3 if sobreconfianza else c)


class Particion(unittest.TestCase):
    def test_disjunta_completa_y_determinista(self):
        f = claves(500)
        aj, ta = CC.particion_hash(f)
        self.assertEqual(sorted(aj + ta), list(range(500)))
        self.assertFalse(set(aj) & set(ta))
        self.assertEqual((aj, ta), CC.particion_hash(f))
        self.assertTrue(180 < len(ta) < 320)
        self.assertNotEqual(ta, CC.particion_hash(f, sal="otra")[1])

    def test_duplicados_en_el_mismo_lado_y_tarea_cuenta(self):
        f = claves(50) + claves(50)           # cada texto dos veces
        aj, ta = CC.particion_hash(f)
        for i in range(50):
            self.assertEqual(i in ta, i + 50 in ta)
        self.assertNotEqual(CC.lado_hash({"tarea": "a", "texto": "x"}), None)


class RiesgoCobertura(unittest.TestCase):
    def test_curva_y_aurc_a_mano(self):
        cob, rie = CC.curva_riesgo_cobertura([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1])
        self.assertEqual(list(cob), [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(rie, [0, 0, 1 / 3, 0.25])
        self.assertTrue(np.isnan(CC.aurc([], [])))

    def test_empates_se_aceptan_juntos(self):
        cob, rie = CC.curva_riesgo_cobertura([0.9, 0.9, 0.5], [1, 0, 1])
        self.assertEqual(list(cob), [2 / 3, 1.0])
        np.testing.assert_allclose(rie, [0.5, 1 / 3])

    def test_buena_confianza_baja_el_aurc(self):
        c, ok = sintetico()
        azar = np.random.default_rng(1).uniform(size=len(c))
        self.assertLess(CC.aurc(c, ok), CC.aurc(azar, ok))
        self.assertLess(CC.aurc(ok.astype(float) + 0.01 * c, ok), CC.aurc(c, ok))     # el oráculo gana
        r = CC.resumen_rc(c, ok)
        self.assertEqual(r["n"], len(c))
        self.assertEqual(len(r["curva"]), 10)
        json.dumps(r)


class Calibradores(unittest.TestCase):
    def test_isotonica_es_monotona_y_platt_baja_el_ece(self):
        c, ok = sintetico()
        xs, ys = CC.isotonica(c, ok)
        self.assertTrue(np.all(np.diff(ys) >= -1e-12) and np.all(np.diff(xs) >= 0))
        for metodo in ("isotonica", "platt"):
            f, par = CC.ajustar_confianza(metodo, c[:1500], ok[:1500])
            self.assertLess(CC.ece(f(c[1500:]), ok[1500:]), CC.ece(c[1500:], ok[1500:]) / 2, metodo)
            json.dumps(par)

    def test_platt_una_sola_clase(self):
        a, b = CC.platt([0.6, 0.7], [1, 1])
        self.assertEqual((a, b), (0.0, 8.0))

    def test_calibrar_capa_tau_fuera_de_pliegue(self):
        c, ok = sintetico(4000)
        r = CC.calibrar_capa(c, ok, claves(4000))
        self.assertEqual(r["n_ajuste"] + r["n_tau"], 4000)
        self.assertTrue(r["ajustado"])
        self.assertLess(r["ece_calibrado"], r["ece_crudo"])
        self.assertGreaterEqual(r["precision_cubierta"], 0.95 - 1e-9) if r["cobertura"] > 0 else None
        self.assertLessEqual(r["rc_calibrado"]["aurc"], r["rc_crudo"]["aurc"] + 0.01)
        # la confianza calibrada con filas de «ajuste» es una función monótona: el τ es un umbral válido en la escala nueva
        self.assertTrue(0 <= r["tau"] <= 1)

    def test_pocas_filas_no_se_ajusta_y_se_dice(self):
        c, ok = sintetico(30)
        r = CC.calibrar_capa(c, ok, claves(30))
        self.assertFalse(r["ajustado"])
        self.assertEqual(r["calibrador"], {"metodo": "ninguno"})

    def test_tau_con_ajuste_optimista_frente_a_fuera_de_pliegue(self):
        # una capa que memoriza sus filas de ajuste (confianza 1 y acierto) y falla fuera: τ en las mismas filas sale 0,5; fuera, no hay τ útil
        r = np.random.default_rng(0)
        n = 2000
        cl = claves(n)
        aj, ta = CC.particion_hash(cl)
        ok = np.zeros(n, bool)
        ok[aj] = True
        ok[ta] = r.uniform(size=len(ta)) < 0.6
        conf = np.full(n, 0.97)
        conf[aj] = 0.99
        mismas = CC.tau_precision(conf, ok)               # todas las filas juntas
        fuera = CC.tau_precision(conf[ta], ok[ta])        # solo las no vistas
        self.assertGreaterEqual(mismas[2], 0.95)
        self.assertEqual(fuera[1], 0.0)                   # nada llega a 0,95: se escala siempre
        self.assertEqual(fuera[0], 1.0)

    def test_cli(self):
        c, ok = sintetico(600)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            et = [{"id": "x%d" % i, "tarea": "t", "texto": "m%d" % i, "etiqueta": "a" if o else "b"} for i, o in enumerate(ok)]
            pr = [{"id": "x%d" % i, "probs": {"a": float(p), "b": float(1 - p)}} for i, p in enumerate(c)]
            (d / "et.jsonl").write_text("".join(json.dumps(x) + "\n" for x in et))
            (d / "l0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pr))
            self.assertEqual(CC.main(["--capa", "L0=%s" % (d / "l0.jsonl"), "--capa", "L3=%s" % (d / "l0.jsonl"), "--etiquetas", str(d / "et.jsonl"),
                                      "--salida", str(d / "s.json"), "--metodo", "platt"]), 0)
            s = json.loads((d / "s.json").read_text())
            self.assertEqual(set(s), {"L0", "L3"})
            self.assertEqual(s["L0"]["calibrador"]["metodo"], "platt")


if __name__ == "__main__":
    unittest.main()
