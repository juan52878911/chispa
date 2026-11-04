import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from model.evaluacion.jevbench import cargar_jevbench
from model.l2 import diag_score_jevbench as DJ
from model.l2.modelo import ModeloL2
from model.l2.tests.falso import CodificadorFalso

FX = Path(__file__).resolve().parents[2] / "evaluacion" / "tests" / "fixtures" / "jevbench"


class Diagnostico(unittest.TestCase):
    def test_signo_histograma_y_cli(self):
        filas = [r for r in cargar_jevbench(FX) if r["tipo"] == "score"]
        self.assertTrue(filas)
        for cab in ("coral1", "acumulados"):
            torch.manual_seed(0)
            m, enc = ModeloL2(cabeza_score=cab).eval(), CodificadorFalso()
            a = DJ.predecir(m, enc, filas)
            b = DJ.predecir(m, enc, filas, -1.0)
            c = DJ.predecir(m, enc, filas)
            self.assertEqual(a, c)            # el parche del signo se deshace
            self.assertNotEqual(a, b)
            res = DJ.resumir(filas, a, b)
            self.assertEqual(res["n"], len(filas))
            for k, d in res["por_k"].items():
                self.assertEqual(sum(d["s"]["mediana_predicha"].values()), d["n"])
                self.assertEqual(sum(d["menos_s"]["mediana_predicha"].values()), d["n"])
                self.assertEqual(sum(d["oro"].values()), d["n"])
            self.assertTrue(0 <= res["acc_s"] <= 1 and 0 <= res["acc_menos_s"] <= 1)
        with tempfile.TemporaryDirectory() as d:
            m = ModeloL2()
            for k in {len(r["opciones"]) for r in filas}:
                m.asegurar_cabeza("score%d" % k)
            torch.save({"modelo": m.state_dict(), "cabezas": sorted(m.coral.keys())}, Path(d) / "m.pt")
            enc = CodificadorFalso()
            with mock.patch("model.l2.modelo.Codificador", lambda **kw: enc):
                self.assertEqual(DJ.main(["--ckpt", str(Path(d) / "m.pt"), "--jevbench", str(FX), "--salida", str(Path(d) / "s")]), 0)
            js = json.loads((Path(d) / "s" / "diag_score_jevbench.json").read_text())
            self.assertEqual(js["cabeza_score"], "coral1")
            self.assertIn("mediana predicha con −s", (Path(d) / "s" / "diag_score_jevbench.md").read_text())


if __name__ == "__main__":
    unittest.main()
