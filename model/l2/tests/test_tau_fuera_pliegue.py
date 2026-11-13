"""τ fuera de pliegue en `calibrar`, aviso de `filas_dev` y la comparación dev → test."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from model.l2 import calibrar as C
from model.l2 import calibrar_capas as CC
from model.l2 import tau_val_vs_test as TV
from model.l2.modelo import ModeloL2
from model.l2.tests.falso import CodificadorFalso, fila


def filas(n=80, desplazo=0):
    return [fila(desplazo + i, t, 4 if t == "choice" else 2, tarea="tarea_" + t, k=5) for t in ("choice", "noul", "score") for i in range(n)]


class TauFueraDePliegue(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m, self.enc = ModeloL2().eval(), CodificadorFalso()

    def test_el_ajuste_y_tau_usan_filas_distintas(self):
        fs = filas()
        inf = C.calibrar(self.m, self.enc, fs, tau_fuera_pliegue=True)
        g = inf["global"]
        self.assertTrue(inf["tau_fuera_pliegue"])
        self.assertEqual(g["n_ajuste"] + g["n_tau"], len(fs))
        self.assertGreater(g["n_tau"], 0)
        for t, r in inf["tareas"].items():
            self.assertEqual(r["n"] + r["n_tau"], 80, t)         # n = filas de ajuste, n_tau = las que eligen τ
            self.assertIn("rc", r)
        self.assertIn("aurc", g)
        # las filas de ajuste son exactamente el complemento de las de τ
        aj, ta = CC.particion_hash(fs)
        self.assertEqual(g["n_ajuste"], len(aj))

    def test_el_procedimiento_antiguo_sigue_igual(self):
        fs = filas()
        antiguo = C.calibrar(self.m, self.enc, fs)
        self.assertFalse(antiguo["tau_fuera_pliegue"])
        for t, r in antiguo["tareas"].items():
            self.assertEqual(r["n"], r["n_tau"], t)
        explicito = C.calibrar(self.m, self.enc, fs, tau_fuera_pliegue=False)
        self.assertEqual(json.dumps(antiguo["tareas"], sort_keys=True), json.dumps(explicito["tareas"], sort_keys=True))

    def test_tareas_no_vistas_dan_aurc_aparte(self):
        inf = C.calibrar(self.m, self.enc, filas(), tau_fuera_pliegue=True, tareas_no_vistas=["tarea_noul"])
        rc = inf["riesgo_cobertura"]
        self.assertEqual(set(rc), {"global", "no_vistas", "vistas"})
        self.assertTrue(inf["tareas"]["tarea_noul"]["no_vista"])
        self.assertFalse(inf["tareas"]["tarea_choice"]["no_vista"])
        json.dumps(inf, default=float)

    def test_filas_dev_avisa_a_gritos_al_caer_a_todas(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "x.jsonl"
            ruta.write_text("".join(json.dumps(r) + "\n" for r in filas(3)))        # todas split=train
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.assertEqual(len(C.filas_dev([str(ruta)])), 9)
            self.assertTrue(any("split=dev" in str(x.message) for x in w))
            with self.assertRaises(SystemExit):
                C.filas_dev([str(ruta)], estricto=True)
            dev = [dict(r, split="dev") for r in filas(3)]
            ruta.write_text("".join(json.dumps(r) + "\n" for r in dev))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.assertEqual(len(C.filas_dev([str(ruta)])), 9)
            self.assertFalse(w)

    def test_comparacion_dev_test(self):
        dev, test = filas(80), filas(80, desplazo=1000)
        res = TV.comparar(self.m, self.enc, dev, test)
        self.assertEqual(set(res), {"antiguo_mismas_filas", "nuevo_fuera_de_pliegue"})
        for r in res.values():
            self.assertIn("tau_test", next(iter(r["tareas"].values())))
            self.assertIn("media", r)
        self.assertIn("| procedimiento |", TV.tabla(res))
        json.dumps(res, default=float)


if __name__ == "__main__":
    unittest.main()
