import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from model.l2 import entrenar
from model.l2.experimentos import correr
from model.l2.modelo import ModeloL2
from model.l2.tests.falso import CodificadorFalso, fila

FX = Path(__file__).resolve().parents[2] / "evaluacion" / "tests" / "fixtures" / "jevbench"


def ns(**kw):
    base = dict(ckpt="m1.pt", tsjev="ts", pasos=100, semilla=3, variante="control", ruido_p=0.15, lote=64, dev_max=500, device="cpu")
    base.update(kw)
    return argparse.Namespace(**base)


class Argumentos(unittest.TestCase):
    def test_variantes_solo_cambian_cabeza_y_ruido(self):
        por = {v: correr.args_entrenar(ns(variante=v), "sal") for v in correr.VARIANTES}
        control = por["control"]
        self.assertEqual(control[control.index("--cabeza-score") + 1], "coral1")
        self.assertNotIn("--ruido-cond", control)
        for v, cab in (("coral_vec", "coral"), ("acumulados", "acumulados"), ("grueso_fino", "grueso_fino")):
            a = por[v]
            self.assertEqual(a[a.index("--cabeza-score") + 1], cab)
            self.assertEqual([x for x in a if x != cab], [x for x in control if x != "coral1"])
        r = por["ruido"]
        self.assertEqual(r[r.index("--ruido-cond") + 1], "0.15")
        self.assertEqual([x for x in r if x not in ("--ruido-cond", "0.15")], control)

    def test_hiperparametros_del_ajuste_1(self):
        a = correr.args_entrenar(ns(), "sal")
        g = lambda f: a[a.index(f) + 1]
        self.assertEqual((g("--max-tok"), g("--lote"), g("--desde"), g("--meta"), g("--semilla"), g("--pasos")), ("256", "64", "m1.pt", "ts", "3", "100"))
        self.assertEqual((g("--tope-tarea"), g("--alfa")), ("2000", "0.3"))
        for f in ("--aumento", "--blandas", "--episodios", "--maestro-extra", "--datos", "--lr", "--lr-torre"):   # lr: el de --desde (3e-5, 1e-5)
            self.assertNotIn(f, a)
        self.assertEqual(g("--pasos"), "100")
        self.assertEqual(correr.args_entrenar(ns(), "s", pasos=20)[correr.args_entrenar(ns(), "s", pasos=20).index("--pasos") + 1], "20")

    def test_segundos_por_paso(self):
        self.assertEqual(correr.segundos_por_paso("paso 1/20 x 5.00 s/paso cache=1/2\npaso 20/20 ce=1 lr=1e-5 1.25 s/paso cache=3/4"), 1.25)
        self.assertIsNone(correr.segundos_por_paso("nada"))


class DePuntaAPunta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        filas = [fila(i, "choice", 4, "a") for i in range(24)] + [fila(100 + i, "noul", tarea="b") for i in range(12)] \
            + [fila(200 + i, "score", tarea="c", k=4) for i in range(16)]
        (self.d / "ts").mkdir()
        (self.d / "ts" / "tsjev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in filas))
        torch.manual_seed(0)
        m = ModeloL2()
        m.asegurar_cabeza("score4")
        torch.save({"modelo": m.state_dict(), "cabezas": ["score4"], "paso": 0, "max_tok": 256, "texto_torre": "v1"}, self.d / "m1.pt")

    def correr(self, variante, salida, *extra):
        enc = CodificadorFalso()
        with mock.patch.object(entrenar, "Codificador", lambda **kw: enc), mock.patch("model.l2.modelo.Codificador", lambda **kw: enc):
            return correr.main(["--ckpt", str(self.d / "m1.pt"), "--tsjev", str(self.d / "ts"), "--jevbench", str(FX), "--salida", str(self.d / salida),
                                "--pasos", "3", "--lote", "8", "--semilla", "1", "--variante", variante, *extra])

    def test_cuatro_variantes_escriben_preds_y_mismos_lotes(self):
        cantidades = {}
        for v in correr.VARIANTES:
            self.assertEqual(self.correr(v, v), 0, v)
            filas = [json.loads(l) for l in open(self.d / v / "jevbench" / "pred" / "l2_k0.jsonl")]
            self.assertEqual(len(filas), 13)
            for p in filas:
                self.assertAlmostEqual(sum(p["probs"].values()), 1.0, places=5)
            ck = torch.load(self.d / v / "entrenamiento" / "ultimo.pt", weights_only=False)
            self.assertEqual(ck["paso"], 3)
            self.assertEqual(ck["cabeza_score"], {"coral_vec": "coral", "acumulados": "acumulados", "grueso_fino": "grueso_fino"}.get(v, "coral1"))
            cantidades[v] = json.loads((self.d / v / "entrenamiento" / "entrenamiento.json").read_text())["filas"]
            self.assertEqual(json.loads((self.d / v / "correr.json").read_text())["variante"], v)
        self.assertEqual(len(set(cantidades.values())), 1)      # los mismos datos en todas

    def test_control_reproducible(self):
        self.correr("control", "a")
        self.correr("control", "b")
        lee = lambda d: [{k: v for k, v in json.loads(l).items() if k != "ms"} for l in open(self.d / d / "jevbench" / "pred" / "l2_k0.jsonl")]   # `ms` es un tiempo
        self.assertEqual(lee("a"), lee("b"))

    def test_solo_tiempo(self):
        self.assertEqual(self.correr("ruido", "t", "--solo-tiempo"), 0)
        t = json.loads((self.d / "t" / "tiempo.json").read_text())
        self.assertEqual(t["pasos_medidos"], 20)
        self.assertEqual(t["pasos"], 3)
        self.assertGreater(t["s_por_paso"], 0)
        self.assertFalse((self.d / "t" / "jevbench").exists())      # no evalúa


if __name__ == "__main__":
    unittest.main()
