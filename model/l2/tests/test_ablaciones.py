"""Cabeza noul de dos torres, ablaciones de L2 (K, coseno, λ), línea base kNN, diagnósticos de noul y de prototipos."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from model.evaluacion.jevbench import cargar_jevbench
from model.l2 import diag_noul_descripciones as DN
from model.l2 import diag_prototipos_jevbench as DP
from model.l2 import linea_base_knn_e5 as KN
from model.l2.evaluar import cargar_modelo
from model.l2.modelo import CABEZAS_NOUL, ModeloL2, textos_torre_fila
from model.l2.tests.falso import CodificadorFalso, fila

FX = Path(__file__).resolve().parents[2] / "evaluacion" / "tests" / "fixtures" / "jevbench"


def noul(n=12):
    return [fila(i, "noul", tarea="hate") for i in range(n)]


def probs(m, filas, enc=None):
    with torch.no_grad():
        p, _ = m(filas, enc or CodificadorFalso(), entrenar=False)
    return torch.stack(p)


class CabezaNoul(unittest.TestCase):
    def test_una_es_el_defecto_y_no_cambia_el_estado(self):
        torch.manual_seed(0)
        a = ModeloL2()
        torch.manual_seed(0)
        b = ModeloL2(cabeza_noul="una")
        self.assertEqual(a.cabeza_noul, "una")
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        self.assertNotIn("log_t.noul", a.state_dict())
        self.assertTrue(torch.equal(probs(a.eval(), noul()), probs(b.eval(), noul())))
        self.assertEqual(len(textos_torre_fila(noul(1)[0])), 1)

    def test_dos_torres_codifican_true_y_false(self):
        f = noul(1)[0]
        for c in ("dos_dif", "dos_softmax"):
            ts = textos_torre_fila(f, "v1", "coral1", c)
            self.assertEqual(len(ts), 2)
            self.assertIn("true", ts[0])
            self.assertIn("false", ts[1])
        with self.assertRaises(ValueError):
            ModeloL2(cabeza_noul="otra")

    def test_probabilidades_y_gradiente_llegan_a_la_torre_de_false(self):
        for c in ("dos_dif", "dos_softmax"):
            torch.manual_seed(1)
            m = ModeloL2(cabeza_noul=c)
            enc = CodificadorFalso()
            p, ps = m(noul(), enc, entrenar=True)
            for x in p:
                self.assertAlmostEqual(float(x.sum()), 1.0, places=5)
            ps["total"].backward()
            self.assertGreater(float(m.torre.mlp[0].weight.grad.abs().sum()), 0)

    def test_la_descripcion_de_false_solo_importa_con_dos_torres(self):
        torch.manual_seed(2)
        filas = noul()
        for c, debe_cambiar in (("una", False), ("dos_dif", True), ("dos_softmax", True)):
            m = ModeloL2(cabeza_noul=c).eval()
            cambiadas = [DN.variante(r, "false_cambiada") for r in filas]
            a, b = probs(m, filas), probs(m, cambiadas)
            self.assertEqual(not torch.equal(a, b), debe_cambiar, c)
            res = DN.diagnosticar(m, CodificadorFalso(), filas)
            self.assertEqual(res["cabeza_noul"], c)
            self.assertEqual(res["false_cambiada"]["frac_identicas"] == 1.0, not debe_cambiar)
            self.assertEqual(res["n"], len(filas))

    def test_checkpoint_antiguo_carga_como_una(self):
        torch.manual_seed(3)
        m = ModeloL2()
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "viejo.pt"
            torch.save({"modelo": m.state_dict(), "cabezas": [], "paso": 0}, ruta)    # sin cabeza_noul, sim ni k_ranuras
            n, ck = cargar_modelo(str(ruta))
        self.assertEqual((n.cabeza_noul, n.sim_modo), ("una", "maxsim"))
        self.assertTrue(torch.equal(probs(m.eval(), noul()), probs(n, noul())))

    def test_checkpoint_de_una_carga_en_dos_softmax(self):
        m = ModeloL2()
        n = ModeloL2(cabeza_noul="dos_softmax")
        n.load_state_dict(m.state_dict())
        self.assertAlmostEqual(float(n.log_t["noul"].exp()), 0.25, places=5)

    def test_calib_no_exporta_dos_torres(self):
        from model.l2 import calibrar as C
        with self.assertRaises(ValueError):
            C.a_calib_chispa({"tipos": {}, "tau_escalate": 0.5, "cabeza_noul": "dos_dif", "sim": "maxsim"})
        C.a_calib_chispa({"tipos": {}, "tau_escalate": 0.5, "cabeza_noul": "una", "sim": "maxsim"})

    def test_calibrar_con_dos_torres(self):
        from model.l2 import calibrar as C
        for c in CABEZAS_NOUL:
            torch.manual_seed(4)
            m = ModeloL2(cabeza_noul=c).eval()
            inf = C.calibrar(m, CodificadorFalso(), noul(60), tau_fuera_pliegue=True)
            self.assertEqual(inf["cabeza_noul"], c)
            self.assertTrue(0 <= inf["tau_escalate"] <= 1)


class Ablaciones(unittest.TestCase):
    def test_k_ranuras_y_carga_recortada(self):
        torch.manual_seed(5)
        m16 = ModeloL2()
        m1 = ModeloL2(k=1)
        m1.load_state_dict(m16.state_dict())
        self.assertTrue(torch.equal(m1.ranuras.q, m16.ranuras.q[:1]))
        with self.assertRaises(ValueError):
            m16.load_state_dict(m1.state_dict())
        with tempfile.TemporaryDirectory() as d:
            torch.save({"modelo": m1.state_dict(), "cabezas": []}, Path(d) / "k1.pt")
            n, _ = cargar_modelo(str(Path(d) / "k1.pt"))
        self.assertEqual(n.ranuras.q.shape[0], 1)
        self.assertEqual(probs(n, noul(4)).shape, (4, 2))

    def test_coseno_y_no_exportable(self):
        torch.manual_seed(6)
        m = ModeloL2(sim="coseno").eval()
        S = torch.nn.functional.normalize(torch.randn(3, 16, 256), dim=-1)
        h = torch.nn.functional.normalize(torch.randn(5, 256), dim=-1)
        s = m.sim(S, h)
        self.assertEqual(s.shape, (3, 5))
        self.assertLessEqual(float(s.abs().max()), 1.0 + 1e-6)
        with self.assertRaises(ValueError):
            m.exportable()
        ModeloL2().exportable()
        with self.assertRaises(ValueError):
            ModeloL2(sim="otra")

    def test_lambdas_a_cero_apagan_las_perdidas(self):
        filas = [fila(i, "choice", 4) for i in range(8)]
        enc = CodificadorFalso()
        torch.manual_seed(7)
        base = ModeloL2()
        torch.manual_seed(7)
        sin = ModeloL2(lambda_nce=0.0, lambda_brier=0.0)
        _, pb = base(filas, enc)
        _, ps = sin(filas, enc)
        self.assertGreater(float(pb["nce"]), 0)
        self.assertAlmostEqual(float(ps["total"]), float(ps["ce"] + ps["kl"]), places=5)
        self.assertAlmostEqual(float(pb["total"]), float(pb["ce"] + pb["kl"] + 0.5 * pb["nce"] + 0.5 * pb["brier"]), places=5)

    def test_variantes_del_runner(self):
        from model.l2.experimentos import correr
        import argparse
        base = dict(ckpt="m1.pt", tsjev="ts", pasos=10, semilla=0, variante="control", ruido_p=0.15, lote=8, dev_max=10, device="cpu")
        control = correr.args_entrenar(argparse.Namespace(**base), "s")
        for v, extra in correr.ABLACIONES.items():
            a = correr.args_entrenar(argparse.Namespace(**{**base, "variante": v}), "s")
            self.assertEqual(a[:len(control)], control)
            self.assertEqual(a[len(control):], extra)
        for f in ("--cabeza-noul", "--k-ranuras", "--sim", "--lambda-nce", "--lambda-brier"):
            self.assertNotIn(f, control)         # apagadas por defecto


class LineaBaseYDiagnosticos(unittest.TestCase):
    def test_knn_e5_mismo_soporte(self):
        filas = cargar_jevbench(FX)
        enc = CodificadorFalso()
        for k in (8, 32):
            preds = KN.predecir_knn(enc, filas, k)
            self.assertEqual([p["id"] for p in preds], [r["id"] for r in filas])
            for p, r in zip(preds, filas):
                self.assertAlmostEqual(sum(p["probs"].values()), 1.0, places=6)
                self.assertEqual(set(p["probs"]), set(r["opciones"]))

    def test_diag_prototipos_en_el_fixture(self):
        filas = cargar_jevbench(FX)
        for t in ("choice", "noul", "score"):
            torch.manual_seed(8)
        m = ModeloL2()
        for k in {len(r["opciones"]) for r in filas if r["tipo"] == "score"}:
            m.asegurar_cabeza("score%d" % k)
        res = DP.diagnosticar(m, CodificadorFalso(), filas, k=2, semillas=(1,))
        self.assertTrue(res["lectura"]["H1_cableado_ok"])      # β = 0 reproduce k = 0 (el cableado del soporte no cuela nada más)
        self.assertEqual(res["soporte"]["fugas_caso_o_parafrasis_en_soporte"], 0)
        self.assertIn("H5_etiquetas_barajadas", res["variantes"])
        json.dumps(res)
        self.assertIn("| variante |", DP.md(res))


if __name__ == "__main__":
    unittest.main()
