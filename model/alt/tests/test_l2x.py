import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertModel

from model.alt import entrenar as EN
from model.alt.l2x import L2x
from model.alt.modelo import opciones_ordenadas

from .diminuto import filas, tokenizer


def modelo(**kw):
    tok = tokenizer(None)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=256)
    return L2x(BertModel(cfg), tok, "diminuto-aleatorio", **kw)


def fila_grande(n_ops=20, i=0):
    ops = ["op%d" % j for j in range(n_ops)]
    return {"id": "g:train:%d" % i, "tarea": "grande", "tipo": "choice", "pregunta": "cual es", "texto": "texto de prueba largo " * 6, "opciones": ops,
            "descripciones": {o: "descripcion de %s" % o for o in ops}, "etiqueta": ops[i % n_ops], "idioma": "es", "split": "train", "fuente": "demo",
            "licencia": "cc-by-4.0", "uso_comercial": True}


class Entrada(unittest.TestCase):
    def test_secuencia_lleva_una_marca_por_opcion_y_tramos_coherentes(self):
        m = modelo()
        f = filas(1)[0]
        ids, spans, st = m.armar(f, ["a", "b", "c"])
        self.assertEqual(ids.count(m.marca_id), 3)
        self.assertEqual([ids[a] for a, _ in spans], [m.marca_id] * 3)
        self.assertLessEqual(spans[-1][1], st[0])   # las opciones van antes que el texto
        self.assertEqual(st[1], len(ids) - 1)        # el texto acaba justo antes del último [SEP]
        self.assertIn("segunda", m.tok.decode(ids[spans[1][0]:spans[1][1]]))

    def test_presupuesto_deja_al_texto_su_sitio_con_ocho_opciones(self):
        m = modelo(max_tok=256)
        f = fila_grande(8)
        f["pregunta"] = "pregunta larga " * 40
        f["descripciones"] = {o: "descripcion larguisima " * 40 for o in f["opciones"]}
        f["texto"] = "palabra " * 400
        ids, spans, (a, b) = m.armar(f, f["opciones"])
        self.assertLessEqual(len(ids), 256)
        self.assertGreaterEqual(b - a, 64)

    def test_tensores_rellenan_con_tramos_vacios(self):
        m = modelo()
        f2, f3 = filas(3)[1], filas(3)[0]   # noul de 2 opciones y choice de 3
        ids, mask, om, tm = m.tensores([(f3, ["a", "b", "c"]), (f2, ["false", "true"])])
        self.assertEqual(om.shape[:2], (2, 3))
        self.assertEqual(float(om[1, 2].sum()), 0.0)
        self.assertGreater(float(tm[0].sum()), 0)


class Salida(unittest.TestCase):
    def test_probs_suman_uno_con_pocas_y_con_muchas_opciones(self):
        m = modelo()
        fs = filas(6) + [fila_grande(20), fila_grande(75, 1)]
        for f, p in zip(fs, m.probs(fs)):
            self.assertEqual(set(p), set(f["opciones"]))
            self.assertAlmostEqual(sum(p.values()), 1.0, places=6)

    def test_score_se_ordena_por_id_numerico(self):
        m = modelo()
        sc = dict(next(f for f in filas(6) if f["tipo"] == "score"), opciones=["3", "1", "2"])
        self.assertEqual(opciones_ordenadas(sc), ["1", "2", "3"])
        self.assertEqual(list(m.probs([sc])[0]), ["1", "2", "3"])

    def test_perdida_con_muchas_opciones_conserva_la_correcta(self):
        m = modelo()
        f = fila_grande(30, 5)
        for _ in range(20):
            ops = m._elegir(f)
            self.assertEqual(len(ops), m.max_opciones)
            self.assertIn(f["etiqueta"], ops)
        self.assertTrue(torch.isfinite(m.perdida([f, fila_grande(3, 2)], 0.5)))

    def test_se_puede_sobreajustar_dos_filas(self):
        torch.manual_seed(0)
        m = modelo()
        fs = [dict(f, tarea="x") for f in filas(6)[:2]]
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        m.train()
        antes = float(m.perdida(fs))
        for _ in range(60):
            opt.zero_grad()
            p = m.perdida(fs)
            p.backward()
            opt.step()
        self.assertLess(float(m.perdida(fs)), antes * 0.5)

    def test_blanda_cambia_la_perdida(self):
        m = modelo()
        f = filas(8)[0]
        self.assertIn("blanda", f)
        m.eval()
        torch.manual_seed(0)
        m._rng.seed(0)
        a = float(m.perdida([f], 0.0))
        m._rng.seed(0)
        b = float(m.perdida([f], 1.0))
        self.assertNotAlmostEqual(a, b, places=4)

    def test_guardar_y_cargar_dan_las_mismas_probabilidades(self):
        m = modelo(max_opciones=5)
        fs = filas(6)
        with tempfile.TemporaryDirectory() as d:
            m.guardar(Path(d) / "ck")
            n = L2x.cargar(Path(d) / "ck")
            self.assertEqual(n.max_opciones, 5)
            for p, q in zip(m.probs(fs), n.probs(fs)):
                for o in p:
                    self.assertAlmostEqual(p[o], q[o], places=5)


class Entrenamiento(unittest.TestCase):
    def test_entrenar_dos_pasos_y_evaluar_con_el_arnes(self):
        from model.alt import evaluar as EV
        m = modelo()
        train, dev = [f for f in filas(20) if f["split"] == "train"], [f for f in filas(20) if f["split"] == "dev"]

        class Man:
            def escribir_junto(self, d):
                Path(d).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as d:
            hist = EN.entrenar(m, Man(), train, dev, d, epocas=1, lote=4, pasos=2, log=lambda *_: None)
            self.assertEqual(hist[0]["pasos"], 2)
            self.assertTrue((Path(d) / "mejor" / "modelo.pt").exists())
            pr = EV.predecir(L2x.cargar(Path(d) / "mejor"), dev, lote=2)
            self.assertEqual(len(pr), len(dev))

    def test_sondeo_cuenta_las_secuencias_de_l2x(self):
        m = modelo()
        r = EN.sondeo(m, 30, max_tokens=2048)
        self.assertTrue(r["ok"], r)


class Onnx(unittest.TestCase):
    def test_exportar_y_predecir_con_ort_da_lo_mismo_que_torch(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("sin onnxruntime")
        from model.alt.l2x_onnx import OrtL2x, exportar
        m = modelo()
        fs = filas(6) + [fila_grande(20)]
        with tempfile.TemporaryDirectory() as d:
            m.guardar(Path(d) / "ck")
            r = exportar(Path(d) / "ck", Path(d) / "onnx")
            self.assertLess(r["max_dif_fp32"], 1e-3)
            o = OrtL2x(r["fp32"], Path(d) / "ck")
            for p, q in zip(m.probs(fs), o.probs(fs)):
                for k in p:
                    self.assertAlmostEqual(p[k], q[k], places=3)


if __name__ == "__main__":
    unittest.main()
