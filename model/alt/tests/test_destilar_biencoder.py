"""Piloto R10: pérdidas, exclusiones, episodios, entrenamiento de principio a fin con datos sintéticos y evaluación int8. Sin red."""
import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from model.alt import destilar, destilar_biencoder as DB, eval_r10_piloto as EV, r10_sintetico
from model.l2 import datos as D
from model.l2.modelo import ModeloL2, prototipos
from model.l2.tests.falso import CodificadorFalso, fila

CRIT = Path(__file__).resolve().parents[1] / "crit-r10-piloto.json"
CRIT_SHA = "dd2d102b001e1882f52d07b2ea60a34410332802dd530dfa6ac642a0a8f4bccb"


def con_blanda(r, p=None):
    ops = r["opciones"]
    r = dict(r)
    r["blanda"] = p or {o: (0.6 if i == 0 else 0.4 / (len(ops) - 1)) for i, o in enumerate(ops)}
    return r


class Perdidas(unittest.TestCase):
    def test_kl_cero_si_iguales_y_positivo_si_no(self):
        t = torch.tensor([0.7, 0.2, 0.1])
        self.assertAlmostEqual(float(DB.kl_temp(t, t.clone())), 0.0, places=6)
        self.assertGreater(float(DB.kl_temp(t, torch.tensor([0.1, 0.2, 0.7]))), 0.01)

    def test_temperatura_aplana(self):
        p = torch.tensor([0.9, 0.05, 0.05])
        self.assertLess(float(DB.tempering(p, 2.0)[0]), 0.9)
        torch.testing.assert_close(DB.tempering(p, 1.0), p)

    def test_emd_sensible_a_la_distancia(self):
        oro = F.one_hot(torch.tensor(0), 5).float()
        cerca, lejos = F.one_hot(torch.tensor(1), 5).float(), F.one_hot(torch.tensor(4), 5).float()
        self.assertEqual(float(DB.emd(oro, oro)), 0.0)
        self.assertLess(float(DB.emd(cerca, oro)), float(DB.emd(lejos, oro)))
        self.assertAlmostEqual(float(DB.emd(lejos, oro)), 1.0)

    def test_oro_suavizado_decrece_con_la_distancia(self):
        w = DB.suavizar_oro(5, 2)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        self.assertGreater(float(w[2]), float(w[1]))
        self.assertGreater(float(w[1]), float(w[0]))

    def test_desacuerdo_oro_manda(self):
        r = con_blanda(fila(0, "choice", 3), {"op0": 0.1, "op1": 0.8, "op2": 0.1})   # etiqueta = op0, el maestro dice op1
        t, disc = DB.objetivo_maestro(r, "intercambiar")
        self.assertTrue(disc)
        self.assertEqual(int(t.argmax()), 0)
        self.assertAlmostEqual(float(t.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(t[1]), 0.1, places=5)
        self.assertIsNone(DB.objetivo_maestro(r, "ignorar")[0])
        self.assertEqual(DB.objetivo_maestro(r, "oro")[0].tolist(), [1.0, 0.0, 0.0])
        t, disc = DB.objetivo_maestro(con_blanda(fila(0, "choice", 3), {"op0": 0.8, "op1": 0.1, "op2": 0.1}))
        self.assertFalse(disc)
        self.assertEqual(int(t.argmax()), 0)
        self.assertEqual(DB.objetivo_maestro(fila(0, "choice", 3)), (None, False))


def cfg(**kw):
    c = {"peso_kl": 0.6, "peso_ce": 0.2, "peso_contr": 0.2, "peso_ref": 2.0, "temp_kl": 2.0, "minadas": 4, "desacuerdo": "intercambiar", "score_perdida": "emd"}
    c.update(kw)
    return c


def lote(n=6):
    fs = []
    for i in range(n):
        fs += [con_blanda(fila(i, "choice", 4, tarea="c%d" % (i % 2))), con_blanda(fila(i, "noul", tarea="n")), con_blanda(fila(i, "score", k=5, tarea="s"))]
    return fs


class Calculo(unittest.TestCase):
    def test_probs_coinciden_con_el_modelo_de_produccion(self):
        torch.manual_seed(0)
        m, enc = ModeloL2(), CodificadorFalso()
        fs = lote()
        for r in fs:
            m.asegurar_cabeza("score5" if r["tipo"] == "score" else r["tipo"])
        _, probs = DB.calcular(m, enc, None, fs, cfg(peso_ref=0))
        with torch.no_grad():
            ref, _ = m(fs, enc, entrenar=False)
        for a, b in zip(probs, ref):
            torch.testing.assert_close(a.detach(), b.detach(), atol=1e-5, rtol=1e-4)

    def test_probs_con_prototipos_coinciden(self):
        torch.manual_seed(0)
        m, enc = ModeloL2(), CodificadorFalso()
        sop = [fila(i, "choice", 3, tarea="c") for i in range(12)]
        fs = [fila(i + 20, "choice", 3, tarea="c") for i in range(4)]
        protos = {"c": prototipos(m, enc, sop)}
        _, probs = DB.calcular(m, enc, None, fs, cfg(peso_ref=0), protos=protos)
        with torch.no_grad():
            ref, _ = m(fs, enc, entrenar=False, protos=protos)
        for a, b in zip(probs, ref):
            torch.testing.assert_close(a.detach(), b.detach(), atol=1e-5, rtol=1e-4)

    def test_solo_proto_ignora_la_descripcion(self):
        torch.manual_seed(0)
        m, enc = ModeloL2(), CodificadorFalso()
        sop = [fila(i, "choice", 3, tarea="c") for i in range(12)]
        fs = [fila(i + 20, "choice", 3, tarea="c") for i in range(4)]
        protos = {"c": prototipos(m, enc, sop)}
        _, p1 = DB.calcular(m, enc, None, fs, cfg(peso_ref=0), protos=protos, solo_proto=True)
        fs2 = [dict(r, descripciones={o: "otra descripcion " + o for o in r["opciones"]}) for r in fs]
        _, p2 = DB.calcular(m, enc, None, fs2, cfg(peso_ref=0), protos=protos, solo_proto=True)
        for a, b in zip(p1, p2):
            torch.testing.assert_close(a.detach(), b.detach())
        _, p3 = DB.calcular(m, enc, None, fs, cfg(peso_ref=0), protos=protos, solo_proto=False)
        self.assertFalse(torch.allclose(p1[0].detach(), p3[0].detach()))

    def test_gradiente_y_penalidad_e5(self):
        torch.manual_seed(0)
        m, enc, ref = ModeloL2(), CodificadorFalso(descongelar=1), CodificadorFalso()
        ps, _ = DB.calcular(m, enc, ref, lote(), cfg())
        self.assertAlmostEqual(float(ps["ref_msg"]), 0.0, places=4)   # mismo e5: sin penalidad
        enc.w.data += torch.randn_like(enc.w) * 0.5
        enc.invalidar()
        ps, _ = DB.calcular(m, enc, ref, lote(), cfg())
        self.assertGreater(float(ps["ref_msg"]), 1e-3)
        self.assertGreater(float(ps["ref_pas"]), 1e-3)
        for k in ("kl", "ce", "contrastivo", "total"):
            self.assertTrue(torch.isfinite(ps[k]), k)
        ps["total"].backward()
        self.assertIsNotNone(enc.w.grad)
        self.assertTrue(all(p.grad is not None for p in m.ranuras.parameters()))
        self.assertGreater(float(m.torre.mlp[0].weight.grad.abs().sum()), 0.0)

    def test_sin_ref_sin_penalidad(self):
        ps, _ = DB.calcular(ModeloL2(), CodificadorFalso(descongelar=1), None, lote(), cfg())
        self.assertEqual(float(ps["ref_pas"]), 0.0)

    def test_contrastivo_usa_negativos_de_otras_tareas(self):
        torch.manual_seed(0)
        m, enc = ModeloL2(), CodificadorFalso()
        ps0, _ = DB.calcular(m, enc, None, lote(), cfg(minadas=0, peso_ref=0))
        ps1, _ = DB.calcular(m, enc, None, lote(), cfg(minadas=8, peso_ref=0))
        self.assertGreaterEqual(float(ps1["contrastivo"]), float(ps0["contrastivo"]) - 1e-6)
        self.assertNotAlmostEqual(float(ps1["contrastivo"]), float(ps0["contrastivo"]), places=4)

    def test_score_emd_y_suavizado(self):
        torch.manual_seed(0)
        fs = [con_blanda(fila(i, "score", k=5, tarea="s")) for i in range(4)]
        for modo in ("emd", "suavizado"):
            ps, _ = DB.calcular(ModeloL2(), CodificadorFalso(), None, fs, cfg(score_perdida=modo, peso_ref=0))
            self.assertTrue(torch.isfinite(ps["total"]), modo)
        e, _ = DB.calcular(ModeloL2(), CodificadorFalso(), None, fs, cfg(score_perdida="emd", peso_ref=0))
        self.assertLessEqual(float(e["ce"]), 1.0)   # EMD normalizada: 0..1


class Exclusiones(unittest.TestCase):
    def setUp(self):
        self.h = DB.huellas_exclusion()

    def fila(self, tarea, id_="tsjev:x:1", fuente="x"):
        return {"tarea": tarea, "id": id_, "fuente": fuente}

    def test_v1_y_medicion(self):
        for t in ("multilingual/massive", "clinc_oos/plus", "prompt-injection-dataset", "Prompt-injection-dataset/full", "PromptShield", "multilingual/tweet_sentiment_multilingual"):
            self.assertIsNotNone(DB.motivo_exclusion(self.fila(t), self.h), t)
        self.assertIsNotNone(DB.motivo_exclusion(self.fila("x", id_="massive:1"), self.h))
        self.assertIsNotNone(DB.motivo_exclusion(self.fila("x", id_="clinc150:1"), self.h))

    def test_huellas_de_v2_selladas_y_dev(self):
        for h in ("consultas-unlu", "horeca-spanish-reviews", "consumer-finance-complaints", "usajobs-scraping", "congressional-record-speeches-1995-2026"):
            self.assertIn(h, self.h)
        self.assertIsNotNone(DB.motivo_exclusion(self.fila("t", fuente="mrcsgh/horeca-spanish-reviews"), self.h))
        self.assertIsNotNone(DB.motivo_exclusion(self.fila("consultas-unlu"), self.h))

    def test_filas_admitidas_pasan(self):
        for t in ("CONDAQA", "esci", "soporte_ti", "spam", "go_emotions/simplified", "banking77"):
            self.assertIsNone(DB.motivo_exclusion(self.fila(t, id_="x:1"), self.h), t)

    def test_casi_duplicados(self):
        ref = CodificadorFalso()
        d = DB.casi_duplicados(ref, ["hola mundo", "otro texto distinto", "hola mundo"], ["hola mundo", "xyz"], 0.95)
        self.assertEqual(d, {0, 2})
        self.assertEqual(DB.casi_duplicados(ref, ["a"], [], 0.95), set())

    def test_dev_v2_no_abre_selladas(self):
        with tempfile.TemporaryDirectory() as t:
            s = Path(t) / "selladas"
            s.mkdir()
            with self.assertRaises(ValueError):
                DB.textos_dev_v2(s)


class Recorte(unittest.TestCase):
    def test_recorta_por_textos_distintos_y_deja_4(self):
        m = ModeloL2()
        fs = [fila(i, "choice", 10, tarea="t%d" % i) for i in range(10)]
        self.assertEqual(len(DB.recortar_lote(fs, m, 10 ** 6)), 10)
        self.assertEqual(len(DB.recortar_lote(fs, m, 35)), 4)   # con la fila 4 ya hay 40 > 35, pero se dejan al menos 4
        self.assertEqual(len(DB.recortar_lote(fs, m, 1)), 4)
        self.assertEqual(len(DB.recortar_lote(fs, m, 60)), 6)


class Episodios(unittest.TestCase):
    def filas(self):
        out = []
        for c in range(3):
            for i in range(14):
                r = fila(i + 100 * c, "choice", 3, tarea="fija")
                r["etiqueta"], r["id"] = "op%d" % c, "e:%d:%d" % (c, i)
                out.append(r)
        for i in range(30):   # opciones distintas en cada fila: no elegible
            r = fila(i, "choice", 3, tarea="mcq")
            r["opciones"] = ["a%d" % i, "b%d" % i, "c%d" % i]
            r["etiqueta"], r["descripciones"] = "a%d" % i, {}
            out.append(r)
        return out

    def test_elegibles_y_determinismo(self):
        e = DB.EpisodiosProto(self.filas(), lote=8, k=8, semilla=3)
        self.assertEqual(e.tareas, ["fija"])
        a, b = e.paso(5), e.paso(5)
        self.assertEqual([r["id"] for r in a["filas"]], [r["id"] for r in b["filas"]])
        self.assertNotEqual([r["id"] for r in a["filas"]], [r["id"] for r in e.paso(6)["filas"]])

    def test_soporte_de_8_por_clase_y_disjunto(self):
        e = DB.EpisodiosProto(self.filas(), lote=8, k=8)
        ep = e.paso(0)
        por = {}
        for r in ep["soporte"]:
            por.setdefault(r["etiqueta"], []).append(r["id"])
        self.assertEqual({c: len(v) for c, v in por.items()}, {"op0": 8, "op1": 8, "op2": 8})
        self.assertFalse({r["id"] for r in ep["filas"]} & {r["id"] for r in ep["soporte"]})

    def test_clases_recortadas_en_las_consultas(self):
        e = DB.EpisodiosProto(self.filas(), lote=8, k=8, vias=2)
        ep = e.paso(0)
        self.assertEqual(len({r["etiqueta"] for r in ep["soporte"]}), 2)
        self.assertTrue(all(len(r["opciones"]) == 2 for r in ep["filas"]))


class DeExtremoAExtremo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.d = Path(cls.dir.name)
        r10_sintetico.generar(cls.d)
        from model.alt.manifiesto import cargar
        destilar.aplicar(cargar(), cls.d / "raiz", str(cls.d / "blandas.jsonl.gz"), str(cls.d / "blandas.json"))
        cls.env = {k: os.environ.get(k) for k in ("CHISPA_E5", "HF_HUB_OFFLINE")}
        os.environ["CHISPA_E5"], os.environ["HF_HUB_OFFLINE"] = str(cls.d / "e5"), "1"
        a = DB.parser().parse_args(["entrenar", "--datos", str(cls.d / "raiz"), "--partida", str(cls.d / "partida.pt"), "--salida", str(cls.d / "out"),
                                    "--dev-dir", str(cls.d / "dev"), "--umbral-dup", "0.9999", "--pasos", "10", "--lote", "8", "--tope-tarea", "0",
                                    "--dev-max", "20", "--vias", "4", "--device", "cpu", "--descongelar", "1", "--log-cada", "5"])
        cls.logs = []
        cls.rc = DB.entrenar(a, log=cls.logs.append)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        cls.dir.cleanup()

    def test_entrena_y_deja_checkpoints_cargables(self):
        self.assertEqual(self.rc, 0)
        for f in ("final.pt", "mejor.pt", "historial.json", "datos.json"):
            self.assertTrue((self.d / "out" / f).exists(), f)
        from model.l2.evaluar import cargar_modelo
        m, ck = cargar_modelo(str(self.d / "out" / "final.pt"))
        self.assertEqual(ck["max_tok"], 48)
        self.assertEqual(ck["paso"], 10)
        self.assertTrue(ck["enc"])   # capa descongelada guardada

    def test_filtros_aplicados(self):
        d = json.loads((self.d / "out" / "datos.json").read_text())
        self.assertGreater(d["excluidas"]["filas"], 0)
        motivos = " ".join(d["excluidas"]["por_motivo"])
        self.assertIn("massive", motivos)
        self.assertIn("inyecciones", motivos)
        self.assertEqual(d["casi_duplicados"]["quitadas"], 1)
        self.assertEqual(d["con_blanda_entrenamiento"], d["filas_entrenamiento"])
        self.assertLess(d["filas_con_blanda"], d["filas_train_totales"])   # 9 de cada 10 tienen blanda

    def test_solo_blandas_y_sin_tareas_excluidas(self):
        h = json.loads((self.d / "out" / "historial.json").read_text())
        self.assertGreater(h["episodios"], 0)
        self.assertIn("coseno_e5", h)
        self.assertLessEqual(h["coseno_e5"]["query"], 1.0)

    def test_evaluacion_int8_de_los_dos_checkpoints(self):
        dev = EV.leer_dev(self.d / "dev")
        vocab = json.loads((self.d / "vocab-diminuto.json").read_text())
        rutas = sorted(str(p) for p in (self.d / "dev").glob("*.jsonl"))
        res = {}
        for nombre, ck in (("base", self.d / "partida.pt"), ("dest", self.d / "out" / "final.pt")):
            res[nombre] = EV.evaluar(str(ck), dev, vocab, None, 1, rutas, log=lambda s: None)
            self.assertEqual(sorted(res[nombre]["tareas"]), sorted(EV.tareas_dev_v2()))
            for t in res[nombre]["tareas"].values():
                self.assertEqual(sorted(t), ["k8", "zeroshot"])
                self.assertTrue(0.0 <= t["k8"]["macro_f1"] <= 1.0)
            self.assertGreater(res[nombre]["info"]["consulta_mb"], 0)
        crit = json.loads(CRIT.read_text())
        r = EV.comparar(res["base"], res["dest"], crit, set(crit["sinteticas"]), CRIT_SHA)
        self.assertIn(r["veredicto"], ("SIGUE", "PARA", "ENTRE_MEDIAS"))
        self.assertEqual(r["sin_sinteticas"]["zeroshot"]["tareas"], 4)
        self.assertIn("| 6 tareas | zeroshot |", EV.markdown(r))

    def test_leer_dev_rechaza_lo_que_no_es_dev(self):
        with tempfile.TemporaryDirectory() as t:
            for f in (self.d / "dev").glob("*.jsonl"):
                (Path(t) / f.name).write_text(f.read_text())
            (Path(t) / "unlu-consultas.jsonl").write_text("{}\n")   # una SELLADA
            with self.assertRaises(SystemExit):
                EV.leer_dev(t)
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "horeca-resenas.jsonl").write_text((self.d / "dev" / "horeca-resenas.jsonl").read_text())
            with self.assertRaises(SystemExit):   # faltan 5
                EV.leer_dev(t)


class Criterio(unittest.TestCase):
    def test_sha256_congelado(self):
        self.assertEqual(hashlib.sha256(CRIT.read_bytes()).hexdigest(), CRIT_SHA)
        self.assertEqual(EV.comprobar_criterio(CRIT, CRIT_SHA)["id"], "r10-piloto-destilacion-4b-biencoder")
        with self.assertRaises(SystemExit):
            EV.comprobar_criterio(CRIT, "0" * 64)
        with self.assertRaises(SystemExit):
            EV.comprobar_criterio(CRIT, "")

    def test_las_tareas_del_criterio_son_las_dev_de_v2(self):
        c = json.loads(CRIT.read_text())
        self.assertEqual(sorted(c["tareas"]), sorted(EV.tareas_dev_v2()))
        self.assertEqual(sorted(c["sinteticas"]), ["finanzas-documentos", "horeca-resenas"])

    def test_veredicto(self):
        c = json.loads(CRIT.read_text())
        self.assertEqual(EV.veredicto(c, 4.0, -1.0), "SIGUE")
        self.assertEqual(EV.veredicto(c, 6.2, 0.3), "SIGUE")
        self.assertEqual(EV.veredicto(c, 4.0, -1.01), "ENTRE_MEDIAS")
        self.assertEqual(EV.veredicto(c, 3.99, 2.0), "ENTRE_MEDIAS")
        self.assertEqual(EV.veredicto(c, 2.0, 0.0), "ENTRE_MEDIAS")
        self.assertEqual(EV.veredicto(c, 1.99, 5.0), "PARA")
        self.assertEqual(EV.veredicto(c, -3.0, 0.0), "PARA")

    def test_bootstrap_determinista(self):
        a = EV.bootstrap_tareas([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(a, EV.bootstrap_tareas([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        self.assertLess(a[0], 3.5)
        self.assertGreater(a[1], 3.5)
        self.assertEqual(EV.bootstrap_tareas([2.0] * 6), [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
