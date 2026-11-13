"""E2 (cabezas de score sobre el vector de K similitudes: coral, acumulados, grueso_fino; coral1 = la de siempre) y E3 (ruido de condicionamiento)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from model.evaluacion import metricas as M
from model.l2 import entrenar
from model.l2.evaluar import cargar_modelo
from model.l2.modelo import (CABEZAS_SCORE, CLASES_CABEZA, AcumuladosV, Codificador, Coral, CoralV, GruesoFinoV, ModeloL2, bloques_gruesos,
                             ruidificar_ids, texto_consulta, textos_torre_fila)
from model.l2.tests.falso import CodificadorFalso, fila


def probs_coral_a_mano(c, x):
    """La fórmula de siempre, sin pasar por la clase: P(y>j) = σ(w·x − θ_j)."""
    th = c.theta1 + torch.cat([torch.zeros(1), torch.cumsum(torch.nn.functional.softplus(c.delta), 0)])
    sup = torch.sigmoid(c.w * x.unsqueeze(-1) - th)
    uno = torch.ones_like(sup[:, :1])
    return (torch.cat([uno, sup], 1) - torch.cat([sup, 0 * uno], 1)).clamp(min=0)


def vector(n, k, semilla=0):
    return torch.randn(n, k, generator=torch.Generator().manual_seed(semilla))


class Cabezas(unittest.TestCase):
    X = torch.linspace(-2, 3, 41)

    def test_defecto_es_coral1_igual_que_antes(self):
        self.assertEqual(CABEZAS_SCORE[0], "coral1")
        m = ModeloL2()
        self.assertEqual(m.cabeza_score, "coral1")
        m.asegurar_cabeza("score5")
        self.assertIs(type(m.coral["score5"]), Coral)
        self.assertEqual({k.split(".", 2)[2] for k in m.state_dict() if k.startswith("coral.")}, {"w", "theta1", "delta"})
        c = m.coral["score5"]
        with torch.no_grad():
            c.w.fill_(3.3), c.theta1.fill_(0.4), c.delta.copy_(torch.tensor([0.1, -0.5, 1.0]))
        self.assertTrue(torch.equal(c(self.X), probs_coral_a_mano(c, self.X)))
        self.assertEqual(textos_torre_fila(fila(0, "score", k=5), "v1", "coral1"), [texto_consulta(fila(0, "score", k=5), "v1")])

    def test_textos_de_torre_por_nivel(self):
        r = fila(0, "score", k=4)
        ts = textos_torre_fila(r, "v1", "acumulados")
        self.assertEqual(len(ts), 4)
        self.assertEqual(len(set(ts)), 4)
        self.assertEqual(ts[-1], texto_consulta(r, "v1"))      # el último es el del nivel máximo de coral1
        self.assertEqual(textos_torre_fila(r, "v1", "coral"), ts)

    def test_formas_y_suma_uno(self):
        for tipo in ("coral", "acumulados", "grueso_fino"):
            for k in (2, 3, 4, 5, 7):
                torch.manual_seed(k)
                m = ModeloL2(cabeza_score=tipo)
                m.asegurar_cabeza("score%d" % k)
                h = m.coral["score%d" % k]
                self.assertIs(type(h), CLASES_CABEZA[tipo])
                with torch.no_grad():
                    for p in h.parameters():
                        p.add_(torch.randn_like(p))
                x = vector(41, k)
                p = h(x)
                self.assertEqual(p.shape, (41, k))
                self.assertTrue(torch.allclose(p.sum(1), torch.ones(41), atol=1e-5), (tipo, k))
                self.assertTrue(bool((p >= 0).all()))

    def test_acumulados_monotonia_forzada(self):
        for s in range(10):
            torch.manual_seed(s)
            h = AcumuladosV(6)
            with torch.no_grad():
                h.W.normal_(0, 4)        # pesos de cualquier signo: sin la fuerza de la monotonía las acumuladas se cruzarían
                h.theta.normal_(0, 3)
            x = vector(41, 6, s)
            sup = h.acumuladas(x)
            self.assertTrue(bool((sup[:, 1:] <= sup[:, :-1] + 1e-7).all()))
            p = h(x)
            self.assertTrue(bool((p >= 0).all()))
            self.assertTrue(torch.allclose(p.sum(1), torch.ones(41), atol=1e-5))
            self.assertTrue(torch.allclose(p.flip(1).cumsum(1).flip(1)[:, 1:], sup, atol=1e-5))   # la cola de la distribución es la acumulada

    def test_inits_de_vector_parten_de_coral1(self):
        """Con el init, coral y acumulados sobre el vector dan lo mismo que `Coral` sobre su último componente (el nivel máximo)."""
        x = vector(41, 5)
        base = Coral(5)(x[:, -1])
        self.assertTrue(torch.allclose(CoralV(5)(x), base, atol=1e-6))
        self.assertTrue(torch.allclose(AcumuladosV(5)(x), base, atol=1e-6))

    def test_grueso_fino_bloques_y_producto(self):
        self.assertEqual(bloques_gruesos(4), [2, 1, 1])
        self.assertEqual(bloques_gruesos(5), [2, 2, 1])
        self.assertEqual(bloques_gruesos(6), [2, 2, 2])
        self.assertEqual(bloques_gruesos(3), [1, 1, 1])
        self.assertEqual(bloques_gruesos(2), [1, 1])
        h = GruesoFinoV(5)
        with torch.no_grad():
            h.U.normal_(), h.v.normal_(), h.c.normal_(), h.A.normal_()
        x = vector(41, 5)
        p = h(x)
        pb = torch.softmax(x @ h.A.T + h.c, -1)
        z = x @ h.U.T + h.v
        esperado = torch.cat([pb[:, :1] * torch.softmax(z[:, 0:2], -1), pb[:, 1:2] * torch.softmax(z[:, 2:4], -1), pb[:, 2:3] * torch.softmax(z[:, 4:5], -1)], 1)
        self.assertTrue(torch.allclose(p, esperado, atol=1e-6))
        self.assertTrue(torch.allclose(p[:, 0:2].sum(1), pb[:, 0], atol=1e-6))   # la masa de cada bloque es P(bloque)
        self.assertTrue(torch.allclose(p[:, 4], pb[:, 2], atol=1e-6))

    def test_grueso_fino_con_mas_similitud_en_el_nivel_alto_sube_el_nivel(self):
        h = GruesoFinoV(5)
        medianas = []
        for v in (-3.0, 0.0, 3.0):
            x = torch.zeros(1, 5)
            x[0, -1] = v
            medianas.append(int(M.etiqueta_coral(h(x)[0].tolist())))
        self.assertEqual(medianas, sorted(medianas))
        self.assertGreater(medianas[-1], medianas[0])

    def test_forward_completo_y_gradiente(self):
        for tipo in ("coral", "acumulados", "grueso_fino"):
            torch.manual_seed(0)
            m, enc = ModeloL2(cabeza_score=tipo), CodificadorFalso()
            filas = [fila(0, "choice", 4), fila(1, "noul"), fila(2, "score", k=5), fila(3, "score", k=4)]
            probs, ps = m(filas, enc)
            for r, p in zip(filas, probs):
                self.assertEqual(p.shape, (len(r["opciones"]),))
                self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
            ps["total"].backward()
            for n, p in m.coral["score5"].named_parameters():
                self.assertIsNotNone(p.grad, (tipo, n))
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
            # la entrada son las K similitudes: el gradiente llega a los pesos de todos los niveles
            g = {"coral": "v", "acumulados": "W", "grueso_fino": "U"}[tipo]
            self.assertGreater(float(getattr(m.coral["score5"], g).grad[..., :-1].abs().sum()), 0)

    def test_perdida_grueso_fino_es_suma_de_las_dos_ce(self):
        h = GruesoFinoV(5)
        with torch.no_grad():
            h.c.normal_(), h.v.normal_(), h.U.normal_()
        x = vector(1, 5)
        nivel = 3        # 0-based: bloque 1 (niveles 2 y 3), posición 1 en el bloque
        p = h(x)[0]
        pb = torch.softmax(x[0] @ h.A.T + h.c, -1)
        pf = torch.softmax((x[0] @ h.U.T + h.v)[2:4], -1)
        self.assertAlmostEqual(float(-torch.log(p[nivel])), float(-torch.log(pb[1]) - torch.log(pf[1])), places=5)

    def test_mediana_igual_regla_en_todas(self):
        """La etiqueta evaluada sale de la distribución con la misma regla (`etiqueta_coral`) en cualquier cabeza."""
        for h in (CoralV(5), AcumuladosV(5), GruesoFinoV(5)):
            for x in vector(6, 5):
                p = h(x.unsqueeze(0))[0]
                self.assertEqual(M.etiqueta_coral(p.tolist()), int((torch.cumsum(p.flip(0), 0).flip(0)[1:] > 0.5).sum()))


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def ck_coral1(self):
        torch.manual_seed(1)
        m = ModeloL2()
        m.asegurar_cabeza("score4")
        with torch.no_grad():
            c = m.coral["score4"]
            c.w.fill_(2.5), c.theta1.fill_(0.3), c.delta.copy_(torch.tensor([0.4, -0.2]))
        ruta = self.dir / "m.pt"
        torch.save({"modelo": m.state_dict(), "cabezas": ["score4"], "paso": 3}, ruta)   # como los de antes de E2: sin `cabeza_score`
        return m, ruta

    def test_checkpoint_antiguo_carga_como_coral1(self):
        m0, ruta = self.ck_coral1()
        m, ck = cargar_modelo(str(ruta))
        self.assertEqual(m.cabeza_score, "coral1")
        self.assertNotIn("cabeza_score", ck)
        x = torch.linspace(-1, 2, 9)
        self.assertTrue(torch.equal(m.coral["score4"](x), m0.coral["score4"](x)))

    def test_coral1_a_coral_y_acumulados_conserva_la_funcion(self):
        m0, ruta = self.ck_coral1()
        ck = torch.load(ruta, weights_only=False)
        x = vector(9, 4)
        esperado = m0.coral["score4"](x[:, -1])
        for tipo in ("coral", "acumulados"):
            a = ModeloL2(cabeza_score=tipo)
            a.asegurar_cabeza("score4")
            a.load_state_dict(ck["modelo"])     # convierte las claves de coral1
            self.assertTrue(torch.allclose(a.coral["score4"](x), esperado, atol=1e-6), tipo)
            self.assertTrue(torch.equal(a.ranuras.q, m0.ranuras.q))

    def test_coral1_a_grueso_fino_carga_con_init_nuevo(self):
        m0, ruta = self.ck_coral1()
        g = ModeloL2(cabeza_score="grueso_fino")
        g.asegurar_cabeza("score4")
        ini = {k: v.clone() for k, v in g.coral["score4"].state_dict().items()}
        g.load_state_dict(torch.load(ruta, weights_only=False)["modelo"])
        for k, v in g.coral["score4"].state_dict().items():
            self.assertTrue(torch.equal(v, ini[k]))
        self.assertTrue(torch.equal(g.ranuras.q, m0.ranuras.q))   # el resto sí viene del checkpoint

    def test_ida_y_vuelta_con_el_tipo_en_el_checkpoint(self):
        for tipo in ("coral", "acumulados", "grueso_fino"):
            torch.manual_seed(2)
            m = ModeloL2(cabeza_score=tipo)
            m.asegurar_cabeza("score4")
            with torch.no_grad():
                for p in m.coral["score4"].parameters():
                    p.add_(torch.randn_like(p) * 0.5)
            ruta = self.dir / (tipo + ".pt")
            torch.save({"modelo": m.state_dict(), "cabezas": ["score4"], "cabeza_score": tipo}, ruta)
            m2, ck = cargar_modelo(str(ruta))
            self.assertEqual(m2.cabeza_score, tipo)
            x = vector(9, 4)
            self.assertTrue(torch.equal(m2.coral["score4"](x), m.coral["score4"](x)))

    def test_entrenar_guarda_el_tipo_y_parte_de_coral1(self):
        filas = [fila(i, "choice", 4, "a") for i in range(20)] + [fila(200 + i, "score", tarea="c", k=4) for i in range(12)]
        with open(self.dir / "d.jsonl", "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in filas))

        def correr(salida, *extra):
            enc = CodificadorFalso()
            with mock.patch.object(entrenar, "Codificador", lambda **kw: enc):
                return entrenar.main(["--salida", str(self.dir / salida), "--datos", str(self.dir / "d.jsonl"), "--pasos", "4", "--lote", "8",
                                      "--device", "cpu", "--ckpt-cada", "0", *extra])
        self.assertEqual(correr("base"), 0)
        base = torch.load(self.dir / "base" / "mejor.pt", weights_only=False)
        self.assertEqual(base["cabeza_score"], "coral1")
        for tipo in ("coral", "acumulados", "grueso_fino"):
            self.assertEqual(correr(tipo, "--desde", str(self.dir / "base" / "mejor.pt"), "--cabeza-score", tipo), 0)
            ck = torch.load(self.dir / tipo / "mejor.pt", weights_only=False)
            self.assertEqual(ck["cabeza_score"], tipo)
            self.assertEqual(torch.load(self.dir / tipo / "ultimo.pt", weights_only=False)["cabeza_score"], tipo)
            m, _ = cargar_modelo(str(self.dir / tipo / "mejor.pt"))
            self.assertEqual(m.cabeza_score, tipo)
        # sin la bandera, `--desde` sigue el tipo del checkpoint de partida
        self.assertEqual(correr("sigue", "--desde", str(self.dir / "grueso_fino" / "mejor.pt")), 0)
        self.assertEqual(torch.load(self.dir / "sigue" / "mejor.pt", weights_only=False)["cabeza_score"], "grueso_fino")

    def test_exportar_solo_admite_coral1(self):
        from model.l2.calibrar import cabeza_inicial
        for tipo in ("coral", "acumulados", "grueso_fino"):
            m = ModeloL2(cabeza_score=tipo)
            m.asegurar_cabeza("score4")
            with self.assertRaises(ValueError):
                cabeza_inicial(m, "score4")
        c = ModeloL2()
        c.asegurar_cabeza("score4")
        self.assertEqual(set(cabeza_inicial(c, "score4")), {"a", "b"})


class Ruido(unittest.TestCase):
    ESP = [0, 1, 2, 3, 99]     # <s>, <pad>, </s>, <unk>, <mask> de mentira
    VOCAB, MASK = 100, 99

    def entrada(self):
        ids = torch.randint(4, 98, (6, 30))
        ids[:, 0] = 0
        ids[:, -1] = 2
        mascara = torch.ones_like(ids)
        mascara[3:, 20:] = 0
        ids[3:, 20:] = 1
        prot = torch.zeros_like(ids, dtype=torch.bool)
        prot[:, 1:4] = True      # prefijo
        return ids, mascara, prot

    def test_p_cero_no_cambia_nada(self):
        ids, m, prot = self.entrada()
        self.assertTrue(torch.equal(ruidificar_ids(ids, m, prot, 0.0, self.VOCAB, self.MASK, self.ESP, torch.Generator().manual_seed(0)), ids))

    def test_no_toca_especiales_prefijo_ni_relleno_y_mezcla_mitad_mitad(self):
        ids, m, prot = self.entrada()
        g = torch.Generator().manual_seed(0)
        sal = ruidificar_ids(ids, m, prot, 0.5, self.VOCAB, self.MASK, self.ESP, g)
        cambiado = sal != ids
        self.assertFalse(bool(cambiado[:, 0].any() or cambiado[:, -1].any()))
        self.assertFalse(bool(cambiado[prot].any()))
        self.assertFalse(bool(cambiado[m == 0].any()))
        self.assertTrue(bool(sal.max() <= 99) and bool(sal.min() >= 0))
        elegibles = (m.bool() & ~prot & ~torch.isin(ids, torch.tensor(self.ESP))).sum()
        self.assertTrue(0.35 < cambiado.sum() / elegibles < 0.65)
        # entre lo cambiado: <mask> vs token al azar (nunca especial): ~ mitad y mitad
        nuevos = sal[cambiado]
        frac_mask = (nuevos == self.MASK).float().mean()
        self.assertTrue(0.35 < frac_mask < 0.65)
        self.assertFalse(bool(torch.isin(nuevos[nuevos != self.MASK], torch.tensor(self.ESP)).any()))

    def test_p_uno_cambia_todo_lo_elegible_y_es_reproducible(self):
        ids, m, prot = self.entrada()
        a = ruidificar_ids(ids, m, prot, 1.0, self.VOCAB, self.MASK, self.ESP, torch.Generator().manual_seed(3))
        b = ruidificar_ids(ids, m, prot, 1.0, self.VOCAB, self.MASK, self.ESP, torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(a, b))
        elegible = m.bool() & ~prot & ~torch.isin(ids, torch.tensor(self.ESP))
        self.assertTrue(bool((a[~elegible] == ids[~elegible]).all()))
        self.assertGreater(float((a[elegible] != ids[elegible]).float().mean()), 0.95)

    def test_no_usa_el_azar_global(self):
        ids, m, prot = self.entrada()
        torch.manual_seed(5)
        x = torch.rand(1)
        torch.manual_seed(5)
        ruidificar_ids(ids, m, prot, 0.5, self.VOCAB, self.MASK, self.ESP, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(torch.rand(1), x))


class CacheDiscoCodificador(unittest.TestCase):
    """La caché en disco devuelve lo mismo (en fp16) en la ejecución que la llena y en las que la leen, sin volver a pasar por e5."""

    def test_segunda_ejecucion_lee_lo_mismo_sin_e5(self):
        import tempfile
        try:
            Codificador(max_tok=48)
        except Exception as e:      # sin e5 en la máquina
            raise unittest.SkipTest("sin e5: %s" % e)
        textos, ops = ["un mensaje de prueba", "otro mensaje algo más largo"], ["passage: opción uno", "passage: opción dos"]
        with tempfile.TemporaryDirectory() as d:
            a = Codificador(max_tok=48, cache_disco=d)
            H1, m1 = a.tokens(textos)
            v1 = a.pasaje(ops)
            b = Codificador(max_tok=48, cache_disco=d)
            b._pase = lambda *_: (_ for _ in ()).throw(AssertionError("no debería pasar por e5"))
            H2, m2 = b.tokens(textos)
            v2 = b.pasaje(ops)
            self.assertTrue(torch.equal(H1, H2) and torch.equal(m1, m2) and torch.equal(v1, v2))
            self.assertEqual(H1.dtype, torch.float32)
        with self.assertRaises(ValueError):
            Codificador(max_tok=48, cache_disco="x", descongelar=1)


class RuidoCodificador(unittest.TestCase):
    """Con el tokenizador de e5 real (como `test_largos.ConModelo`)."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.enc = Codificador(max_tok=48)
        except Exception as e:      # sin e5 en la máquina
            raise unittest.SkipTest("sin e5: %s" % e)

    TEXTOS = ["un mensaje de prueba bastante largo para que haya muchos tokens que ensuciar, con varias palabras distintas", "otro mensaje"]
    OPS = ["passage: ¿Cuál? [SEP] opción uno [SEP] descripción de la opción", "passage: ¿Cuál? [SEP] dos [SEP] otra descripción"]

    def salidas(self):
        H, m = self.enc.tokens(self.TEXTOS)
        return H.clone(), m.clone(), self.enc.pasaje(self.OPS).clone()

    def test_noop_con_p_cero_y_en_inferencia(self):
        self.enc.configurar_ruido(0.0)
        base = self.salidas()
        with self.enc.con_ruido():
            con_cero = self.salidas()
        for a, b in zip(base, con_cero):
            self.assertTrue(torch.equal(a, b))
        self.enc.configurar_ruido(0.5, 1)          # con P > 0 pero fuera de `con_ruido` (evaluación / inferencia): igual
        fuera = self.salidas()
        for a, b in zip(base, fuera):
            self.assertTrue(torch.equal(a, b))

    def test_con_ruido_cambia_y_deja_la_cache_limpia(self):
        self.enc.configurar_ruido(0.0)
        base = self.salidas()
        self.enc.configurar_ruido(0.5, 1)
        with self.enc.con_ruido():
            H, _, v = self.salidas()
        self.assertFalse(torch.equal(H, base[0]))
        self.assertFalse(torch.equal(v, base[2]))
        # lo ensuciado no se coló en la caché de vectores de opción
        self.assertTrue(torch.equal(self.enc.pasaje(self.OPS), base[2]))

    def test_prefijo_intacto(self):
        self.enc.configurar_ruido(0.999, 2)
        textos = ["query: hola mundo cruel", "passage: hola mundo cruel"]
        b = self.enc.tok(textos, padding=True, return_tensors="pt")
        sucio = self.enc._ensuciar({k: v.clone() for k, v in b.items()}, textos)
        n_q = len(self.enc.tok("query: ", add_special_tokens=False)["input_ids"])
        n_p = len(self.enc.tok("passage: ", add_special_tokens=False)["input_ids"])
        self.assertTrue(torch.equal(sucio["input_ids"][0, :1 + n_q], b["input_ids"][0, :1 + n_q]))
        self.assertTrue(torch.equal(sucio["input_ids"][1, :1 + n_p], b["input_ids"][1, :1 + n_p]))
        self.assertEqual(int(sucio["input_ids"][0, -1]), self.enc.tok.sep_token_id)
        self.assertFalse(torch.equal(sucio["input_ids"], b["input_ids"]))
        self.enc.configurar_ruido(0.0)


class RuidoEntrenar(unittest.TestCase):
    def test_p_cero_da_el_mismo_entrenamiento_que_sin_la_bandera(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            filas = [fila(i, "choice", 4, "a") for i in range(20)] + [fila(100 + i, "score", tarea="c", k=4) for i in range(12)]
            (d / "d.jsonl").write_text("".join(json.dumps(r) + "\n" for r in filas))

            def correr(salida, *extra):
                torch.manual_seed(0)
                enc = CodificadorFalso()
                with mock.patch.object(entrenar, "Codificador", lambda **kw: enc):
                    self.assertEqual(entrenar.main(["--salida", str(d / salida), "--datos", str(d / "d.jsonl"), "--pasos", "4", "--lote", "8",
                                                    "--device", "cpu", "--ckpt-cada", "0", *extra]), 0)
                return torch.load(d / salida / "ultimo.pt", weights_only=False)["modelo"]
            a, b = correr("a"), correr("b", "--ruido-cond", "0")
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), k)


if __name__ == "__main__":
    unittest.main()
