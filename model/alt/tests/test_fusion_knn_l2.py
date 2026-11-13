"""Pruebas de `model.alt.fusion_knn_l2` con datos de juguete: sin torch."""
import unittest

import numpy as np

from model.alt import fusion_knn_l2 as F
from model.alt.p0_medir import ic, ic_dif, mcnemar


def juguete(n_clases=3, n_tr=6, n_te=30, semilla=0, ruido=0.6):
    """Tres tareas: vectores normalizados en torno a un centro por clase; L2 de juguete ruidoso."""
    rng = np.random.RandomState(semilla)
    datos = {}
    for t in range(3):
        clases = ["c%d" % i for i in range(n_clases)]
        centros = rng.randn(n_clases, 8)

        def vec(y):
            v = centros[y] + ruido * rng.randn(8)
            return v / np.linalg.norm(v)

        ytr = [i % n_clases for i in range(n_clases * n_tr)]
        yte = [int(rng.randint(n_clases)) for _ in range(n_te)]
        ops = [clases] * n_te
        Pl = np.full((n_te, n_clases), 0.1)
        for i, y in enumerate(yte):
            Pl[i, y if rng.rand() < 0.6 else rng.randint(n_clases)] = 0.8
        Pl /= Pl.sum(1, keepdims=True)
        d = {"ids": ["t%d:%d" % (t, i) for i in range(n_te)], "ops": ops, "M": F.mascara(ops),
             "Xtr": np.array([vec(y) for y in ytr]), "Xte": np.array([vec(y) for y in yte]), "ytr": [clases[y] for y in ytr],
             "y": np.array(yte), "Pl": Pl, "Pc": Pl.copy()}
        d["Pk_guardado"] = F.knn(d["Xte"], d["Xtr"], d["ytr"], ops)[0]
        datos["tarea%d" % t] = d
    return datos


class Knn(unittest.TestCase):
    def test_muestreo_k_por_clase_y_semilla(self):
        ys = ["a"] * 5 + ["b"] * 2
        idx = F.muestrear(ys, 3, 1)
        self.assertEqual(sorted(ys[i] for i in idx), ["a"] * 3 + ["b"] * 2)   # b tiene menos de 3: entra entero
        self.assertTrue((F.muestrear(ys, 3, 1) == idx).all())
        self.assertEqual(len(F.muestrear(ys, None, 0)), 7)

    def test_igual_que_la_formula_de_referencia(self):
        d = juguete()["tarea0"]
        P, simv, _ = F.knn(d["Xte"], d["Xtr"], d["ytr"], d["ops"], vecinos=5, temp=0.05)
        q = d["Xte"][0]
        s = d["Xtr"] @ q
        idx = np.argsort(-s, kind="stable")[:5]
        w = np.exp((s[idx] - s[idx].max()) / 0.05)
        ref = dict.fromkeys(d["ops"][0], 0.0)
        for i, wi in zip(idx, w):
            ref[d["ytr"][i]] += wi
        tot = sum(ref.values())
        self.assertTrue(np.allclose(P[0], [ref[o] / tot for o in d["ops"][0]]))
        self.assertAlmostEqual(simv[0], s.max())

    def test_sin_soporte_es_uniforme(self):
        P, _, _ = F.knn(np.ones((1, 4)), np.zeros((0, 4)), [], [["a", "b"]])
        self.assertTrue(np.allclose(P, 0.5))


class Fusion(unittest.TestCase):
    def test_extremos_y_normalizacion(self):
        Pk = np.array([[0.9, 0.1, 0.0]])
        Pl = np.array([[0.1, 0.8, 0.1]])
        M = np.array([[True, True, True]])
        for modo in ("loglineal", "lineal"):
            self.assertEqual(F.top1(F.fusionar(Pk, Pl, M, 1.0, modo), M)[0], 0)
            self.assertEqual(F.top1(F.fusionar(Pk, Pl, M, 0.0, modo), M)[0], 1)
            self.assertAlmostEqual(F.fusionar(Pk, Pl, M, 0.5, modo).sum(), 1.0)

    def test_relleno_no_recibe_masa(self):
        M = np.array([[True, True, False]])
        P = F.fusionar(np.array([[0.5, 0.5, 0.0]]), np.array([[0.5, 0.5, 0.0]]), M, 0.5)
        self.assertEqual(P[0, 2], 0.0)

    def test_rasgos(self):
        M = np.array([[True, True]])
        X = F.rasgos(np.array([[0.7, 0.3]]), np.array([[0.9, 0.1]]), np.array([[0.2, 0.8]]), M, np.array([0.5]), np.array([[0.4, 0.1]]))
        self.assertEqual(X[0, 0], 0.0)                 # kNN y L2 discrepan
        self.assertAlmostEqual(X[0, 1], 0.7)
        self.assertAlmostEqual(X[0, 2], 0.4)
        self.assertAlmostEqual(X[0, 5], 0.4)           # centroide de la clase predicha (0)


class Escalado(unittest.TestCase):
    def test_lr_separa_una_variable_informativa(self):
        rng = np.random.RandomState(0)
        x = rng.randn(400)
        y = (x + 0.3 * rng.randn(400) > 0).astype(float)
        m = F.ajustar_lr(np.column_stack([x, rng.randn(400)]), y)
        p = F.predecir_lr(m, np.column_stack([x, rng.randn(400)]))
        self.assertGreater(F.auc(p, y), 0.9)

    def test_media_ic(self):
        m, h = F.media_ic([0.5, 0.5, 0.5])
        self.assertEqual((m, h), (0.5, 0.0))
        self.assertEqual(F.media_ic([0.4])[1], 0.0)


class Extremo(unittest.TestCase):
    def test_medir_y_resumen_de_juguete(self):
        datos = juguete()
        res = F.medir(datos, semillas=3, ks=(1, 2), verbose=False)
        self.assertEqual(set(res["k_elegido"]), set(datos))
        out = F.resumen(res, datos, ic, ic_dif, mcnemar)
        out["veredicto"] = F.veredicto(out)
        self.assertIn(out["veredicto"]["veredicto"], ("PASA", "REFUTADO", "no concluyente"))
        self.assertEqual(set(out["curva_k"]), {"1", "2", "todo"})
        self.assertTrue(0.0 <= out["tabla"]["agrupada"]["fusion"]["acc"] <= 1.0)
        self.assertEqual(out["verificacion_knn"]["agrupada_recalculada"], round(float(np.mean(np.concatenate(
            [(F.top1(d["Pk_guardado"], d["M"]) == d["y"]) for d in datos.values()]))), 4))

    def test_el_peso_no_se_ajusta_con_la_tarea_evaluada(self):
        datos = juguete()
        res = F.medir(datos, semillas=2, ks=(2,), verbose=False)
        w_tarea = res["_por_k"][2]["tarea0"]["w"][0]
        d = {c: v for c, v in datos.items() if c != "tarea0"}
        cfg = F.configuracion(d, 2, 0)
        acc = np.mean([[F.acierto(F.fusionar(cfg[c][0], d[c]["Pl"], d[c]["M"], w), d[c]).mean() for w in F.REJILLA_W] for c in d], axis=0)
        self.assertEqual(w_tarea, F.REJILLA_W[int(np.argmax(acc))])

    def test_falta_cobertura_de_ids(self):
        with self.assertRaises(ValueError):
            F.rellenar({"a": {"x": 1.0}}, ["a", "b"], [["x"], ["x"]], "l2/t")


if __name__ == "__main__":
    unittest.main()
