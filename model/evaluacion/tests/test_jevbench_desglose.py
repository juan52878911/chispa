import json
import tempfile
import unittest
from pathlib import Path

from model.evaluacion import jevbench as J
from model.evaluacion import jevbench_desglose as DG

FIX = Path(__file__).parent / "fixtures" / "jevbench"


def fila(i, familia, split, tipo="choice", etq="a", ops=("a", "b")):
    return {"id": i, "tarea": familia, "tipo": tipo, "opciones": list(ops), "etiqueta": etq, "fuente": "JevBench " + split}


FILAS = [
    fila("c1", "intent", "easy"), fila("c2", "intent", "easy"),
    fila("t1", "trap", "hard"), fila("t2", "trap", "hard"),
    fila("a1", "adversarial", "hard"), fila("a2", "adversarial", "original"),
    fila("s1", "ordinal", "hard", "score", "3", ("1", "2", "3", "4")),   # oro nivel 3
    fila("s2", "ordinal", "easy", "score", "1", ("1", "2", "3", "4")),   # oro nivel 1
]


def pred(aciertos_choice, niv1=3, niv2=1):
    """probs de FILAS: `aciertos_choice` = ids de choice acertados; en score, la distribución concentrada en el nivel dado."""
    out = {}
    for r in FILAS:
        if r["tipo"] == "choice":
            bien = r["id"] in aciertos_choice
            out[r["id"]] = {"id": r["id"], "probs": {"a": 0.9 if bien else 0.1, "b": 0.1 if bien else 0.9}}
        else:
            n = niv1 if r["id"] == "s1" else niv2
            out[r["id"]] = {"id": r["id"], "probs": {o: 1.0 if int(o) == n else 0.0 for o in r["opciones"]}}
    return out


class Desglose(unittest.TestCase):
    def test_cifras(self):
        d = DG.desglose(FILAS, pred({"c1", "t1", "a1", "a2"}, niv1=1))      # s1 predice 1 (oro 3), s2 acierta
        self.assertEqual(d["global"]["n"], 8)
        self.assertAlmostEqual(d["global"]["acc"], 5 / 8)                    # c1 t1 a1 a2 + s2
        lo, hi = d["global"]["ic95"]
        self.assertTrue(lo <= 5 / 8 <= hi)
        self.assertEqual(d["por_categoria"]["intent"], {"n": 2, "acc": 0.5})
        self.assertEqual(d["por_categoria"]["trap"], {"n": 2, "acc": 0.5})
        self.assertEqual(d["por_categoria"]["adversarial"], {"n": 2, "acc": 1.0})
        self.assertEqual(d[DG.COMBINADA], {"n": 4, "acc": 0.75})
        self.assertEqual(d["por_split"]["easy"]["n"], 3)
        self.assertEqual(d["por_split"]["hard"]["n"], 4)
        self.assertEqual(d["por_split"]["original"], {"n": 1, "acc": 1.0})
        self.assertEqual(d["mae_score"], {"n": 2, "mae": 1.0})              # |1−3| = 2 y |1−1| = 0

    def test_ic95_determinista(self):
        v = [1.0] * 30 + [0.0] * 20
        self.assertEqual(DG.ic95(v), DG.ic95(v))
        m, lo, hi = DG.ic95(v)
        self.assertAlmostEqual(m, 0.6)
        self.assertTrue(0.45 < lo < 0.6 < hi < 0.75)
        self.assertEqual(DG.ic95([]), (None, None, None))

    def test_sin_respuesta_cuenta_como_fallo(self):
        p = pred(set())
        p["c1"] = {"id": "c1", "probs": {}}
        p["s1"] = {"id": "s1", "probs": {}}
        d = DG.desglose(FILAS, p)
        self.assertEqual(d["por_categoria"]["intent"]["acc"], 0.0)
        self.assertEqual(d["mae_score"]["mae"], (2 + 0) / 2)                 # sin respuesta en s1: el extremo más lejano al oro (3 → 1)

    def test_semillas_y_base(self):
        base = [pred({"c1", "t1"}), pred({"c1", "t1"}), pred({"c1", "t1"})]
        mejor = [pred({"c1", "c2", "t1", "t2", "a1"}), pred({"c1", "c2", "t1", "t2", "a2"}), pred({"c1", "c2", "t1", "t2", "a1", "a2"})]
        peor = [pred({"c1"}), pred(set()), pred({"c1", "t1", "t2", "a1", "a2"})]
        res = DG.agrupar(FILAS, {"base": base, "mejor": mejor, "peor": peor}, "base")
        self.assertEqual(res["base"]["n_semillas"], 3)
        md = res["mejor"]["media_desv"]["global"]
        accs = [(5 + 2) / 8, (5 + 2) / 8, (6 + 2) / 8]                       # 5 de choice + 2 score; el tercero, 6 + 2
        self.assertAlmostEqual(md["media"], sum(accs) / 3)
        self.assertGreater(md["desv"], 0)
        self.assertTrue(res["mejor"]["vs_base"]["global"])                   # media − desv por encima del base
        self.assertFalse(res["peor"]["vs_base"]["global"])
        self.assertNotIn("vs_base", res["base"])
        self.assertEqual(res["base"]["media_desv"]["global"]["desv"], 0.0)

    def test_una_semilla_sin_desviacion(self):
        res = DG.agrupar(FILAS, {"a": [pred({"c1"})]}, "a")
        self.assertEqual(res["a"]["media_desv"]["global"]["desv"], 0.0)

    def test_fixture_real_y_cli(self):
        filas = J.cargar_jevbench(FIX)
        self.assertEqual({DG.split_de(r) for r in filas}, {"easy", "hard", "original"})
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "oraculo.jsonl"
            with open(ruta, "w") as f:
                for r in filas:    # oráculo: toda la masa en el oro
                    f.write(json.dumps({"id": r["id"], "probs": {o: float(o == r["etiqueta"]) for o in r["opciones"]}}) + "\n")
            self.assertEqual(DG.main(["--jevbench", str(FIX), "--grupo", "oro=%s,%s" % (ruta, ruta), "--grupo", "otro=%s" % ruta,
                                      "--salida", str(Path(d) / "sal")]), 0)
            js = json.loads((Path(d) / "sal" / "desglose.json").read_text())
            self.assertEqual(js["grupos"]["oro"]["media_desv"]["global"], {"media": 1.0, "desv": 0.0, "n_semillas": 2})
            self.assertEqual(js["grupos"]["otro"]["semillas"][0]["mae_score"]["mae"], 0.0)
            self.assertIn("trap+adversarial", (Path(d) / "sal" / "desglose.md").read_text())


if __name__ == "__main__":
    unittest.main()
