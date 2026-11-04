import gzip
import json
import tempfile
import unittest
from pathlib import Path

from model.meta import solape_jevbench as S

LARGO = "The quick brown fox jumps over the lazy dog while the farmer counts twelve sheep near the old stone bridge today"   # 21 palabras


def jev(i, texto):
    return {"id": i, "texto": texto}


def train(i, texto, split="train"):
    return {"id": i, "texto": texto, "split": split}


class Normalizar(unittest.TestCase):
    def test_normalizar(self):
        self.assertEqual(S.normalizar("  Hola,   MUNDO!!\n¿Qué tal? "), "hola mundo qué tal")
        self.assertEqual(S.normalizar('{"a": "B-c"}'), "a b c")

    def test_shingles(self):
        self.assertEqual(S.shingles("a b c", 13), set())
        self.assertEqual(len(S.shingles(" ".join("w%d" % i for i in range(20)), 13)), 8)


class Solape(unittest.TestCase):
    def test_exacto_casi_y_limpio(self):
        casos = [jev("e", LARGO), jev("c", LARGO + " and then everything changed completely"), jev("l", "Algo totalmente distinto en el tren de la tarde con mucha gente cansada que vuelve a casa luego del trabajo"),
                 jev("corto", "hola mundo")]
        filas = [
            train("t1", LARGO.upper() + "!!"),                                    # exacto de e tras normalizar (y casi de c)
            train("t2", "prefijo distinto " + LARGO + " sufijo distinto"),          # casi duplicado de e y c, no exacto
            train("t3", "hola, mundo"),                                            # exacto del corto (no tiene shingles)
            train("t4", "nada que ver con nada " * 8),
            train("t5", LARGO, split="dev"),                                      # otro split: se ignora
        ]
        r = S.solape(casos, (f for f in filas if f["split"] == "train"))
        self.assertEqual(r["filas_train"], 4)
        self.assertEqual(r["exactos"]["ids"], ["corto", "e"])
        self.assertEqual(r["exactos"]["detalle"]["e"]["filas_train"], 1)
        self.assertEqual(r["casi_duplicados"]["ids"], ["c", "e"])
        self.assertEqual(r["sin_shingles"]["ids"], ["corto"])
        self.assertNotIn("l", r["cualquiera"].get("ids", []))
        self.assertEqual(r["cualquiera"]["casos"], 3)

    def test_streaming_desde_ficheros_y_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "ts").mkdir()
            (d / "ts" / "a.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [train("t1", LARGO), train("t2", "otra cosa", "dev")]))
            with gzip.open(d / "ts" / "b.jsonl.gz", "wt") as f:
                f.write(json.dumps(train("t3", "z " + LARGO)) + "\n")
            vistas = list(S.filas_train(d / "ts"))
            self.assertEqual([x["id"] for x in vistas], ["t1", "t3"])
            jb = d / "jb"
            (jb / "datasets" / "public").mkdir(parents=True)
            caso = {"id": "easy-x-00", "family": "intent", "group": None, "split": "public", "state": LARGO, "labels": ["a", "b"], "expected": "a",
                    "question": {"type": "choice", "instructions": "¿?", "criteria": {"a": "x", "b": "y"}}, "provenance": {"license": "MIT"}}
            (jb / "datasets" / "public" / "easy.jsonl").write_text(json.dumps(caso) + "\n")
            self.assertEqual(S.main(["--tsjev", str(d / "ts"), "--jevbench", str(jb), "--salida", str(d / "sal")]), 0)
            js = json.loads((d / "sal" / "solape.json").read_text())
            self.assertEqual(js["exactos"]["ids"], ["jev:public:easy-x-00"])
            self.assertEqual(js["casi_duplicados"]["casos"], 1)
            self.assertEqual(js["filas_train"], 2)
            self.assertIn("exacto", (d / "sal" / "solape.md").read_text())


if __name__ == "__main__":
    unittest.main()
