import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from model.l2 import tasa_unk as U

PALABRAS = ["<s>", "<pad>", "</s>", "<unk>", "query", ":", "hola", "mundo", "casa", "perro", "gato", "ñandú", "secretoXYZ", "cliente",
            "αβγ", "dia", "noche", "sol", "luna", "mar"]


def tokenizador(dir_):
    t = Tokenizer(models.WordLevel({p: i for i, p in enumerate(PALABRAS)}, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    ruta = Path(dir_) / "tokenizer.json"
    t.save(str(ruta))
    return ruta


def vocab_json(dir_, conservar):
    ruta = Path(dir_) / "vocab.json"
    ruta.write_text(json.dumps({"modelo": "t", "original": len(PALABRAS), "conservados": len(conservar), "unk": 3, "ids_originales": sorted(conservar)}))
    return ruta


def jsonl(ruta, textos, campo="texto"):
    Path(ruta).write_text("".join(json.dumps({campo: t, "otro": "NO-LEER"}) + "\n" for t in textos))


class TasaUnk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.tok = tokenizador(self.d)
        # conservados: especiales + palabras comunes; fuera: ñandú, secretoXYZ, cliente, αβγ y las de 14 en adelante
        self.voc = vocab_json(self.d, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        (self.d / "corpus").mkdir()
        jsonl(self.d / "corpus" / "limpio.jsonl", ["hola mundo", "casa perro gato"] * 5)
        jsonl(self.d / "corpus" / "sucio.jsonl", ["hola secretoXYZ", "ñandú cliente αβγ", "casa"] * 4)
        with gzip.open(self.d / "corpus" / "comprimido.jsonl.gz", "wt") as f:
            f.write(json.dumps({"texto": "hola mundo secretoXYZ"}) + "\n" + "esto no es json\n")

    def correr(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = U.main(["--textos", str(self.d / "corpus"), "--vocab", str(self.voc), "--tokenizador", str(self.tok), "--max-tok", "32", *extra])
        self.assertEqual(rc, 0)
        return out.getvalue(), err.getvalue()

    def test_cuentas_a_mano(self):
        salida = self.d / "r.json"
        self.correr("--salida", str(salida))
        r = json.loads(salida.read_text())
        # limpio: «query : hola mundo» = 4 tokens y «query : casa perro gato» = 5 (el tokenizador de prueba no añade especiales); ninguno fuera
        l = r["corpus"]["limpio.jsonl"]
        self.assertEqual((l["textos"], l["tokens_podados"], l["tipos_podados"]), (10, 0, 0))
        self.assertEqual(l["tokens"], 5 * 4 + 5 * 5)
        # sucio: «hola secretoXYZ» 1 podado de 4; «ñandú cliente αβγ» 3 de 5; «casa» 0 de 3 (×4)
        s = r["corpus"]["sucio.jsonl"]
        self.assertEqual((s["textos"], s["tokens"], s["tokens_podados"], s["textos_con_unk"]), (12, 4 * (4 + 5 + 3), 4 * 4, 8))
        self.assertAlmostEqual(s["frac_unk"], 16 / 48)
        self.assertEqual(s["tipos_podados"], 4)
        self.assertEqual(r["total"]["tokens"], l["tokens"] + s["tokens"] + r["corpus"]["comprimido.jsonl.gz"]["tokens"])
        self.assertEqual(r["por_encima_del_tope"], ["comprimido.jsonl.gz", "sucio.jsonl"])    # > 2 %

    def test_no_imprime_ni_guarda_textos_ni_ids(self):
        out, err = self.correr("--anonimizar", "--salida", str(self.d / "r.json"))
        todo = out + err + (self.d / "r.json").read_text()
        for secreto in ("secretoXYZ", "ñandú", "cliente", "αβγ", "NO-LEER", "limpio", "sucio"):
            self.assertNotIn(secreto, todo)
        self.assertIn("corpus_1", out)
        self.assertIn("no son JSON", err)                  # solo el recuento de líneas rotas

    def test_campos_y_prefijo(self):
        jsonl(self.d / "corpus" / "limpio.jsonl", ["hola"], campo="pregunta")
        salida = self.d / "r.json"
        self.correr("--campos", "pregunta", "--prefijo", "", "--salida", str(salida))
        l = json.loads(salida.read_text())["corpus"]["limpio.jsonl"]
        self.assertEqual((l["textos"], l["tokens"]), (1, 1))       # solo «hola»: sin prefijo y sin especiales

    def test_barrido_y_recomendacion(self):
        salida = self.d / "r.json"
        self.correr("--barrido", "11", "14", "20", "--salida", str(salida), "--tope", "0.05")
        r = json.loads(salida.read_text())
        self.assertTrue(r["barrido_en_muestra"])
        fr = {b["n"]: b["peor_frac_unk"] for b in r["barrido"]}
        self.assertGreaterEqual(fr[11], fr[14])
        self.assertEqual(fr[20], 0.0)                              # todo el vocabulario
        self.assertGreater(fr[11], 0.05)
        self.assertEqual(r["recomendacion"]["n"], 14 if fr[14] <= 0.05 else 20)
        self.assertEqual(r["barrido"][0]["tabla_embeddings_mb"]["fp32"], 11 * 384 * 4 / 1e6)
        self.assertIn("optimista", r["recomendacion"]["texto"])

    def test_barrido_con_corpus_de_poda_aparte(self):
        poda = self.d / "poda.jsonl"
        jsonl(poda, ["hola mundo casa", "perro gato"])
        salida = self.d / "r.json"
        self.correr("--barrido", "11", "--textos-poda", str(poda), "--salida", str(salida))
        r = json.loads(salida.read_text())
        self.assertFalse(r["barrido_en_muestra"])
        self.assertNotIn("optimista", r["recomendacion"]["texto"])

    def test_vocab_de_tamano_rellena_con_latinas_por_id(self):
        tok = Tokenizer.from_file(str(self.tok))
        v = U.vocab_de_tamano(tok, 12, {5: 3, 9: 1}, [0, 1, 2, 3], len(PALABRAS))
        self.assertTrue({0, 1, 2, 3, 5, 9} <= v)
        self.assertEqual(len(v), 12)
        self.assertNotIn(14, v)            # «αβγ» no es latina y entra la última
        self.assertEqual(U.recomendar([{"n": 5, "peor_frac_unk": 0.5}], 0.02)[0], None)


if __name__ == "__main__":
    unittest.main()
