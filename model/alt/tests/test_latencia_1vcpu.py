"""Prueba de `model/alt/aws/latencia_1vcpu/bench.py` con un `.chispa` v3 sintético y diminuto, construido en un directorio temporal.

Necesita numpy, onnx, onnxruntime y tokenizers; sin alguno de ellos las pruebas se saltan. No baja nada ni escribe en el repo.
"""
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    import onnxruntime  # noqa: F401
    import tokenizers
    from onnx import TensorProto, helper, numpy_helper
    HAY = True
except ImportError:
    HAY = False

RUTA = Path(__file__).resolve().parents[1] / "aws" / "latencia_1vcpu" / "bench.py"
D, K = 8, 4


def bench():
    spec = importlib.util.spec_from_file_location("bench_latencia_1vcpu", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def onnx_diminuto(salida, vocab, rng):
    """ids (B,T) -> media de embeddings (B,D); con `ranuras` lo multiplica por (K,D) y da (B,K,D)."""
    emb = numpy_helper.from_array(rng.standard_normal((vocab, D)).astype(np.float32), "emb")
    nodos = [helper.make_node("Gather", ["emb", "input_ids"], ["g"]), helper.make_node("ReduceMean", ["g"], ["m"], axes=[1], keepdims=0)]
    inits = [emb]
    if salida == "S":
        inits.append(numpy_helper.from_array(rng.standard_normal((K, D)).astype(np.float32), "w"))
        nodos += [helper.make_node("Unsqueeze", ["m", "eje"], ["mu"]), helper.make_node("Mul", ["mu", "w"], [salida])]
        inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "eje"))
        forma = ["b", K, D]
    else:
        nodos.append(helper.make_node("Identity", ["m"], [salida]))
        forma = ["b", D]
    g = helper.make_graph(nodos, "g", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "t"]),
                                       helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "t"])],
                          [helper.make_tensor_value_info(salida, TensorProto.FLOAT, forma)], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    return m.SerializeToString()


def tokenizador(dir_tmp, palabras):
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {x: i for i, x in enumerate(["<s>", "<pad>", "</s>", "<unk>"] + sorted(set(palabras)))}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    ruta = Path(dir_tmp) / "tok.json"
    t.save(str(ruta))
    return str(ruta), len(vocab)


def l0tf_bytes(palabras):
    grams = sorted({(" " + p + " ")[i:i + n] for p in palabras for n in range(2, 6) for i in range(len(p) + 3 - n)})
    V, C = len(grams), 3
    rng = np.random.RandomState(1)
    etiquetas = b"".join(struct.pack("<H", len(e)) + e for e in (b"a", b"b", b"c"))
    blob = "".join(grams).encode("utf-8")
    offs, p = [0], 0
    for g in grams:
        p += len(g.encode("utf-8"))
        offs.append(p)
    return (struct.pack("<IIII", 1, V, C, 1) + etiquetas + rng.standard_normal(C).astype("<f4").tobytes() + (1 + rng.rand(V)).astype("<f4").tobytes()
            + np.asarray(offs, dtype="<u4").tobytes() + blob + rng.standard_normal(V * C).astype("<f4").tobytes())


def l1_bytes(vocab, dim=8):
    rng = np.random.RandomState(2)
    pesos = rng.randint(-127, 128, (vocab, dim)).astype(np.int8).tobytes()
    escalas = (rng.rand(vocab) / 127).astype("<f4").tobytes()
    cab = bytearray(256)
    cab[:4] = b"CHL1"
    struct.pack_into("<III", cab, 8, 1, dim, vocab)
    struct.pack_into("<QQ", cab, 36 + 16 * 6, 256, len(pesos))
    struct.pack_into("<QQ", cab, 36 + 16 * 7, 256 + len(pesos), len(escalas))
    return bytes(cab) + pesos + escalas


def chispa_v3(secs):
    """secs: {tipo: bytes}. Mismo contenedor que `format/chispa3`: cabecera de 32 B, entradas de 56 B, datos alineados a 16."""
    tipos = sorted(secs)
    datos, entradas, off = b"", b"", 0
    for t in tipos:
        pad = (-off) % 16
        datos += b"\0" * pad
        off += pad
        entradas += struct.pack("<IIQQ", t, 0, off, len(secs[t])) + b"\0" * 32
        datos += secs[t]
        off += len(secs[t])
    cab = bytearray(32)
    cab[:8] = b"CHISPA3\0"
    struct.pack_into("<H", cab, 8, 3)
    struct.pack_into("<I", cab, 12, len(tipos))
    struct.pack_into("<Q", cab, 16, 32 + len(entradas) + len(datos))
    return bytes(cab) + entradas + datos


@unittest.skipUnless(HAY, "faltan numpy, onnx, onnxruntime o tokenizers")
class TestLatencia1vCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = bench()
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        palabras = cls.m.PALABRAS + ["query:", "passage:", "·"]
        tok, vocab = tokenizador(d, palabras)
        rng = np.random.RandomState(0)
        torr = struct.pack("<IIII", 1, D, 32, 9) + b"passage: " + onnx_diminuto("h", vocab, rng)
        cls.chispa = d / "modelo-int8.chispa"
        cls.chispa.write_bytes(chispa_v3({1: b"{}", 3: l1_bytes(vocab), 4: onnx_diminuto("S", vocab, rng), 8: l0tf_bytes(cls.m.PALABRAS[:40]), 9: torr}))
        Path(d / "tokenizer-e5.json").write_text(Path(tok).read_text(encoding="utf-8"), encoding="utf-8")
        Path(d / "tokenizer-potion.json").write_text(Path(tok).read_text(encoding="utf-8"), encoding="utf-8")
        cls.d = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def corre(self, *extra, dir_=None):
        salida = Path(self.tmp.name) / ("salida-%d.json" % len(list(Path(self.tmp.name).glob("salida-*.json"))))
        rc = self.m.main(["--dir", str(dir_ or self.d), "--salida", str(salida), "--opciones", "3", *extra])
        return rc, json.loads(salida.read_text(encoding="utf-8"))

    def test_secciones(self):
        mm = memoryview(self.chispa.read_bytes())
        self.assertEqual(sorted(self.m.secciones_chispa(mm)), [1, 3, 4, 8, 9])
        with self.assertRaises(ValueError):
            self.m.secciones_chispa(memoryview(b"\0" * 40))

    def test_caliente_mide_todas_las_capas(self):
        rc, r = self.corre("--fase", "caliente", "--n", "30", "--calentar", "3")
        self.assertEqual(rc, 0)
        self.assertEqual(r["omitidos"], {})
        c = r["caliente"]
        for capa in ("L0", "L1", "L2", "torres", "cadena", "cadena_zero_shot"):
            self.assertEqual(c[capa]["n"], 30, capa)
            self.assertLessEqual(c[capa]["p50_ms"], c[capa]["p99_ms"])
        self.assertGreaterEqual(c["cadena"]["p50_ms"], c["L2"]["p50_ms"])
        self.assertGreater(r["rss_pico_mb"], 0)

    def test_frio(self):
        rc, r = self.corre("--fase", "frio")
        self.assertEqual(rc, 0)
        self.assertIn("L2", r["primera_peticion_ms"])
        self.assertGreater(r["tiempo_hasta_primera_respuesta_desde_arranque_proceso_ms"], 0)

    def test_capa_ausente_se_omite_sin_fallar(self):
        d = Path(self.tmp.name) / "sin_tok"
        d.mkdir()
        (d / "modelo-int8.chispa").write_bytes(self.chispa.read_bytes())
        rc, r = self.corre("--fase", "caliente", "--n", "10", "--calentar", "1", dir_=d)
        self.assertEqual(rc, 0)
        self.assertIn("L1", r["omitidos"])
        self.assertIn("L2", r["omitidos"])
        self.assertIn("torres", r["omitidos"])
        self.assertEqual(r["capas_medidas"], ["L0"])

    def test_textos_jsonl(self):
        f = Path(self.tmp.name) / "t.jsonl"
        f.write_text("\n".join(json.dumps({"texto": "quiero solicitar una cita %d" % i}) for i in range(5)) + "\n", encoding="utf-8")
        rc, r = self.corre("--fase", "caliente", "--n", "12", "--calentar", "1", "--textos", str(f))
        self.assertEqual(rc, 0)
        self.assertEqual(r["textos"]["filas_distintas_en_la_fuente"], 5)

    def test_percentiles(self):
        self.assertEqual(self.m.pct([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(self.m.pct([0, 10], 95), 9.5)


if __name__ == "__main__":
    unittest.main()
