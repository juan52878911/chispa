"""Qwen3 en miniatura para probar sin red ni pesos: un tokenizador BPE byte-level con el MISMO pipeline que el de Qwen2/Qwen3 (NFC, Split con el
regex de Qwen2 + ByteLevel, `<|endoftext|>` y compañía como añadidos, plantilla que añade `<|endoftext|>` al final) entrenado sobre el fixture,
y un modelo Qwen3 (arquitectura real, 2 capas, hidden 64) con pesos aleatorios de semilla fija. Sirve para el test del exportador a ONNX, para los
ids de referencia del tokenizador Go y para la paridad del codificador Go con ORT. NO es el modelo de la medida: esa baja Qwen/Qwen3-Embedding-0.6B.

  python model/alt/qwen3emb_mini.py DIR [--fixture runtime/tok/testdata/qwen/fixture_paridad.jsonl] [--sin-plantilla]
"""
import argparse
import json
from pathlib import Path

REGEX_QWEN2 = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ESPECIALES = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def textos_entrenamiento(fixture):
    out = []
    for l in open(fixture, encoding="utf-8"):
        r = json.loads(l)
        out += [r["texto"], r["pregunta"]]
    out += ["Instruct: Classify the text to answer this question: Query:", "Hello world, it's 2026 — ¿qué tal? ñandú café año 日本語 한국어 😀 👍"] * 20
    return out


def crear(dir_, fixture, con_plantilla=True, vocab=1200):
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    d = Path(dir_)
    d.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE(unk_token=None, fuse_unk=False, byte_fallback=False))
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(REGEX_QWEN2), behavior="isolated", invert=False),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(textos_entrenamiento(fixture), trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tk.add_special_tokens(ESPECIALES)
    if con_plantilla:
        tk.post_processor = processors.TemplateProcessing(single="$A:0 <|endoftext|>:0", pair="$A:0 <|endoftext|>:0 $B:1 <|endoftext|>:1",
                                                          special_tokens=[("<|endoftext|>", tk.token_to_id("<|endoftext|>"))])
    tk.save(str(d / "tokenizer.json"))
    tk.model.save(str(d))   # vocab.json y merges.txt
    json.dump({"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "unk_token": None, "add_prefix_space": False,
               "clean_up_tokenization_spaces": False, "model_max_length": 131072}, open(d / "tokenizer_config.json", "w"), indent=1)
    import torch
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(20261119)
    cfg = Qwen3Config(vocab_size=len(tk.get_vocab()), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=2048, rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
                      tie_word_embeddings=True, pad_token_id=tk.token_to_id("<|endoftext|>"), bos_token_id=None, eos_token_id=tk.token_to_id("<|endoftext|>"))
    m = Qwen3Model(cfg).eval()
    m.save_pretrained(str(d), safe_serialization=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--fixture", default=str(Path(__file__).resolve().parents[2] / "runtime/tok/testdata/qwen/fixture_paridad.jsonl"))
    ap.add_argument("--sin-plantilla", action="store_true")
    a = ap.parse_args()
    print(crear(a.dir, a.fixture, not a.sin_plantilla))
