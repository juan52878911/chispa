"""Datos sintéticos diminutos para la prueba en seco del piloto R10 (`python -m model.alt.destilar_biencoder sintetico --salida DIR`).

Deja en DIR (nada sale de la máquina, nada se descarga):
    e5/                      un BERT aleatorio de 1 capa con hidden 384 (como e5-small) y tokenizador de letras: `CHISPA_E5=DIR/e5`
    vocab-diminuto.json      vocabulario «podado» (sin unos pocos ids) con el formato de model/bench/vocab-podado-e5-small.json
    raiz/datos/*.jsonl.gz    los ficheros del manifiesto-entrenamiento.json (vacíos salvo spam_tanaos y tsjev-admitido), SIN blanda
    blandas.jsonl.gz         {id, blanda} de 9 de cada 10 filas de train (lo que aplicaría `model.alt.destilar --aplicar`)
    dev/<tarea>.jsonl        las 6 tareas DEV de v2 con soporte (split train, 8 por clase) y test
    partida.pt               un checkpoint de L2 con pesos aleatorios, en el formato de `model.l2.entrenar`
Incluye tareas que se tienen que excluir (multilingual/massive, prompt-injection-dataset) y una fila de train con el texto de un test de DEV
(casi-duplicado), para que la prueba vea trabajar los filtros.
"""
import gzip
import json
import random
import string
from pathlib import Path

import torch

PALABRAS = ["casa", "gato", "perro", "sol", "luna", "mesa", "libro", "agua", "fuego", "viento", "pago", "queja", "error", "bueno", "malo",
            "rapido", "lento", "gracias", "ayuda", "cuenta", "banco", "nota", "plato", "vino", "cena", "voto", "ley", "norma", "curso", "beca"]
DEV = {   # tarea: (tipo, clases, idioma) como las dev de la batería v2
    "finanzas-documentos": ("choice", ["recibo", "factura", "extracto"], "es"),
    "horeca-resenas": ("score", ["1", "2", "3", "4"], "es"),
    "jql-valor-educativo": ("score", ["1", "2", "3", "4"], "es"),
    "cfpb-quejas": ("choice", ["cobro", "credito", "hipoteca"], "en"),
    "usajobs-nivel": ("score", ["1", "2", "3", "4"], "en"),
    "congreso-eeuu-partido": ("choice", ["rojo", "azul", "verde"], "en"),
}


def crear_e5(destino):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    piezas = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(string.ascii_lowercase + string.digits + string.punctuation)
    piezas += ["##" + c for c in string.ascii_lowercase + string.digits]
    piezas = list(dict.fromkeys(piezas))
    Path(destino).mkdir(parents=True, exist_ok=True)
    (Path(destino) / "vocab.txt").write_text("\n".join(piezas) + "\n", encoding="utf-8")   # vocab_file: vale en transformers 4.x y 5.x
    try:   # transformers 5: vocab=dict; 4.x: vocab_file
        tok = BertTokenizerFast(vocab={x: i for i, x in enumerate(piezas)}, do_lower_case=True)
    except Exception:
        tok = BertTokenizerFast(vocab_file=str(Path(destino) / "vocab.txt"), do_lower_case=True)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(piezas), hidden_size=384, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                     max_position_embeddings=130, pad_token_id=0)
    BertModel(cfg, add_pooling_layer=False).save_pretrained(destino)
    tok.save_pretrained(destino)
    Path(destino, "..", "vocab-diminuto.json").resolve().write_text(json.dumps(
        {"modelo": "diminuto", "original": len(piezas), "ids_originales": [i for i in range(len(piezas)) if i % 17 != 16 or i < 5],
         "unk": 1}) + "\n", encoding="utf-8")


def _texto(rng, n=8):
    return " ".join(rng.choice(PALABRAS) for _ in range(n))


def _fila(id_, tarea, tipo, ops, desc, etq, texto, split, fuente="sintetico", pregunta=None):
    return {"id": id_, "tarea": tarea, "tipo": tipo, "pregunta": pregunta or "¿%s?" % tarea, "texto": texto, "opciones": ops,
            "descripciones": desc, "etiqueta": etq, "idioma": "es", "split": split, "fuente": fuente, "licencia": "apache-2.0",
            "uso_comercial": True}


def _blanda(rng, ops, etq):
    p = {o: rng.random() * 0.4 for o in ops}
    if rng.random() < 0.7:
        p[etq] += 1.0
    z = sum(p.values())
    return {o: v / z for o, v in p.items()}


def _gz(ruta, filas):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wt", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generar(salida, semilla=0):
    sal = Path(salida)
    rng = random.Random(semilla)
    crear_e5(sal / "e5")
    man = json.loads((Path(__file__).parent / "manifiesto-entrenamiento.json").read_text(encoding="utf-8"))
    datos = {c["fichero"]: [] for c in man["conjuntos"]}
    blandas = {}
    # dev v2: soporte de 8 por clase y test de 10 por clase; el texto lleva la clase para que haya algo que aprender
    dev_dir = sal / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)
    primer_test = None
    for t, (tipo, cl, idi) in DEV.items():
        desc = {c: "texto de clase %s" % c for c in cl}
        filas = []
        for split, n in (("train", 8), ("test", 10)):
            for c in cl:
                for i in range(n):
                    tx = "%s %s" % (c, _texto(rng))
                    filas.append(_fila("%s:%s:%s:%d" % (t, split, c, i), t, tipo, cl, desc, c, tx, split, "dev-v2-sintetica"))
                    if split == "test" and primer_test is None:
                        primer_test = tx
        (dev_dir / (t + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filas), encoding="utf-8")
    # train: spam (choice de clases fijas), CONDAQA (choice), una tarea score, y dos tareas que hay que excluir
    def tarea(fichero, tarea, tipo, cl, n, prefijo, dev_n=6, fuente="sintetico"):
        desc = {c: "descripcion de %s" % c for c in cl}
        for split, m in (("train", n), ("dev", dev_n)):
            for i in range(m):
                c = cl[rng.randrange(len(cl))]
                r = _fila("%s:%s:%s:%d" % (prefijo, tarea, split, i), tarea, tipo, cl, desc, c,
                          "%s %s" % (c, _texto(rng)), split, fuente)
                datos[fichero].append(r)
                if split == "train" and rng.random() < 0.9:
                    blandas[r["id"]] = _blanda(rng, cl, c)
    tarea("datos/spam_tanaos.jsonl.gz", "spam", "choice", ["ham", "spam", "otro"], 70, "spam_tanaos")
    tarea("datos/tsjev-admitido.jsonl.gz", "CONDAQA", "choice", ["a", "b", "c", "d"], 70, "tsjev")
    tarea("datos/tsjev-admitido.jsonl.gz", "acceptability-prediction/rating_votes", "score", ["1", "2", "3", "4"], 50, "tsjev")
    tarea("datos/tsjev-admitido.jsonl.gz", "multilingual/massive", "choice", ["x", "y", "z"], 20, "tsjev")            # a excluir (v1)
    tarea("datos/tsjev-admitido.jsonl.gz", "prompt-injection-dataset", "choice", ["no", "si"], 20, "tsjev")           # a excluir (v1)
    # casi-duplicado: una fila de train con el texto de un test de DEV
    r = dict(datos["datos/tsjev-admitido.jsonl.gz"][0], id="tsjev:CONDAQA:train:dup", texto=primer_test)
    datos["datos/tsjev-admitido.jsonl.gz"].append(r)
    blandas[r["id"]] = _blanda(rng, r["opciones"], r["etiqueta"])
    for f, filas in datos.items():
        _gz(sal / "raiz" / f, filas)
    _gz(sal / "blandas.jsonl.gz", [{"id": i, "blanda": b} for i, b in blandas.items()])
    # checkpoint de partida (pesos aleatorios, formato de model.l2.entrenar)
    from model.l2.modelo import ModeloL2
    torch.manual_seed(1)
    m = ModeloL2()
    for k in ("score4",):
        m.asegurar_cabeza(k)
    torch.save({"modelo": m.state_dict(), "cabezas": sorted(m.coral.keys()), "paso": 0, "f1": None, "enc": {}, "max_tok": 48,
                "texto_torre": "v1", "cabeza_score": "coral1", "cabeza_noul": "una", "sim": "maxsim", "k_ranuras": 16}, sal / "partida.pt")
    return {"dev": sorted(DEV), "train_filas": sum(len(v) for v in datos.values()), "blandas": len(blandas)}
