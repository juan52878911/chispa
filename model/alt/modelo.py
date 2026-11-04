"""Rival de Chispa en el hito 4 (b): un codificador abierto descongelado con cabeza cross-encoder por opción.

Cada opción de una pregunta es una entrada aparte: `<prefijo><pregunta> [SEP] <texto> [SEP] <descripción>` (el prefijo es
«query: » en e5 y vacío en mmBERT; [SEP] es el separador del tokenizer). Una cabeza lineal sobre el pooling (media o [CLS]) da una
puntuación por opción y un softmax sobre las opciones de LA pregunta da las probabilidades. Sin ranuras ni torre MLP.
  - noul: choice de 2 opciones («no»/«sí»; si la fila trae descripción, esa), opciones ["false", "true"].
  - score: choice con las opciones ordenadas por id numérico; las probabilidades salen por id (la etiqueta CORAL la saca el arnés).
  - pérdida: entropía cruzada con la etiqueta dura y, si la fila trae `blanda`, mezcla con la blanda (peso `peso_blanda`).
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CODIFICADORES = {
    "e5-small": "intfloat/multilingual-e5-small",
    "mmbert-small": "jhu-clsp/mmBERT-small",
}
PREFIJOS = {"intfloat/multilingual-e5-small": "query: "}   # mmBERT no pide prefijo
PLANTILLA_NOUL = {"false": "no", "true": "sí"}
MAX_PREGUNTA, MAX_DESC = 48, 64   # tokens; el texto se queda con el resto de max_tok
# Tokens con relleno por pasada. 384 secuencias × 256 (≈ 98 000) dieron OOM en la g5 (A10G, 22 GiB) aun con bf16: un tercio deja margen.
MAX_TOKENS = 32768


def resolver(nombre):
    """Alias corto (`e5-small`, `mmbert-small`) o id de HF completo."""
    return CODIFICADORES.get(nombre, nombre)


def opciones_ordenadas(fila):
    """Ids de las opciones en el orden en que se puntúan: score por id numérico; el resto, el de la fila."""
    ops = list(fila["opciones"])
    return sorted(ops, key=int) if fila.get("tipo") == "score" else ops


def trozos(filas, max_tokens, largo):
    """Grupos consecutivos de `filas` cuyo tensor relleno (secuencias × la más larga, `largo(fila)` tokens) no pasa de `max_tokens`.
    Una fila sola nunca se parte, aunque sola pase del tope (clinc150 trae 150 opciones)."""
    out, act, n, m = [], [], 0, 0
    for f in filas:
        k, l = len(f["opciones"]), largo(f)
        if act and (n + k) * max(m, l) > max_tokens:
            out.append(act)
            act, n, m = [], 0, 0
        act.append(f)
        n, m = n + k, max(m, l)
    return out + [act] if act else out


def descripcion(fila, o):
    d = (fila.get("descripciones") or {}).get(o) or ""
    if fila.get("tipo") == "noul":
        return d or PLANTILLA_NOUL.get(o, o)
    return d or o


class CrossEncoder(nn.Module):
    def __init__(self, enc, tok, codificador, pooling="media", max_tok=256, dropout=0.1):
        super().__init__()
        if pooling not in ("media", "cls"):
            raise ValueError("pooling: media o cls")
        self.enc, self.tok, self.codificador, self.pooling, self.max_tok = enc, tok, codificador, pooling, max_tok
        self._largos = {}
        self.drop = nn.Dropout(dropout)
        self.cabeza = nn.Linear(enc.config.hidden_size, 1)
        self.prefijo = PREFIJOS.get(codificador, "")

    @classmethod
    def desde_hf(cls, codificador="e5-small", **kw):
        from transformers import AutoModel, AutoTokenizer
        nombre = resolver(codificador)
        return cls(AutoModel.from_pretrained(nombre), AutoTokenizer.from_pretrained(nombre), nombre, **kw)

    # --- entrada
    def _ids(self, texto, tope):
        return self.tok(texto, add_special_tokens=False, truncation=True, max_length=tope)["input_ids"]

    def secuencias(self, f):
        """Una lista de ids por opción de `f`, con el texto recortado para que la opción siempre quepa."""
        t = self.tok
        cls_id = t.cls_token_id if t.cls_token_id is not None else t.bos_token_id
        sep_id = t.sep_token_id if t.sep_token_id is not None else t.eos_token_id
        preg = self._ids(self.prefijo + f["pregunta"], MAX_PREGUNTA)
        descs = [self._ids(descripcion(f, o), MAX_DESC) for o in opciones_ordenadas(f)]
        resto = max(8, self.max_tok - 4 - len(preg) - max(len(d) for d in descs))
        txt = self._ids(f["texto"], resto)
        return [([cls_id] + preg + [sep_id] + txt + [sep_id] + d + [sep_id])[:self.max_tok] for d in descs]

    def largo(self, f):
        """Tokens de la secuencia más larga de `f` (para `trozos`; se calcula una vez por id)."""
        if f["id"] not in self._largos:
            self._largos[f["id"]] = max(len(s) for s in self.secuencias(f))
        return self._largos[f["id"]]

    def codificar(self, filas):
        """(ids, máscara, tamaños): una secuencia por (fila, opción)."""
        seqs, tams = [], []
        for f in filas:
            ss = self.secuencias(f)
            tams.append(len(ss))
            seqs += ss
        n = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (n - len(s)) for s in seqs])
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs])
        return ids, mask, tams

    # --- salida
    def puntuaciones(self, filas):
        """Lista de tensores (n_opciones,) con las puntuaciones de cada fila."""
        dev = next(self.parameters()).device
        ids, mask, tams = self.codificar(filas)
        h = self.enc(input_ids=ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state
        if self.pooling == "cls":
            v = h[:, 0]
        else:
            m = mask.to(dev).unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        s = self.cabeza(self.drop(v)).squeeze(-1)
        return list(torch.split(s, tams))

    def probs(self, filas):
        """[{id_opción: prob}] sin gradiente y en eval."""
        self.eval()
        with torch.no_grad():
            return [dict(zip(opciones_ordenadas(f), F.softmax(s.double(), 0).tolist())) for f, s in zip(filas, self.puntuaciones(filas))]

    def perdida(self, filas, peso_blanda=0.5):
        """Entropía cruzada con la etiqueta dura; con `blanda`, (1-w)·dura + w·blanda."""
        tot = 0.0
        for f, s in zip(filas, self.puntuaciones(filas)):
            lp = F.log_softmax(s, 0)
            ops = opciones_ordenadas(f)
            dura = -lp[ops.index(f["etiqueta"])]
            b = f.get("blanda")
            if b:
                z = sum(max(b.get(o, 0.0), 0.0) for o in ops)
                if z > 0:
                    blanda = -sum(max(b.get(o, 0.0), 0.0) / z * lp[i] for i, o in enumerate(ops))
                    dura = (1 - peso_blanda) * dura + peso_blanda * blanda
            tot = tot + dura
        return tot / len(filas)

    # --- guardado: nada se baja al cargar (config del codificador + tokenizer dentro de la carpeta)
    def guardar(self, directorio, extra=None):
        d = Path(directorio)
        d.mkdir(parents=True, exist_ok=True)
        self.tok.save_pretrained(d / "tokenizer")
        torch.save({"estado": self.state_dict(), "extra": extra or {}}, d / "modelo.pt")
        cfg = {"codificador": self.codificador, "pooling": self.pooling, "max_tok": self.max_tok,
               "encoder_config": self.enc.config.to_dict()}
        (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")

    @classmethod
    def cargar(cls, directorio, device="cpu"):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        d = Path(directorio)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        ec = cfg["encoder_config"]
        enc = AutoModel.from_config(AutoConfig.for_model(ec.pop("model_type"), **{k: v for k, v in ec.items() if k != "transformers_version"}))
        m = cls(enc, AutoTokenizer.from_pretrained(d / "tokenizer"), cfg["codificador"], cfg["pooling"], cfg["max_tok"])
        m.load_state_dict(torch.load(d / "modelo.pt", map_location="cpu", weights_only=False)["estado"])
        return m.to(device).eval()
