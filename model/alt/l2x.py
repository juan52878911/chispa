"""L2x (P3): un codificador descongelado que lee pregunta + todas las opciones + texto en UNA secuencia (estilo GLiClass).

    [CLS] query: pregunta [SEP] [OPC] desc₁ [OPC] desc₂ … [OPC] descₖ [SEP] texto [SEP]

Cada opción se reduce al promedio de los estados de su tramo (la marca [OPC] y su descripción) y el texto, al promedio de los suyos;
la puntuación sale de una cabeza pequeña sobre [opción, texto, opción⊙texto]. El softmax por pregunta da las probabilidades, igual que
en `model.alt.modelo.CrossEncoder` (de quien hereda entrada, guardado y carga), pero con UNA pasada por pregunta y no una por opción.
  - Entrenamiento: si la fila trae más de `max_opciones` (8) opciones, se queda con la correcta y `max_opciones-1` al azar, y las
    opciones se barajan en cada paso (la posición no puede delatar la respuesta). La blanda se recorta a esas opciones y se renormaliza.
  - Evaluación con más de `max_opciones`: torneo por trozos (cada trozo puntúa sus opciones, pasan las mejores, y una pasada final decide).
    No es la preselección con L1 (potion) de la arquitectura objetivo: es el sustituto mientras L1 no esté en este arnés.
  - Presupuesto de tokens: pregunta ≤ 48, descripción ≤ 48 (menos si hay muchas opciones) y al texto siempre le quedan ≥ 64.
Qué NO trae esta versión (y sí la arquitectura objetivo): cabezas CORAL/multi, calibración y cabeza de escalada, preselección con L1.
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modelo import MAX_PREGUNTA, CrossEncoder, descripcion, opciones_ordenadas

MARCA = "[OPC]"
MIN_TEXTO = 64
MAX_DESC_X = 48
EPS_ELIMINADAS = 0.02   # masa que el torneo reparte entre las opciones eliminadas


class L2x(CrossEncoder):
    def __init__(self, enc, tok, codificador, pooling="media", max_tok=256, dropout=0.1, max_opciones=8):
        super().__init__(enc, tok, codificador, pooling, max_tok, dropout)
        if MARCA not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [MARCA]})
        if len(tok) > enc.get_input_embeddings().num_embeddings:
            enc.resize_token_embeddings(len(tok))
        self.marca_id = tok.convert_tokens_to_ids(MARCA)
        h = enc.config.hidden_size
        self.cabeza = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1))
        self.max_opciones = max_opciones
        self._rng = random.Random(0)

    # --- entrada
    def armar(self, f, ops):
        """(ids, tramos de cada opción, tramo del texto) de la fila `f` con las opciones `ops` (ya en el orden en que entran)."""
        t = self.tok
        cls_id = t.cls_token_id if t.cls_token_id is not None else t.bos_token_id
        sep_id = t.sep_token_id if t.sep_token_id is not None else t.eos_token_id
        preg = self._ids(self.prefijo + f["pregunta"], MAX_PREGUNTA)
        k = len(ops)
        fijo = 3 + len(preg) + k + 1   # CLS, SEP, SEP, SEP, las marcas y el sitio del primer token de texto
        por_op = max(3, min(MAX_DESC_X, (self.max_tok - fijo - MIN_TEXTO) // k))
        descs = [self._ids(descripcion(f, o), por_op) for o in ops]
        resto = max(8, self.max_tok - fijo - sum(len(d) for d in descs))
        txt = self._ids(f["texto"], resto)
        ids, spans = [cls_id] + preg + [sep_id], []
        for d in descs:
            spans.append((len(ids), len(ids) + 1 + len(d)))
            ids += [self.marca_id] + d
        ids.append(sep_id)
        span_txt = (len(ids), len(ids) + len(txt))
        ids += txt + [sep_id]
        return ids[:self.max_tok], spans, span_txt

    def largo(self, f):
        """Tokens de la secuencia de `f` con sus primeras `max_opciones` opciones (para `partir`)."""
        if f["id"] not in self._largos:
            ops = opciones_ordenadas(f)[:self.max_opciones]
            self._largos[f["id"]] = len(self.armar(f, ops)[0])
        return self._largos[f["id"]]

    def partir(self, filas, max_tokens):
        """Grupos consecutivos de filas cuyo tensor relleno cabe en `max_tokens`; una fila con muchas opciones cuenta una vez por trozo de opciones."""
        out, act, n, m = [], [], 0, 0
        for f in filas:
            k = max(1, math.ceil(len(f["opciones"]) / self.max_opciones))
            l = self.largo(f)
            if act and (n + k) * max(m, l) > max_tokens:
                out.append(act)
                act, n, m = [], 0, 0
            act.append(f)
            n, m = n + k, max(m, l)
        return out + [act] if act else out

    def tensores(self, items):
        """items = [(fila, ops)] → ids (B,T), máscara (B,T), tramos de opción (B,K,T) y de texto (B,T), todo en float salvo ids y máscara."""
        arm = [self.armar(f, ops) for f, ops in items]
        n = max(len(a[0]) for a in arm)
        kmax = max(len(a[1]) for a in arm)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(arm), n), pad, dtype=torch.long)
        mask = torch.zeros(len(arm), n, dtype=torch.long)
        om = torch.zeros(len(arm), kmax, n)
        tm = torch.zeros(len(arm), n)
        for i, (s, sp, st) in enumerate(arm):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
            for j, (a, b) in enumerate(sp):
                om[i, j, a:b] = 1.0
            tm[i, st[0]:st[1]] = 1.0
        return ids, mask, om, tm

    # --- salida
    def puntuar(self, ids, mask, om, tm):
        """Puntuaciones (B,K) de las opciones (las de relleno valen lo que valgan: el que llama usa solo las suyas)."""
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        om, tm = om.to(h.dtype), tm.to(h.dtype)
        vo = torch.einsum("bkt,bth->bkh", om, h) / om.sum(-1, keepdim=True).clamp(min=1)
        vt = (tm.unsqueeze(-1) * h).sum(1) / tm.sum(-1, keepdim=True).clamp(min=1)
        vt = vt.unsqueeze(1).expand_as(vo)
        return self.cabeza(self.drop(torch.cat([vo, vt, vo * vt], -1))).squeeze(-1)

    def _pase(self, items):
        dev = next(self.parameters()).device
        ids, mask, om, tm = (x.to(dev) for x in self.tensores(items))
        s = self.puntuar(ids, mask, om, tm)
        return [s[i, :len(ops)] for i, (_, ops) in enumerate(items)]

    def _elegir(self, f):
        """Opciones que entran en una pasada de entrenamiento: la correcta y hasta `max_opciones-1` más, barajadas."""
        ops = list(opciones_ordenadas(f))
        if len(ops) > self.max_opciones:
            otras = [o for o in ops if o != f["etiqueta"]]
            self._rng.shuffle(otras)
            ops = [f["etiqueta"]] + otras[:self.max_opciones - 1]
        self._rng.shuffle(ops)
        return ops

    def perdida(self, filas, peso_blanda=0.5):
        """Entropía cruzada con la etiqueta dura; con `blanda`, (1-w)·dura + w·blanda sobre las opciones que entraron."""
        items = [(f, self._elegir(f)) for f in filas]
        tot = 0.0
        for (f, ops), s in zip(items, self._pase(items)):
            lp = F.log_softmax(s, 0)
            dura = -lp[ops.index(f["etiqueta"])]
            b = f.get("blanda")
            if b:
                z = sum(max(b.get(o, 0.0), 0.0) for o in ops)
                if z > 0:
                    blanda = -sum(max(b.get(o, 0.0), 0.0) / z * lp[i] for i, o in enumerate(ops))
                    dura = (1 - peso_blanda) * dura + peso_blanda * blanda
            tot = tot + dura
        return tot / len(filas)

    def probs(self, filas):
        """[{id_opción: prob}] sin gradiente y en eval. Más de `max_opciones`: torneo por trozos."""
        self.eval()
        with torch.no_grad():
            chicas = [f for f in filas if len(f["opciones"]) <= self.max_opciones]
            por = dict(zip([f["id"] for f in chicas], self._probs_lote(chicas) if chicas else []))
            return [por[f["id"]] if f["id"] in por else self._probs_fila(f) for f in filas]

    def _probs_lote(self, filas):
        items = [(f, opciones_ordenadas(f)) for f in filas]
        return [dict(zip(ops, F.softmax(s.double(), 0).tolist())) for (_, ops), s in zip(items, self._pase(items))]

    def _probs_fila(self, f):
        ops = opciones_ordenadas(f)
        m = self.max_opciones
        if len(ops) <= m:
            return self._probs_lote([f])[0]
        vivas, punt = list(ops), {}
        while len(vivas) > m:
            n_trozos = math.ceil(len(vivas) / m)
            tam = math.ceil(len(vivas) / n_trozos)
            trozos = [vivas[i:i + tam] for i in range(0, len(vivas), tam)]
            pasan = max(1, m // len(trozos)) if len(trozos) <= m else 1
            sig = []
            for tr, s in zip(trozos, self._pase([(f, tr) for tr in trozos])):
                orden = sorted(range(len(tr)), key=lambda i: -float(s[i]))
                sig += [tr[i] for i in orden[:pasan]]
                punt.update({o: float(s[i]) for i, o in enumerate(tr)})
            vivas = sig if len(sig) < len(vivas) else sig[:m]
        final = self._pase([(f, vivas)])[0]
        p = F.softmax(final.double(), 0).tolist()
        fuera = [o for o in ops if o not in vivas]
        out = {o: (1 - EPS_ELIMINADAS) * pi for o, pi in zip(vivas, p)}
        out.update({o: EPS_ELIMINADAS / len(fuera) for o in fuera})
        return {o: out[o] for o in ops}

    # --- guardado: igual que el cruzado, más la arquitectura y las opciones por pasada
    def guardar(self, directorio, extra=None):
        import json
        from pathlib import Path
        super().guardar(directorio, extra)
        p = Path(directorio) / "config.json"
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg.update(arquitectura="l2x", max_opciones=self.max_opciones)
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")

    @classmethod
    def cargar(cls, directorio, device="cpu"):
        import json
        from pathlib import Path
        m = super().cargar(directorio, "cpu")
        m.max_opciones = json.loads((Path(directorio) / "config.json").read_text(encoding="utf-8")).get("max_opciones", 8)
        return m.to(device).eval()
