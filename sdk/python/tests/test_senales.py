from chispa.tipos import Resultado


def test_campos_opcionales_servidor_antiguo():
    r = Resultado.de_json({"id": "q", "kind": "choice", "answer": "a", "probs": {"a": 1}, "layer": "L1", "escalate": False, "ms": 1})
    assert r.calibrada is None and r.truncado is False and r.tokens_originales is None and r.unk_pct is None


def test_campos_opcionales_servidor_nuevo():
    r = Resultado.de_json({"id": "q", "kind": "choice", "answer": "a", "probs": {"a": 1}, "layer": "L1", "escalate": False, "ms": 1,
                           "calibrada": False, "truncado": True, "tokens_originales": 900, "unk_pct": 0.25})
    assert r.calibrada is False and r.truncado is True and r.tokens_originales == 900 and r.unk_pct == 0.25
