"""Pruebas de model.alt.llm_logprobs sin red externa: un servidor HTTP falso en un hilo devuelve logprobs fijos."""
import json
import math
import threading

import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model.alt import llm_logprobs as lg


class Falso:
    """Servidor de chat/completions: `top` = [(token, logprob)] del primer token; `prefijo` = {etiqueta: logprob} para prompt_logprobs."""

    def __init__(self, top, prefijo=None):
        self.cuerpos = []
        falso = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                c = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                falso.cuerpos.append(c)
                if "prompt_logprobs" in c:
                    cod = c["messages"][-1]["content"]
                    if cod not in (prefijo or {}):
                        self.send_response(500)
                        self.end_headers()
                        return
                    cuerpo = {"choices": [{"message": {"content": "x"}}],
                              "prompt_logprobs": [None, {"1": {"logprob": -9.0, "decoded_token": "hola"}},
                                                  {"2": {"logprob": prefijo[cod], "decoded_token": cod}}]}
                else:
                    cuerpo = {"choices": [{"message": {"content": top[0][0]}, "logprobs": {"content": [
                        {"token": top[0][0], "logprob": top[0][1], "top_logprobs": [{"token": t, "logprob": l} for t, l in top]}]}}]}
                b = json.dumps(cuerpo).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def cerrar(self):
        self.srv.shutdown()
        self.srv.server_close()


def fila(n=3, tipo="choice", id_="f1"):
    return {"id": id_, "tipo": tipo, "pregunta": "¿cuál?", "opciones": ["op%d" % i for i in range(n)],
            "descripciones": {"op0": "la primera"}, "texto": "hola <b>mundo</b>"}


def correr(tmp_path, filas, srv, *extra):
    p, s = tmp_path / "filas.jsonl", tmp_path / "sal.jsonl"
    p.write_text("".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")
    lg.main(["--filas", str(p), "--url", srv.url, "--modelo", "m", "--salida", str(s), *extra])
    return [json.loads(l) for l in s.read_text(encoding="utf-8").splitlines()]


def test_probs_son_el_softmax_de_los_logprobs(tmp_path):
    srv = Falso([("A", -0.5), ("B", -1.5), ("C", -2.5), ("The", -4.0)])
    try:
        r = correr(tmp_path, [fila()], srv)[0]
    finally:
        srv.cerrar()
    z = sum(math.exp(x) for x in (-0.5, -1.5, -2.5))
    assert r["valida"] and r["intentos"] == 1 and r["error"] == ""
    assert abs(r["probs"]["op0"] - math.exp(-0.5) / z) < 1e-9
    assert abs(sum(r["probs"].values()) - 1) < 1e-9
    assert max(r["probs"], key=r["probs"].get) == "op0"


def test_formato_de_salida_y_prompt(tmp_path):
    srv = Falso([("B", -0.1), ("A", -2.3), ("C", -3.0)])
    try:
        r = correr(tmp_path, [fila()], srv)[0]
        c = srv.cuerpos[0]
    finally:
        srv.cerrar()
    assert list(r) == ["id", "probs", "valida", "raw", "intentos", "ms", "error"]
    assert set(r["probs"]) == {"op0", "op1", "op2"}
    assert json.loads(r["raw"])["del_top"] == 3
    assert c["logprobs"] is True and c["top_logprobs"] == 20 and c["max_tokens"] == 1 and c["temperature"] == 0.0
    assert "response_format" not in c
    ent = json.loads(c["messages"][1]["content"])
    assert [o["etiqueta"] for o in ent["opciones"]] == ["A", "B", "C"]
    assert ent["texto"].startswith("‹dato›") and "<" not in ent["texto"]   # el escape del dato de llm_jevbench
    assert "JSON {" not in c["messages"][0]["content"] and "etiqueta" in c["messages"][0]["content"]


def test_opciones_fuera_del_top_se_puntuan_por_prefijo_y_el_resto_recibe_piso(tmp_path):
    # 8 opciones (A..H); en el top solo A y B; C y D se puntúan por prefijo (--max-extra 2); E..H reciben el piso
    srv = Falso([("A", -0.2), ("B", -2.0), ("zz", -5.0)], prefijo={"C": -3.0, "D": -6.0})
    try:
        r = correr(tmp_path, [fila(8)], srv, "--max-extra", "2")[0]
    finally:
        srv.cerrar()
    p = r["probs"]
    assert r["valida"] and len(p) == 8 and all(v > 0 for v in p.values())
    assert abs(sum(p.values()) - 1) < 1e-9
    z = math.exp(-0.2) + math.exp(-2.0) + math.exp(-3.0) + math.exp(-6.0) + 4 * math.exp(-7.0)   # piso = -6 - 1
    assert abs(p["op2"] - math.exp(-3.0) / z) < 1e-9
    assert abs(p["op7"] - math.exp(-7.0) / z) < 1e-9
    assert json.loads(r["raw"]) == {"token": "A", "del_top": 2, "por_prefijo": 2, "con_piso": 4}
    n_pref = [c for c in srv.cuerpos if "prompt_logprobs" in c]
    assert len(n_pref) == 2 and n_pref[0]["continue_final_message"] is True and n_pref[0]["add_generation_prompt"] is False


def test_variantes_con_espacio_suman_probabilidad(tmp_path):
    srv = Falso([("A", -1.0), (" A", -1.0), ("B", -1.0)])
    try:
        r = correr(tmp_path, [fila(2)], srv)[0]
    finally:
        srv.cerrar()
    assert abs(r["probs"]["op0"] - 2 / 3) < 1e-9


def test_si_el_relleno_falla_las_opciones_quedan_con_piso_no_a_cero(tmp_path):
    srv = Falso([("A", -0.5), ("B", -1.0)], prefijo={})   # el servidor da 500 en prompt_logprobs
    try:
        r = correr(tmp_path, [fila(4)], srv)[0]
    finally:
        srv.cerrar()
    assert r["valida"] and all(v > 0 for v in r["probs"].values()) and r["error"].startswith("prefijo")


def test_sin_ninguna_etiqueta_la_fila_es_invalida(tmp_path):
    srv = Falso([("The", -0.5), ("Sure", -1.0)], prefijo={})
    try:
        r = correr(tmp_path, [fila(3)], srv, "--reintentos", "0")[0]
    finally:
        srv.cerrar()
    assert r["probs"] is None and r["valida"] is False and r["error"]


def test_sin_pensar_llega_en_el_cuerpo_solo_si_se_pide(tmp_path):
    srv = Falso([("A", -0.5), ("B", -1.0)], prefijo={"C": -2.0})
    try:
        (tmp_path / "a").mkdir()
        correr(tmp_path / "a", [fila(3)], srv, "--sin-pensar")
        con = [c.get("chat_template_kwargs") for c in srv.cuerpos]
        srv.cuerpos.clear()
        (tmp_path / "b").mkdir()
        correr(tmp_path / "b", [fila(3)], srv)
        sin = [c.get("chat_template_kwargs") for c in srv.cuerpos]
    finally:
        srv.cerrar()
    assert con and all(k == {"enable_thinking": False} for k in con)   # también en la llamada de prefijo
    assert sin and all(k is None for k in sin)


def test_score_de_una_cifra_usa_el_nivel_como_etiqueta(tmp_path):
    f = fila(3, "score")
    f["opciones"] = ["3", "1", "2"]
    srv = Falso([("2", -0.3), ("1", -1.5), ("3", -2.0)])
    try:
        r = correr(tmp_path, [f], srv)[0]
        ent = json.loads(srv.cuerpos[0]["messages"][1]["content"])
    finally:
        srv.cerrar()
    assert [o["etiqueta"] for o in ent["opciones"]] == ["1", "2", "3"]
    assert max(r["probs"], key=r["probs"].get) == "2"


def test_sesenta_opciones_tienen_etiquetas_distintas_y_mas_de_62_es_invalida(tmp_path):
    _, etq = lg.preparar(fila(60))
    assert len(set(etq)) == 60 and all(len(c) == 1 for c in etq)
    srv = Falso([("A", -0.5)])
    try:
        r = correr(tmp_path, [fila(63)], srv)[0]
    finally:
        srv.cerrar()
    assert r["valida"] is False and "63 opciones" in r["error"] and srv.cuerpos == []


def test_reanuda_y_respeta_max(tmp_path):
    srv = Falso([("A", -0.5), ("B", -1.0)])
    try:
        fs = [fila(2, id_="x%d" % i) for i in range(3)]
        correr(tmp_path, fs, srv, "--max", "2")
        r = correr(tmp_path, fs, srv)
    finally:
        srv.cerrar()
    assert [x["id"] for x in r] == ["x0", "x1", "x2"]


# ---------------------------------------------------------------- R10-T3: ejemplos en contexto (few-shot), apagado por defecto
def soporte(n_por_clase=5, clases=("op0", "op1", "op2"), split="train"):
    return [dict(fila(3, id_="s-%s-%d" % (c, i)), etiqueta=c, split=split, texto="texto %s %d" % (c, i)) for c in clases for i in range(n_por_clase)]


def test_sin_ejemplos_el_prompt_es_identico_byte_a_byte(tmp_path):
    import hashlib
    f = fila()
    msgs, etq = lg.preparar(f)
    # huella calculada con `preparar` ANTES de añadir la opción de ejemplos (la función no se tocó)
    assert hashlib.sha256(json.dumps([msgs, etq], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == \
        "5ecdfeb26ac88b18a31659febbe7dc45178629821679896206b25511213fbab0"
    srv = Falso([("A", -0.5), ("B", -1.5)])
    try:
        correr(tmp_path, [f], srv)
        c = srv.cuerpos[0]
    finally:
        srv.cerrar()
    assert c["messages"] == msgs and len(c["messages"]) == 2
    assert json.dumps(c["messages"], ensure_ascii=False) == json.dumps(msgs, ensure_ascii=False)


def test_ejemplos_estratificados_deterministas_y_solo_train():
    sop = soporte() + soporte(split="test", n_por_clase=2)
    a = lg.elegir_ejemplos(sop, 8, 7)
    b = lg.elegir_ejemplos(list(reversed(sop)), 8, 7)
    assert [e["id"] for e in a] == [e["id"] for e in b] and len(a) == 8
    assert all(e["split"] == "train" for e in a)
    cuenta = {c: sum(1 for e in a if e["etiqueta"] == c) for c in ("op0", "op1", "op2")}
    assert sorted(cuenta.values()) == [2, 3, 3]
    assert [e["id"] for e in a] != [e["id"] for e in lg.elegir_ejemplos(sop, 8, 8)]
    assert len(lg.elegir_ejemplos(soporte(1), 8, 0)) == 3   # menos soporte que k: todo
    with pytest.raises(ValueError):
        lg.elegir_ejemplos(sop, 8, 7, {"s-op0-0"})


def test_con_ejemplos_turnos_previos_con_la_etiqueta_correcta(tmp_path):
    p = tmp_path / "sop.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in soporte()), encoding="utf-8")
    srv = Falso([("A", -0.5), ("B", -1.5)])
    try:
        correr(tmp_path, [fila()], srv, "--ejemplos", str(p), "--k", "8")
        m = srv.cuerpos[0]["messages"]
    finally:
        srv.cerrar()
    assert len(m) == 1 + 16 + 1 and m[0]["role"] == "system" and m[-1] == lg.preparar(fila())[0][1]
    for u, a in zip(m[1:-1:2], m[2:-1:2]):
        assert u["role"] == "user" and a["role"] == "assistant"
        ent = json.loads(u["content"])
        assert all(set(o) == {"id", "etiqueta"} for o in ent["opciones"])
        gold = ent["texto"].split()[1]   # «‹dato›texto opN i›»
        assert a["content"] == {"op0": "A", "op1": "B", "op2": "C"}[gold]
    assert m[0]["content"].startswith(lg.preparar(fila())[0][0]["content"])


def test_demos_por_fila_apagado_por_defecto_y_estrictas(tmp_path):
    sop = soporte()
    p = tmp_path / "sop.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in sop), encoding="utf-8")
    d = tmp_path / "demos.jsonl"
    d.write_text(json.dumps({"id": "f1", "ejemplos": ["s-op2-1", "s-op0-3"]}) + "\n", encoding="utf-8")
    srv = Falso([("A", -0.5), ("B", -1.5)])
    try:
        correr(tmp_path, [fila()], srv, "--ejemplos", str(p), "--demos", str(d))
        m = srv.cuerpos[0]["messages"]
    finally:
        srv.cerrar()
    assert len(m) == 1 + 4 + 1 and [x["content"] for x in m[2:-1:2]] == ["C", "A"]   # las dos demostraciones pedidas, en ese orden, con su etiqueta
    d.write_text(json.dumps({"id": "otra", "ejemplos": ["s-op2-1"]}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        correr(tmp_path, [dict(fila(), id="f2")], Falso([("A", -1.0)]), "--ejemplos", str(p), "--demos", str(d))
