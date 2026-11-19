"""Techo local: cliente harmony, licencia, partición, bench, medida con veredicto y el ud (en seco, con un vLLM falso). Datos inventados, sin red."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from model.alt import lineas_base_v2 as lb
from model.alt import llm_logprobs as ll
from model.alt import techo_local as tl
from model.alt import techo_local_cliente as cl
from model.alt.auditoria import mitad
from model.alt.tests.dev_sintetico_qwen import crear

AQUI = Path(__file__).resolve().parents[1]
RAIZ = AQUI.parents[1]
CRIT = json.loads((AQUI / "crit-techo-local.json").read_text(encoding="utf-8"))


def cargar_falso():
    spec = importlib.util.spec_from_file_location("vllm_falso", AQUI / "techo_local" / "vllm_falso.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def servidor():
    srv = cargar_falso().servir(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def fila(i, ops=("a", "b", "c")):
    return {"id": "t:test:%d" % i, "split": "test", "tipo": "choice", "pregunta": "¿Clase?", "texto": "texto inventado %d" % i, "opciones": list(ops), "descripciones": {o: "clase " + o for o in ops}, "etiqueta": ops[i % len(ops)]}


# ---------------------------------------------------------------- criterio
def test_criterio_tiene_lo_que_pide_el_brief():
    assert CRIT["cambiar"]["macro_min"] == 0.05 and CRIT["cambiar"]["req_s_min"] == 3.0
    assert CRIT["ic95"]["semilla"] == 0 and CRIT["ic95"]["remuestreos"] == 2000
    assert CRIT["referencia"] == "qwen3-4b" and CRIT["licencia"]["permitidas"] == ["apache-2.0", "mit"]
    assert CRIT["modelos"]["qwen3-4b"]["rev"] == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert "sustituto" in CRIT["modelos"]["gpt-oss-20b"] and CRIT["instancia"]["tope_usd"] == 4.0
    horas = CRIT["instancia"]["tope_min"] / 60 * CRIT["instancia"]["precio_usd_hora"]
    assert horas + 0.05 <= CRIT["instancia"]["tope_usd"]   # el tope en minutos cabe en los 4 USD con el EBS


# ---------------------------------------------------------------- harmony
def test_renderizar_harmony_sistema_developer_turnos_y_canal_final():
    msgs = [{"role": "system", "content": "INSTR"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "u2"}]
    h = cl.renderizar_harmony(msgs)
    assert h.startswith("<|start|>system<|message|>You are ChatGPT") and "Reasoning: low" in h
    assert "<|start|>developer<|message|># Instructions\n\nINSTR<|end|>" in h
    assert "<|start|>user<|message|>u1<|end|><|start|>assistant<|channel|>final<|message|>B<|end|><|start|>user<|message|>u2<|end|>" in h
    assert h.endswith("<|start|>assistant<|channel|>final<|message|>")
    abierto = cl.renderizar_harmony(msgs + [{"role": "assistant", "content": "7"}])   # relleno por prefijo: termina en la etiqueta, sin cerrar
    assert abierto.endswith("<|channel|>final<|message|>7") and "u2<|end|><|start|>assistant" in abierto
    with pytest.raises(ValueError):
        cl.renderizar_harmony([{"role": "tool", "content": "x"}])


def test_verificar_compara_con_la_plantilla_y_normaliza_la_fecha():
    igual = lambda m: cl.renderizar_harmony(m)[:-len("<|channel|>final<|message|>")].replace("2025-06-28", "2026-11-19")  # noqa: E731
    assert cl.verificar(igual)["coincide"]
    r = cl.verificar(lambda m: igual(m).replace("Reasoning: low", "Reasoning: high"))
    assert not r["coincide"] and "high" in r["primera_diferencia"]["plantilla"] and "low" in r["primera_diferencia"]["mio"]


def test_cliente_harmony_y_chat_dan_probs_validas_con_ejemplos(servidor, tmp_path):
    filas = [fila(i) for i in range(6)]
    sop = [dict(fila(i), id="t:train:%d" % i, split="train") for i in range(9)]
    (tmp_path / "f.jsonl").write_text("".join(json.dumps(r) + "\n" for r in filas))
    (tmp_path / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sop))
    demos = [{"id": r["id"], "ejemplos": ["t:train:0", "t:train:4", "t:train:8"]} for r in filas]
    (tmp_path / "d.jsonl").write_text("".join(json.dumps(r) + "\n" for r in demos))
    for fmt in ("chat", "harmony"):
        sal = tmp_path / (fmt + ".jsonl")
        cl.main(["--formato", fmt, "--filas", str(tmp_path / "f.jsonl"), "--url", servidor, "--url", servidor, "--modelo", "m", "--salida", str(sal),
                 "--ejemplos", str(tmp_path / "s.jsonl"), "--demos", str(tmp_path / "d.jsonl"), "--max-extra", "1"])
        r = [json.loads(x) for x in sal.read_text().splitlines()]
        assert len(r) == 6 and all(x["valida"] for x in r), r
        assert all(abs(sum(x["probs"].values()) - 1) < 1e-9 for x in r)
    assert ll.primer_token is cl.primer_token_h and ll.prefijo is cl.prefijo_h   # el parche harmony queda aplicado al módulo de llm_logprobs


def test_prefijo_harmony_lee_la_etiqueta_pedida(servidor):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": json.dumps({"opciones": [{"etiqueta": "A"}]})}]
    assert cl.prefijo_h(servidor, "m", msgs, "A", 5, False) == -3.0
    top, gen = cl.primer_token_h(servidor, "m", msgs, 5, False)
    assert top and top[0][0] == gen == "A"


# ---------------------------------------------------------------- licencia
def test_licencia_apache_mit_si_y_otras_no(tmp_path):
    def probar(cab, nombre):
        r = tmp_path / (nombre + ".md")
        r.write_text(cab + "# tarjeta\n")
        lic = tmp_path / (nombre + ".LICENSE")
        lic.write_text("texto")
        rc = tl.main(["licencia", "--readme", str(r), "--licencia-fichero", str(lic), "--modelo", nombre, "--salida", str(tmp_path / (nombre + ".json"))])
        return rc, json.loads((tmp_path / (nombre + ".json")).read_text())
    rc, j = probar("---\nlicense: apache-2.0\nlibrary_name: x\n---\n", "a")
    assert rc == 0 and j["permitida"] and j["tarjeta_sha256"] == hashlib.sha256((tmp_path / "a.md").read_bytes()).hexdigest() and j["fichero_licencia_sha256"]
    assert probar("---\nlicense: MIT\n---\n", "b")[0] == 0
    rc, j = probar("---\nlicense: other\nlicense_name: qwen\n---\n", "c")
    assert rc == 3 and not j["permitida"] and j["licencia"] == "other"
    rc, j = probar("sin encabezado\n", "d")
    assert rc == 3 and j["licencia"] is None
    assert probar("---\ntags: [x]\n---\nlicense: apache-2.0\n", "e")[0] == 3   # solo vale el encabezado, no el texto de la tarjeta


# ---------------------------------------------------------------- preparar, dividir, bench
@pytest.fixture
def preparado(tmp_path):
    dev = tmp_path / "dev"
    crear(dev, n_test=60, por_clase=12)
    T = tmp_path / "t"
    tl.main(["preparar", "--dev", str(dev), "--salida", str(T), "--rubricas", "mecanica"])
    for p in ("test", "rub/test"):
        tl.main(["dividir", "--origen", str(T / p), "--destino-b", str(T / (p + "B")), "--destino-a", str(T / (p + "A"))])
    # demos a mano: las 8 primeras filas del soporte de cada tarea
    (T / "demos").mkdir()
    for f in (T / "test").glob("*.jsonl"):
        sop = [r["id"] for r in lb.leer(T / "soporte" / f.name)][:8]
        (T / "demos" / f.name).write_text("".join(json.dumps({"id": r["id"], "ejemplos": sop}) + "\n" for r in lb.leer(f)))
    return dev, T


def test_preparar_reparte_tareas_y_rubricas_y_dividir_es_la_particion_de_d(preparado):
    dev, T = preparado
    assert (T / "choice.txt").read_text().split() == ["cfpb-quejas", "congreso-eeuu-partido", "finanzas-documentos", "usajobs-nivel"]
    assert (T / "score.txt").read_text().split() == ["horeca-resenas", "jql-valor-educativo"] and (T / "topk.txt").read_text() == ""
    rub = json.loads((T / "rubrica.json").read_text())["tareas"]
    assert set(rub) == {"horeca-resenas", "jql-valor-educativo"}
    fr = lb.leer(T / "rub" / "test" / "horeca-resenas.jsonl")[0]
    assert all("Ejemplo: «" in d for d in fr["descripciones"].values())
    for t in ("cfpb-quejas", "horeca-resenas"):
        todo = {r["id"] for r in lb.leer(T / "test" / (t + ".jsonl"))}
        b = {r["id"] for r in lb.leer(T / "testB" / (t + ".jsonl"))}
        a = {r["id"] for r in lb.leer(T / "testA" / (t + ".jsonl"))}
        assert a | b == todo and not a & b and all(mitad(i) == 1 for i in b) and all(mitad(i) == 0 for i in a)
    una = json.loads((T / "rubrica.json").read_text())
    una["tareas"].pop("jql-valor-educativo")
    (dev.parent / "una.json").write_text(json.dumps(una))
    with pytest.raises(SystemExit):   # la rúbrica debe cubrir exactamente las tareas score
        tl.main(["preparar", "--dev", str(dev), "--salida", str(dev.parent / "t2"), "--rubricas", str(dev.parent / "una.json")])


def test_bench_mide_caudal_y_latencia_y_zs_no_necesita_demos(preparado, servidor, tmp_path):
    _, T = preparado
    for cond in ("fs8-knn", "zs"):
        for fmt in ("chat", "harmony"):
            sal = tmp_path / ("b-%s-%s.json" % (cond, fmt))
            tl.main(["bench", "--t", str(T), "--url", servidor, "--modelo", "m", "--condicion", cond, "--concurrencia", "4", "--n", "20", "--salida", str(sal), "--formato", fmt])
            j = json.loads(sal.read_text())
            assert j["n"] == 20 and j["validas"] == 20 and j["req_s"] > 0 and j["p50_ms"] is not None and j["p95_ms"] >= j["p50_ms"] and j["concurrencia"] == 4
    filas, sop, demos = tl.filas_bench(T, 7, con_demos=False)
    assert len(filas) == 7 and sop == [] and demos == []
    assert all(mitad(r["id"]) == 0 for r in filas)   # filas de la mitad A: no las ve la medida de B antes del bench


# ---------------------------------------------------------------- medir
def escribir_preds(P, dev, acierto, ver_sop=True, tareas=None):
    """Predicciones inventadas de un modelo: fs8 acierta con probabilidad `acierto` (determinista por hash del id y del modelo); zs un poco menos."""
    def gen(r, p, sal):
        h = int(hashlib.sha256((sal + r["id"]).encode()).hexdigest(), 16) % 1000 / 1000
        ops = lb.orden(r)
        y = r["etiqueta"] if h < p else next(o for o in ops if o != r["etiqueta"])
        return {"id": r["id"], "probs": {o: (0.9 if o == y else 0.1 / (len(ops) - 1)) for o in ops}, "valida": True, "ms": 5.0}
    for t, (sop, ev) in dev.items():
        if tareas is not None and t not in tareas:
            continue
        sc = ev[0]["tipo"] == "score"
        sist = {"zs": (ev, acierto - 0.1), ("rub" if sc else "fs8-knn"): (ev, acierto)}
        if sc and ver_sop:
            sist["rub-sop"] = (sop, 0.8)
        for s, (rows, p) in sist.items():
            (P / s).mkdir(parents=True, exist_ok=True)
            (P / s / (t + ".jsonl")).write_text("".join(json.dumps(gen(r, p, s)) + "\n" for r in rows))


def bench_json(P, req_s, n=240):
    (P / "bench-fs8-knn-c32.json").write_text(json.dumps({"n": n, "validas": n, "req_s": req_s, "p50_ms": 100, "p95_ms": 300}))


@pytest.fixture
def mundo(tmp_path):
    dev_dir = tmp_path / "dev"
    crear(dev_dir, n_test=400, por_clase=12)
    T = tmp_path / "t"
    tl.main(["preparar", "--dev", str(dev_dir), "--salida", str(T), "--rubricas", "mecanica"])
    lic = tmp_path / "lic"
    lic.mkdir()
    for m in CRIT["modelos"]:
        (lic / (m + ".json")).write_text(json.dumps({"licencia": "apache-2.0", "permitida": True}))
    return dev_dir, T, lic, lb.cargar_dev(dev_dir)


def medir(mundo, tmp_path, pred):
    dev_dir, T, lic, _ = mundo
    tl.main(["medir", "--dev", str(dev_dir), "--pred", str(pred), "--rubrica", str(T / "rubrica.json"), "--licencias", str(lic), "--salida", str(tmp_path / "c.json")])
    return json.loads((tmp_path / "c.json").read_text())


def test_medir_cambia_si_gana_5_con_ic_positivo_y_caudal(mundo, tmp_path):
    pred = tmp_path / "pred"
    for m, acc, rs in (("qwen3-4b", 0.55, 6.0), ("qwen3-30b-a3b", 0.85, 4.2), ("gpt-oss-20b", 0.57, 9.0)):
        escribir_preds(pred / m, mundo[3], acc)
        bench_json(pred / m, rs)
    o = medir(mundo, tmp_path, pred)
    v = o["veredicto_por_candidato"]
    assert v["qwen3-30b-a3b"]["veredicto"] == "CAMBIAR" and v["qwen3-30b-a3b"]["macro_dif"] > 0.05 and v["qwen3-30b-a3b"]["ic95"][0] > 0
    assert v["gpt-oss-20b"]["veredicto"] == "EL 4B SE QUEDA" and "< +0.05" in v["gpt-oss-20b"]["motivos"][0]
    assert o["veredicto"]["decision"] == "CAMBIAR el L3 a qwen3-30b-a3b" and o["criterio_sha256"] == hashlib.sha256((AQUI / "crit-techo-local.json").read_bytes()).hexdigest()
    p = o["pareadas"]["qwen3-30b-a3b"]["fs8"]["B"]
    assert p["ic95"][0] <= p["macro_dif"] <= p["ic95"][1] and set(p["por_tarea"]) == set(CRIT["tareas_dev"])
    assert o["modelos"]["qwen3-4b"]["fs8"]["B"]["cobertura_total"] == 1.0 and o["modelos"]["qwen3-4b"]["zs"]["B"]["macro"]["valor"] < o["modelos"]["qwen3-4b"]["fs8"]["B"]["macro"]["valor"]
    assert set(o["modelos"]["qwen3-4b"]["umbrales"]) == {"horeca-resenas", "jql-valor-educativo"}
    assert medir(mundo, tmp_path, pred)["pareadas"] == o["pareadas"]   # semilla fija: mismo IC


def test_medir_caudal_bajo_no_cambia_aunque_gane(mundo, tmp_path):
    pred = tmp_path / "pred"
    for m, acc, rs in (("qwen3-4b", 0.55, 6.0), ("qwen3-30b-a3b", 0.85, 2.9)):
        escribir_preds(pred / m, mundo[3], acc)
        bench_json(pred / m, rs)
    v = medir(mundo, tmp_path, pred)["veredicto_por_candidato"]["qwen3-30b-a3b"]
    assert v["veredicto"] == "EL 4B SE QUEDA" and len(v["motivos"]) == 1 and "caudal 2.90" in v["motivos"][0]


def test_medir_ic_que_cruza_cero_no_cambia(mundo, tmp_path):
    """+5 en la media pero con la mitad B recortada a unas pocas filas el IC baja de cero: se pide el límite inferior > 0, no solo la diferencia."""
    pred = tmp_path / "pred"
    dev = {t: (s, ev[:24]) for t, (s, ev) in mundo[3].items()}   # pocas filas por tarea: IC ancho
    for m, acc in (("qwen3-4b", 0.55), ("qwen3-30b-a3b", 0.62)):
        escribir_preds(pred / m, dev, acc)
        bench_json(pred / m, 5.0)
    v = medir(mundo, tmp_path, pred)["veredicto_por_candidato"]["qwen3-30b-a3b"]
    assert v["veredicto"] in ("EL 4B SE QUEDA", "no medido") and (v["veredicto"] == "no medido" or any("IC95" in x or "<" in x for x in v["motivos"]))


def test_medir_no_medido_por_licencia_cobertura_y_bench(mundo, tmp_path):
    pred = tmp_path / "pred"
    for m, acc in (("qwen3-4b", 0.55), ("qwen3-30b-a3b", 0.85)):
        escribir_preds(pred / m, mundo[3], acc)
        bench_json(pred / m, 6.0)
    (pred / "gpt-oss-20b").mkdir()
    (mundo[2] / "qwen3-30b-a3b.json").write_text(json.dumps({"licencia": "other", "permitida": False}))
    o = medir(mundo, tmp_path, pred)
    assert o["veredicto_por_candidato"]["qwen3-30b-a3b"]["veredicto"] == "no medido" and "licencia" in o["veredicto_por_candidato"]["qwen3-30b-a3b"]["faltan"][0]
    assert o["veredicto"]["decision"] == "no medido"
    assert "cobertura" in o["veredicto_por_candidato"]["gpt-oss-20b"]["faltan"][0]   # directorio vacío: cobertura 0
    (mundo[2] / "qwen3-30b-a3b.json").write_text(json.dumps({"licencia": "apache-2.0", "permitida": True}))
    f = pred / "qwen3-30b-a3b" / "fs8-knn" / "cfpb-quejas.jsonl"   # fila sin predicción: cobertura < 1
    f.write_text("".join(f.read_text().splitlines(True)[:-5]))
    v = medir(mundo, tmp_path, pred)["veredicto_por_candidato"]["qwen3-30b-a3b"]
    assert v["veredicto"] == "no medido" and any("cobertura" in x for x in v["faltan"])
    (pred / "qwen3-30b-a3b" / "bench-fs8-knn-c32.json").unlink()
    f.write_text(f.read_text())
    assert any("caudal" in x for x in medir(mundo, tmp_path, pred)["veredicto_por_candidato"]["qwen3-30b-a3b"]["faltan"])


def test_medir_sin_rub_sop_la_tarea_score_queda_sin_cubrir(mundo, tmp_path):
    pred = tmp_path / "pred"
    escribir_preds(pred / "qwen3-4b", mundo[3], 0.55, ver_sop=False)
    bench_json(pred / "qwen3-4b", 6.0)
    o = medir(mundo, tmp_path, pred)
    assert o["modelos"]["qwen3-4b"]["fs8"]["B"]["cobertura_total"] < 1.0 and isinstance(o["modelos"]["qwen3-4b"]["umbrales"]["horeca-resenas"], str)


# ---------------------------------------------------------------- ud y empaquetador
UD = AQUI / "techo_local" / "techo_local.sh"
FASES = AQUI / "techo_local" / "techo_local_fases.sh"
SUST = {"__CHISPA_S3__": "s3://chispa-pruebas-879381269751/x", "__CRIT_SHA__": "0" * 64, "__COD_SHA256__": "1" * 64, "__DEV_SHA256__": "2" * 64, "__N_TEST__": "2930", "__N_SOP__": "728",
        "__REV_Q30__": "a" * 40, "__REV_OSS__": "b" * 40}


def test_ud_sintaxis_placeholders_tamano_y_contenido():
    ud = UD.read_text()
    assert set(re.findall(r"__[A-Z0-9_]+__", ud)) == set(SUST)
    for k, v in SUST.items():
        ud = ud.replace(k, v)
    assert len(ud.encode()) < 15500   # user-data de EC2: 16 384 bytes
    for f in (UD, FASES):
        assert subprocess.run(["bash", "-n", str(f)]).returncode == 0
    for s in ("shutdown -h +$MIN", "MIN=${CHISPA_MIN:-110}", "sha_ok \"$CRIT_SHA\"", "cdbee75f17c01a7cc42f958dc650907174af0554", "ESPERA_MIN", "verificado", "listo", "kill \"${WATCH"):
        assert s in ud, s
    fases = FASES.read_text()
    for s in ("--max-logprobs 20", "licencia --readme", "NO SE MIDE", "ignore_patterns=['original/*','metal/*']", "--verificar-harmony", "esperar topk.ok", "bench zs 32", "bench fs8-knn 32", "pesos-sha256", "fases_main"):
        assert s in fases, s
    assert fases.index("== 4. licencias") < fases.index("bajar_modelo()") < fases.index("arrancar()")   # la licencia se lee antes de bajar pesos
    m = re.search(r"MIN=\$\{CHISPA_MIN:-(\d+)\}", UD.read_text())
    assert int(m.group(1)) == CRIT["instancia"]["tope_min"]
    assert "aws s3 cp" in UD.read_text() and "chispa-pruebas" not in FASES.read_text()   # nada se sube a otro sitio: solo $CHISPA_S3


def test_ud_no_sube_nada_fuera_de_res():
    for f in (UD, FASES):
        t = f.read_text()
        assert not re.search(r"\bcurl\b[^\n]*(-T|--upload|-X *P|-d )", t) and "scp " not in t and "rsync" not in t
    assert UD.read_text().count("s3cp \"$BASE/") == 3   # resultados.tgz, log.txt, listo


def empaquetar(tmp_path, env):
    sal, udd = tmp_path / "sal", tmp_path / "ud"
    dev = tmp_path / "dev"
    crear(dev, n_test=10, por_clase=2)
    r = subprocess.run(["bash", str(AQUI / "techo_local" / "empaquetar_techo_local.sh"), "HEAD", str(dev), str(sal)], cwd=RAIZ, capture_output=True, text=True, env={**os.environ, "CHISPA_UD_DIR": str(udd), **env})
    return r, sal, udd, dev


def en_head():
    return subprocess.run(["git", "cat-file", "-e", "HEAD:model/alt/techo_local/empaquetar_techo_local.sh"], cwd=RAIZ).returncode == 0 and \
        subprocess.run(["git", "cat-file", "-e", "HEAD:model/alt/techo_local/techo_local_fases.sh"], cwd=RAIZ).returncode == 0


@pytest.mark.skipif(not en_head(), reason="el empaquetador trabaja sobre lo commiteado")
def test_empaquetador_en_seco_sin_placeholders_y_se_niega_sin_congelar(tmp_path):
    r, sal, udd, dev = empaquetar(tmp_path, {"CHISPA_ENSAYO": "1"})
    assert r.returncode == 0, r.stderr + r.stdout
    uds = list(udd.glob("ud-techo-local-*.sh"))
    assert len(uds) == 1
    ud = uds[0].read_text()
    assert not re.findall(r"__[A-Z0-9_]+__", ud) and "REV30=" + "0" * 40 in ud and "shutdown -h +$MIN" in ud
    assert (sal / "codigo.tgz").exists() and (sal / "dev.tgz").exists() and "ENSAYO" in r.stdout and "110" in r.stdout and "3.41" in r.stdout
    nombres = subprocess.run(["tar", "-tzf", str(sal / "dev.tgz")], capture_output=True, text=True).stdout.split()
    assert len(nombres) == 6 and not any("sellad" in n for n in nombres)
    cod = subprocess.run(["tar", "-tzf", str(sal / "codigo.tgz")], capture_output=True, text=True).stdout
    assert "model/alt/techo_local/techo_local_fases.sh" in cod and "techo_local/lanzados" not in cod and "aws/lanzados" not in cod and "model/alt/resultados" not in cod and "._" not in cod
    # sin ensayo: el criterio está en BORRADOR (revisiones PENDIENTE) y sin sha congelado: no empaqueta
    r2, *_ = empaquetar(tmp_path / "otro", {"CHISPA_ENSAYO": "0"})
    assert r2.returncode != 0 and "CRIT_SHA256_ESPERADO" in r2.stderr
    crit = hashlib.sha256(subprocess.run(["git", "show", "HEAD:model/alt/crit-techo-local.json"], cwd=RAIZ, capture_output=True).stdout).hexdigest()
    r3, *_ = empaquetar(tmp_path / "otro2", {"CHISPA_ENSAYO": "0", "CRIT_SHA256_ESPERADO": crit})
    assert r3.returncode != 0 and "no está fijada" in r3.stderr
    r4, *_ = empaquetar(tmp_path / "otro3", {"CHISPA_ENSAYO": "0", "CRIT_SHA256_ESPERADO": "f" * 64})
    assert r4.returncode != 0 and "no es el congelado" in r4.stderr


def test_ud_en_local_con_vllm_falso_de_punta_a_punta(tmp_path, servidor):
    """El ud real en modo local: tres modelos contra el servidor falso; el de licencia GPL no se mide; sale cifras.json con veredicto y solo res/ en resultados.tgz."""
    pytest.importorskip("torch")
    sint = tmp_path / "sint"
    r = subprocess.run(["python", "-m", "model.alt.destilar_biencoder", "sintetico", "--salida", str(sint)], cwd=RAIZ, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    # codigo.tgz desde el árbol de trabajo (el empaquetador real trabaja sobre un commit y se prueba aparte)
    s3, base = tmp_path / "s3", tmp_path / "base"
    s3.mkdir()
    subprocess.run("tar -czf %s -C %s --exclude='model/alt/resultados*' --exclude='model/alt/techo_local/lanzados' --exclude=__pycache__ model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2 sdk/python 2>/dev/null; true" % (s3 / "codigo.tgz", RAIZ), shell=True)
    subprocess.run(["tar", "-czf", str(s3 / "dev.tgz"), "-C", str(sint / "dev")] + sorted(p.name for p in (sint / "dev").glob("*.jsonl")), check=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    dev = lb.cargar_dev(sint / "dev")
    ud = UD.read_text()
    sust = dict(SUST, __CHISPA_S3__="s3://x", __CRIT_SHA__=sha(AQUI / "crit-techo-local.json"), __COD_SHA256__=sha(s3 / "codigo.tgz"), __DEV_SHA256__=sha(s3 / "dev.tgz"),
                __N_TEST__=str(sum(len(e) for _, e in dev.values())), __N_SOP__=str(sum(len(s) for s, _ in dev.values())))
    for k, v in sust.items():
        ud = ud.replace(k, v)
    (tmp_path / "ud.sh").write_text(ud)
    env = {**os.environ, "CHISPA_LOCAL": "1", "CHISPA_S3_LOCAL": str(s3), "CHISPA_BASE": str(base), "CHISPA_PY": subprocess.run(["which", "python"], capture_output=True, text=True).stdout.strip(),
           "CHISPA_LOCAL_URL": servidor, "CHISPA_LOCAL_E5": str(sint / "e5"), "CHISPA_LOCAL_N": "16", "CHISPA_HILOS": "4", "CHISPA_MIN": "60", "CHISPA_ESPERA_MIN": "1",
           "CHISPA_LOCAL_LIC_GPT_OSS_20B": "gpl-3.0", "CHISPA_TOPK_UMBRAL": "2"}
    r = subprocess.run(["bash", str(tmp_path / "ud.sh")], cwd=RAIZ, capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stdout[-1500:]
    res = base / "res"
    est = (res / "estado.tsv").read_text()
    assert "[gpt-oss-20b]" not in est and "gpt-oss-20b\tNO SE MIDE" in est and "licencia-gpt-oss-20b\tgpt-oss-20b: licencia gpl-3.0 -> NO se mide" in est
    assert not (res / "FALLO").exists(), (res / "FALLO").read_text()
    c = json.loads((res / "cifras.json").read_text())
    assert c["veredicto"]["decision"] == "EL 4B SE QUEDA" and c["veredicto_por_candidato"]["gpt-oss-20b"]["veredicto"] == "no medido"
    assert c["modelos"]["qwen3-30b-a3b"]["fs8"]["B"]["cobertura_total"] == 1.0 and c["modelos"]["qwen3-4b"]["caudal"]["validas"] == 16
    assert (res / "pred" / "qwen3-4b" / "rub-sop").exists() and (res / "pred" / "qwen3-4b" / "fs8-knn" / "cfpb-quejas.jsonl").exists() and not (res / "pred" / "gpt-oss-20b" / "zs").exists()
    assert (res / "licencias" / "gpt-oss-20b.json").exists() and (res / "licencias" / "qwen3-4b-README.md").exists()
    import tarfile
    with tarfile.open(s3 / "resultados.tgz") as t:
        nombres = t.getnames()
    assert all(n == "res" or n.startswith("res/") for n in nombres)
    texto_dev = {json.loads(l)["texto"] for f in (sint / "dev").glob("*.jsonl") for l in f.read_text().splitlines()}
    blob = b""
    for p in (base / "res").rglob("*.jsonl"):
        blob += p.read_bytes()
    assert not any(t.encode() in blob for t in list(texto_dev)[:200] if len(t) > 12)   # ningún texto de dev en lo que sube
    assert not np.isnan(c["modelos"]["qwen3-4b"]["fs8"]["B"]["macro"]["valor"])
