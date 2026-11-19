"""Pruebas de la batería v3 con datos INVENTADOS (model/datos/tests/fixtures_v3/generar.py); sin red."""
import json
import os
import stat
from pathlib import Path

import pytest

from model.datos import bateria_v3 as b
from model.datos import bateria_v3_licencias as lic
from model.datos import bateria_v3_solapes as sol
from model.datos.bateria_v3_candidatas import CANDIDATAS
from model.datos.tests.fixtures_v3.generar import generar

CLAVES_LECTOR = {"formato", "ficheros", "columnas", "texto", "etiqueta", "split_por_fichero", "split_col", "filtro", "mapa_etiqueta", "prefijo_hasta", "etiqueta_par", "cuantiles",
                 "contar_patron", "pares", "etiqueta_desde_carpeta", "vacio_es", "clases_max", "min_por_clase", "muestra_por_clase", "codificacion", "pendiente"}


# ----------------------------------------------------------------------------------------------------------------- registro de candidatas y manifiesto
def test_candidatas_ocho_por_tipo_y_sin_duplicar_familia():
    por = {t: [c for c in CANDIDATAS if c["tipo"] == t] for t in b.TIPOS}
    assert all(len(v) >= 8 for v in por.values()), {k: len(v) for k, v in por.items()}
    assert len({c["nombre"] for c in CANDIDATAS}) == len(CANDIDATAS)
    assert len({c["familia"] for c in CANDIDATAS}) == len(CANDIDATAS)
    assert {c["idioma"] for c in CANDIDATAS} == {"es", "en"}


def test_candidatas_bien_formadas():
    for c in CANDIDATAS:
        assert set(c["lector"]) <= CLAVES_LECTOR, c["nombre"]
        assert c["fuente"]["tipo"] in ("hf", "url") and c["pregunta"] and c["no_solapa"] and c["riesgos"]
        if not c["lector"].get("pendiente"):
            assert c["lector"]["formato"] and c["lector"]["ficheros"]
            assert c["lector"].get("texto") or c["lector"].get("pares")
        if c["fuente"]["tipo"] == "url":
            assert c.get("licencia_regex_ficha"), c["nombre"]       # sin regex no hay forma automática de admitir una fuente por URL
        assert lic.normalizar(c["licencia_esperada"]) in lic.ADMITIDAS, c["nombre"]


def test_manifiesto_del_repo_es_el_inicial_sin_nada_verificado():
    m = json.loads((b.AQUI / "bateria-v3-manifiesto.json").read_text(encoding="utf-8"))
    assert m == json.loads(json.dumps(b.manifiesto_inicial()))          # si cambia una candidata hay que regenerar (manifiesto-inicial)
    for t in m["tareas"]:
        assert t["licencia_estado"] == b.LICENCIA_PENDIENTE and t["sha256"] == "" and t["licencia_sha256"] == "" and t["rev"] is None and t["estado"] == "candidata"
    assert "abrir_sellada" in m["procedimiento_abrir_sellada"] and len(m["reservas_v2"]) == 4 and len(m["descartadas"]) >= 20
    assert (b.AQUI / "bateria-v3-candidatas.md").read_text(encoding="utf-8") == b.candidatas_md(m)
    assert (b.AQUI / "bateria-v3-aperturas.jsonl").read_text(encoding="utf-8") == ""


# ----------------------------------------------------------------------------------------------------------------- licencias
def tarjeta(lic_, cuerpo=""):
    return "---\nlicense: %s\n---\n%s" % (lic_, cuerpo)


@pytest.mark.parametrize("cab,ok", [("cc-by-4.0", True), ("cc0-1.0", True), ("MIT", True), ("apache-2.0", True), ("odc-by", True), ("cdla-permissive-2.0", True),
                                    ("cc-by-sa-4.0", False), ("cc-by-nc-4.0", False), ("cc-by-nd-4.0", False), ("other", False), ("unknown", False), ("gpl-3.0", False),
                                    ("odbl", False), ("cdla-sharing-1.0", False), ("cc-by-nc-sa-4.0", False)])
def test_licencia_de_cabecera(cab, ok):
    assert (lic.evaluar_tarjeta(tarjeta(cab))["estado"] == "admitida") is ok


def test_licencia_ausente_multiple_o_contradictoria_queda_fuera():
    assert lic.evaluar_tarjeta("")["estado"] == "fuera"
    assert "ausente" in lic.evaluar_tarjeta("# solo cuerpo")["motivo"]
    assert lic.evaluar_tarjeta("---\nlicense:\n- mit\n- cc-by-4.0\n---\n")["estado"] == "fuera"
    assert lic.evaluar_tarjeta(tarjeta("mit", "Uso solo para investigación."))["estado"] == "fuera"
    assert lic.evaluar_tarjeta(tarjeta("cc-by-4.0", "Non-commercial only"))["estado"] == "fuera"
    assert lic.evaluar_tarjeta(tarjeta("cc-by-4.0", "datos de Reddit"))["estado"] == "fuera"
    assert lic.evaluar_tarjeta(tarjeta("cc-by-4.0", "tuits"))["estado"] == "admitida"        # «tuits» en español no dispara: solo twitter/tweet(s)/x.com
    assert lic.evaluar_tarjeta(tarjeta("mit"), origen_declarado="comentarios de Reddit")["estado"] == "fuera"


def test_licencia_esperada_distinta_solo_avisa_si_las_dos_son_admitidas():
    r = lic.evaluar_tarjeta(tarjeta("mit"), esperada="cc-by-4.0")
    assert r["estado"] == "admitida" and r["avisos"]


def test_ficha_por_url_exige_el_texto():
    assert lic.evaluar_ficha("Esto es dominio público", r"dominio p[uú]blico")["estado"] == "admitida"
    assert lic.evaluar_ficha("nada", r"dominio p[uú]blico")["estado"] == "fuera"
    assert lic.evaluar_ficha("", r"x")["estado"] == "fuera"
    assert lic.evaluar_ficha("public domain, pero CC BY-NC", r"public domain")["estado"] == "fuera"


# ----------------------------------------------------------------------------------------------------------------- solapes
def test_huella_y_jaccard():
    a = "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince"
    assert sol.huella(a) == sol.huella("  UNO dos  tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince ")
    cercano = a + " dieciseis"
    ajeno = "otro texto sin nada que ver con el anterior pero igual de largo para comparar bien los sketches de ambos"
    assert sol.jaccard(sol.sketch(a), sol.sketch(a)) == 1.0
    assert sol.jaccard(sol.sketch(a), sol.sketch(cercano)) >= sol.UMBRAL_DESCARTE
    assert sol.jaccard(sol.sketch(a), sol.sketch(ajeno)) < sol.UMBRAL_MARCA
    assert sol.sketch("") == []


def test_referencia_exacto_casi_marca_y_limpio():
    base = " ".join("w%d" % i for i in range(40))
    marca = " ".join("w%d" % i for i in range(10)) + " " + " ".join("z%d" % i for i in range(30))   # comparte 6 de 36 n-gramas: limpio
    medio = " ".join("w%d" % i for i in range(30)) + " " + " ".join("y%d" % i for i in range(10))   # comparte 26 de 46 n-gramas (Jaccard 0,57): marca
    r = sol.Referencia()
    i_exacto, i_casi, i_ajeno = r.añadir("t", 0, base), r.añadir("t", 1, base + " extra1"), r.añadir("t", 2, marca)
    r.comparar([sol.huella(base)], "v2")
    assert r.veredicto(i_exacto) == "descartar" and r.res[i_exacto]["tipo"] == "exacto"
    assert r.veredicto(i_casi) == "descartar" and r.res[i_casi]["tipo"] == "casi"
    assert r.veredicto(i_ajeno) == "limpio"
    r2 = sol.Referencia()
    j = r2.añadir("t", 0, medio)
    r2.comparar([sol.huella(base)], "entrenamiento")
    assert r2.veredicto(j) == "marcar" and r2.res[j]["fuente"] == "entrenamiento"


def test_las_huellas_no_llevan_texto(tmp_path):
    (tmp_path / "x.jsonl").write_text(json.dumps({"texto": "Hola mundo inventado de prueba"}) + "\n", encoding="utf-8")
    n, _ = sol.escribir_huellas([tmp_path], tmp_path / "huellas-x.jsonl.gz", "x")
    assert n == 1
    import gzip
    assert "inventado" not in gzip.open(tmp_path / "huellas-x.jsonl.gz", "rt").read()
    malo = tmp_path / "huellas-malo.jsonl"
    malo.write_text(json.dumps({"h": "a", "s": [1], "texto": "en claro"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(sol.leer_huellas(malo))


# ----------------------------------------------------------------------------------------------------------------- lectores
def test_etiquetas_derivadas():
    assert b._etiqueta(dict(contar_patron=dict(col="a", patron="<<", min=2, max=4)), {"a": "x <<1>> <<2>> <<3>> <<4>> <<5>>"}) == "3"
    assert b._etiqueta(dict(contar_patron=dict(col="a", patron="<<", min=2, max=4)), {"a": "sin"}) == "1"
    assert b._etiqueta(dict(etiqueta_par=dict(col="l", nombre="q")), {"l": {"name": ["x", "q"], "value": [0.1, 0.7]}}) == 0.7
    assert b._etiqueta(dict(etiqueta_par=dict(col="l", nombre="q")), {"l": {"name": ["x"], "value": [0.1]}}) is None
    assert b._etiqueta(dict(etiqueta="c", prefijo_hasta=":"), {"c": "FRENOS: ABS"}) == "FRENOS"
    assert b._etiqueta(dict(etiqueta="c", vacio_es="N", mapa_etiqueta=dict(Y="si", N="no")), {"c": ""}) == "no"
    assert b._etiqueta(dict(etiqueta="c"), {"c": ""}) is None
    assert b._etiqueta(dict(etiqueta="c.d", mapa_etiqueta={"PHASE2": "2"}), {"c": {"d": ["PHASE2"]}}) == "2"
    assert b._etiqueta(dict(etiqueta="c", mapa_etiqueta={"PHASE2": "2"}), {"c": ["PHASE1", "PHASE2"]}) is None     # varias fases: sin etiqueta
    assert b._texto(dict(texto=["q", "c.xs"]), {"q": "pregunta", "c": {"xs": ["a", "b"]}}) == "pregunta\na b"


# ----------------------------------------------------------------------------------------------------------------- construcción en seco
@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    d = tmp_path_factory.mktemp("v3")
    fx = generar(d / "fx")
    os.environ["CHISPA_V3_ESPEJO"] = str(fx["espejo"])
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            man = b.construir(fx["manifiesto"], d / "trabajo", d / "salida", huellas=fx["huellas"])
    finally:
        os.environ.pop("CHISPA_V3_ESPEJO", None)
    return dict(man=man, salida=d / "salida", stdout=buf.getvalue(), fx=fx, dir=d)


def tarea(c, nombre):
    return next(t for t in c["man"]["tareas"] if t["nombre"] == nombre)


def test_estados_y_motivos(corrida):
    est = {t["nombre"]: t["estado"] for t in corrida["man"]["tareas"]}
    assert est == {"mini-choice": "construida", "mini-yesno": "construida", "mini-score": "construida", "mini-parquet": "construida", "mini-nc": "fuera", "mini-sa": "fuera",
                   "mini-sinlic": "fuera", "mini-tuits": "fuera", "mini-url": "construida", "mini-zip": "construida", "mini-pendiente": "pendiente"}
    assert "cc-by-nc" in tarea(corrida, "mini-nc")["motivo_fuera"] and "SA" in tarea(corrida, "mini-sa")["motivo_fuera"]
    assert "Twitter" in tarea(corrida, "mini-tuits")["motivo_fuera"] and "pendiente" in tarea(corrida, "mini-pendiente")["motivo_fuera"]
    r = corrida["man"]["resumen"]
    assert r["por_tipo"]["choice"]["construidas"] == 2 and r["fuera"]["mini-nc"]


def test_licencias_guardadas_con_sha(corrida):
    for t in corrida["man"]["tareas"]:
        if t["estado"] in ("construida", "fuera") and "licencia:" in (t["motivo_fuera"] or "licencia:"):
            f = corrida["salida"] / "licencias" / (t["nombre"] + ".txt")
            assert f.exists() and t["licencia_sha256"] == b.lic.sha256_texto(f.read_text(encoding="utf-8"))
    assert tarea(corrida, "mini-zip")["licencia"] == "cc-by-4.0" and tarea(corrida, "mini-url")["licencia"] == "dominio-publico"


def test_recuentos_y_sha(corrida):
    c = tarea(corrida, "mini-choice")
    assert c["n_soporte"] == 24 and c["cifras"]["soporte_por_clase"] == {"cat-a": 8, "cat-b": 8, "cat-c": 8}
    assert c["cifras"]["duplicados"] >= 1 and c["cifras"]["conflicto"] >= 2
    assert c["solapes"]["descartadas"] == 2 and c["solapes"]["por_fuente"] == {"v2": 2}      # el texto exacto y el casi-duplicado de la «referencia»
    for t in corrida["man"]["tareas"]:
        if t["estado"] == "construida":
            assert t["sha256"] == b.sha256_fichero(corrida["salida"] / t["fichero_test"]) and len(t["sha256"]) == 64 and t["n_test"] >= b.MIN_TEST_TAREA
            assert len((corrida["salida"] / t["fichero_test"]).read_text(encoding="utf-8").splitlines()) == t["n_test"]
    y = tarea(corrida, "mini-yesno")
    assert y["clases"] == ["si", "no"] and y["cifras"]["soporte_por_clase"] == {"si": 8, "no": 8}          # los pares bueno/malo van juntos
    assert tarea(corrida, "mini-score")["clases"] == ["1", "2", "3", "4", "5"] and tarea(corrida, "mini-parquet")["clases"] == ["1", "2", "3", "4"]
    assert tarea(corrida, "mini-url")["clases"] == ["FRENOS", "MOTOR"] and tarea(corrida, "mini-zip")["clases"] == ["1", "2", "3"]


def test_soporte_y_test_no_comparten_texto_ni_par(corrida):
    for t in corrida["man"]["tareas"]:
        if t["estado"] != "construida":
            continue
        sop = [json.loads(l) for l in (corrida["salida"] / t["fichero_soporte"]).read_text(encoding="utf-8").splitlines()]
        ev = [json.loads(l) for l in (corrida["salida"] / t["fichero_test"]).read_text(encoding="utf-8").splitlines()]
        assert not {b.norm(f["texto"]) for f in sop} & {b.norm(f["texto"]) for f in ev}
        assert {f["split"] for f in sop} == {"train"} and {f["split"] for f in ev} == {"test"}
        assert len({b.norm(f["texto"]) for f in ev}) == len(ev)


def test_sellada_solo_lectura_y_stdout_sin_texto(corrida):
    for f in (corrida["salida"] / "sellada").glob("*.jsonl"):
        assert stat.S_IMODE(f.stat().st_mode) == 0o444
    textos = [json.loads(l)["texto"] for f in (corrida["salida"] / "sellada").glob("*.jsonl") for l in f.read_text(encoding="utf-8").splitlines()]
    assert len(textos) > 1000
    assert not [x for x in textos if x[:40] in corrida["stdout"]]
    assert "clasea" not in corrida["stdout"]


def test_determinista(corrida, tmp_path):
    os.environ["CHISPA_V3_ESPEJO"] = str(corrida["fx"]["espejo"])
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            m2 = b.construir(corrida["fx"]["manifiesto"], tmp_path / "t", tmp_path / "s", huellas=corrida["fx"]["huellas"])
    finally:
        os.environ.pop("CHISPA_V3_ESPEJO", None)
    assert {t["nombre"]: t["sha256"] for t in m2["tareas"]} == {t["nombre"]: t["sha256"] for t in corrida["man"]["tareas"]}


def test_ficha_por_url_inexistente_deja_fuera(corrida, tmp_path):
    m = json.loads(Path(corrida["fx"]["manifiesto"]).read_text(encoding="utf-8"))
    t = next(t for t in m["tareas"] if t["nombre"] == "mini-url")
    t["url_ficha"] = (tmp_path / "no-existe.txt").as_uri()
    m["tareas"] = [t]
    (tmp_path / "m.json").write_text(json.dumps(m), encoding="utf-8")
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        r = b.construir(tmp_path / "m.json", tmp_path / "t", tmp_path / "s")
    assert r["tareas"][0]["estado"] == "fuera" and "ficha" in r["tareas"][0]["motivo_fuera"]


def test_tarea_con_pocas_filas_queda_fuera(corrida, tmp_path):
    m = json.loads(Path(corrida["fx"]["manifiesto"]).read_text(encoding="utf-8"))
    m["tareas"] = [t for t in m["tareas"] if t["nombre"] == "mini-choice"]
    (tmp_path / "m.json").write_text(json.dumps(m), encoding="utf-8")
    os.environ["CHISPA_V3_ESPEJO"] = str(corrida["fx"]["espejo"])
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            r = b.construir(tmp_path / "m.json", tmp_path / "t", tmp_path / "s", tope=10)
    finally:
        os.environ.pop("CHISPA_V3_ESPEJO", None)
    assert r["tareas"][0]["estado"] == "fuera" and "mínimo" in r["tareas"][0]["motivo_fuera"]


# ----------------------------------------------------------------------------------------------------------------- abrir_sellada
def test_abrir_sellada_anota_antes_de_leer(corrida, tmp_path):
    t = tarea(corrida, "mini-choice")
    reg = tmp_path / "aperturas.jsonl"
    filas = b.abrir_sellada(corrida["salida"] / t["fichero_test"], "prueba de apertura inventada", registro=reg, manifiesto=corrida["salida"] / "manifiesto.json", quien="tester")
    l = json.loads(reg.read_text(encoding="utf-8"))
    assert len(filas) == t["n_test"] and l["sha256"] == t["sha256"] and l["coincide_con_manifiesto"] is True and l["quien"] == "tester" and l["fichero"] == "mini-choice.jsonl"


def test_abrir_sellada_exige_motivo_y_carpeta(corrida, tmp_path):
    reg = tmp_path / "a.jsonl"
    f = corrida["salida"] / "sellada" / "mini-choice.jsonl"
    with pytest.raises(PermissionError):
        b.abrir_sellada(f, "corto", registro=reg)
    with pytest.raises(PermissionError):
        b.abrir_sellada(f, "   ", registro=reg)
    with pytest.raises(ValueError):
        b.abrir_sellada(corrida["salida"] / "soporte" / "mini-choice.jsonl", "motivo largo y claro", registro=reg)
    assert not reg.exists()


def test_abrir_sellada_con_sha_distinto_anota_y_se_niega(corrida, tmp_path):
    d = tmp_path / "sellada"
    d.mkdir()
    f = d / "mini-choice.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    reg = tmp_path / "a.jsonl"
    with pytest.raises(ValueError):
        b.abrir_sellada(f, "comprobar un fichero alterado", registro=reg, manifiesto=corrida["salida"] / "manifiesto.json")
    assert json.loads(reg.read_text(encoding="utf-8"))["coincide_con_manifiesto"] is False


def test_auditar_log(corrida, tmp_path):
    limpio = tmp_path / "limpio.log"
    limpio.write_text("HECHA mini-choice soporte=24 test=207\n", encoding="utf-8")
    assert b.auditar_log(corrida["salida"], limpio, registro=tmp_path / "ap.jsonl") == 0
    primera = json.loads((corrida["salida"] / "sellada" / "mini-choice.jsonl").read_text(encoding="utf-8").splitlines()[0])["texto"]
    sucio = tmp_path / "sucio.log"
    sucio.write_text("algo %s\n" % primera, encoding="utf-8")
    assert b.auditar_log(corrida["salida"], sucio, registro=tmp_path / "ap.jsonl") >= 1
    assert len((tmp_path / "ap.jsonl").read_text(encoding="utf-8").splitlines()) >= 2 * len(list((corrida["salida"] / "sellada").glob("*.jsonl")))
