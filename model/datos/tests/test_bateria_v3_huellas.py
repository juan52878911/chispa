"""Huellas de v2 y del entrenamiento dentro del ud (W3b), con datos INVENTADOS y sin red."""
import gzip
import io
import json
import tarfile

import pytest

from model.datos import bateria_v3 as b
from model.datos import bateria_v3_huellas as h
from model.datos import bateria_v3_solapes as sol
from model.datos.tests.fixtures_v3.generar import generar


@pytest.fixture()
def fx(tmp_path, monkeypatch):
    f = generar(tmp_path / "fx")
    monkeypatch.setenv("CHISPA_V3_ESPEJO", str(f["espejo"]))
    return f


def leer(ruta):
    return [json.loads(l) for l in gzip.open(ruta, "rt", encoding="utf-8")]


def test_repos_de_y_clasificacion_del_manifiesto_real():
    assert h.repos_de("FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_1 y _2") == ["FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_1", "FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_2"]
    assert h.repos_de("tasksource/it-support-tickets") == ["tasksource/it-support-tickets"] and h.repos_de("buzón interno de quejas") == []
    c = h.clasificar_entrenamiento(h.cargar_entrenamiento())
    assert {x["nombre"] for x in c["publicos"]} == {"toxicidad_protestas", "toxicidad_constitucion", "goemociones_es", "spam_tanaos", "soporte_ti", "ciberacoso", "tsjev"}
    no = {x["nombre"] for x in c["no_comprobado"]}
    assert {"PQRS de datos.gov.co", "Clandestino", "ALIA", "XNLI"} <= no and all(x["motivo"] for x in c["no_comprobado"])
    m = json.loads((b.AQUI / "bateria-v3-manifiesto.json").read_text(encoding="utf-8"))
    assert m["solape_no_comprobado"] == json.loads(json.dumps(c["no_comprobado"]))


def test_extraccion_segura_rechaza_rutas_y_enlaces(tmp_path):
    def tgz(nombre, miembros):
        with tarfile.open(tmp_path / nombre, "w:gz") as tf:
            for ti, datos in miembros:
                tf.addfile(ti, io.BytesIO(datos) if datos is not None else None)
        return tmp_path / nombre

    def ti(nombre, tipo=tarfile.REGTYPE, link=""):
        t = tarfile.TarInfo(nombre)
        t.type, t.linkname = tipo, link
        t.size = 1 if tipo == tarfile.REGTYPE else 0
        return t
    assert h.extraer_tgz(tgz("ok.tgz", [(ti("a.jsonl"), b"x"), (ti("._a.jsonl"), b"x")]), tmp_path / "ok") == 1 and not (tmp_path / "ok" / "._a.jsonl").exists()
    for malo in (ti("../fuera"), ti("/abs"), ti("l", tarfile.SYMTYPE, "/etc/passwd")):
        with pytest.raises(RuntimeError):
            h.extraer_tgz(tgz("m.tgz", [(malo, b"x" if malo.type == tarfile.REGTYPE else None)]), tmp_path / ("m" + malo.name.replace("/", "_")))
    with pytest.raises(RuntimeError):                               # la carpeta tiene que ser nueva y vacía
        h.extraer_tgz(tmp_path / "ok.tgz", tmp_path / "ok")


def test_huellas_v2_sin_texto_y_con_recuento(fx, tmp_path, capsys):
    sal = tmp_path / "hue"
    assert h.main(["v2", "--dev", str(fx["dev"]), "--selladas", str(fx["selladas"]), "--trabajo", str(tmp_path / "t"), "--salida", str(sal)]) == 0
    hs = leer(sal / "huellas-v2.jsonl.gz")
    assert hs and all(set(x) == {"h", "s", "f"} and x["f"] == "v2" for x in hs) and sol.huella(fx["ref_exacto"])["h"] in {x["h"] for x in hs}
    assert fx["ref_exacto"][:30] not in (sal / "huellas-v2.jsonl.gz").read_bytes().decode("latin-1") and fx["ref_exacto"][:30] not in capsys.readouterr().out
    est = json.loads((sal / "estado-huellas.json").read_text(encoding="utf-8"))
    assert est["v2"]["ficheros"] == {"dev": 6, "selladas": 8} and est["v2"]["filas_sin_campo_de_texto"] == 0
    # nº de ficheros distinto del esperado: falla y no deja huellas a medias como buenas
    assert h.main(["v2", "--dev", str(fx["dev"]), "--selladas", str(fx["selladas"]), "--trabajo", str(tmp_path / "t2"), "--salida", str(tmp_path / "h2"), "--n-selladas", "9"]) == 1


def test_huellas_v2_falla_si_las_filas_no_traen_texto(fx, tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    for k in range(6):
        (d / ("d%d.jsonl" % k)).write_text("".join(json.dumps({"otro": "x" * 30}) + "\n" for _ in range(5)), encoding="utf-8")
    t = tmp_path / "d.tgz"
    with tarfile.open(t, "w:gz") as tf:
        for f in sorted(d.glob("*.jsonl")):
            tf.add(f, arcname=f.name)
    assert h.main(["v2", "--dev", str(t), "--selladas", str(fx["selladas"]), "--trabajo", str(tmp_path / "t"), "--salida", str(tmp_path / "s")]) == 1


def test_huellas_entrenamiento_publicas_si_y_no_publicas_como_no_comprobadas(fx, tmp_path):
    sal, mu = tmp_path / "hue", tmp_path / "mu"
    assert h.main(["entrenamiento", "--manifiesto", str(fx["entrenamiento"]), "--trabajo", str(tmp_path / "t"), "--salida", str(sal), "--muestra", str(mu)]) == 0
    hs = leer(sal / "huellas-entrenamiento.jsonl.gz")
    assert len(hs) == 12 and all(set(x) == {"h", "s", "f"} for x in hs) and sol.huella(fx["ref_entren"])["h"] in {x["h"] for x in hs}
    e = json.loads((sal / "estado-huellas.json").read_text(encoding="utf-8"))
    assert [c["nombre"] for c in e["entrenamiento"]["comprobados"]] == ["publica", "tabla"] and e["entrenamiento"]["comprobados"][0]["rev"].startswith("espejo-")
    assert {x["nombre"] for x in e["no_comprobado"]} == {"sindatos", "interna", "Fuente Propia A", "inventada-v1"} and e == json.loads((sal / "estado-huellas.json").read_text(encoding="utf-8"))
    assert not (tmp_path / "t" / "entrenamiento" / "ent__publica").exists()                    # el texto bajado se borra al terminar cada fuente
    assert {p.name for p in mu.iterdir()} == {"ent__publica.jsonl.gz", "ent__tabla.jsonl.gz"}


def test_entrenamiento_con_tope_de_tiempo_deja_la_fuente_como_parcial(fx, tmp_path):
    sal = tmp_path / "hue"
    n, comp, no = h.huellas_entrenamiento(str(fx["entrenamiento"]), tmp_path / "t", sal, tmp_path / "mu", tope_min=-1)     # tiempo agotado desde el principio
    assert n == 0 and not comp and any("sin ficheros" in x["motivo"] or "sin tiempo" in x["motivo"] for x in no)


def test_ciberacoso_se_huella_con_y_sin_menciones(tmp_path, monkeypatch):
    esp = tmp_path / "hf" / "FelipeGuerra__Colombian_Spanish_Cyberbullying_Dataset_1"
    esp.mkdir(parents=True)
    t = "hola @pedro_88 esto es un mensaje bastante largo para contar"
    (esp / "d.jsonl").write_text(json.dumps({"texto": t}) + "\n", encoding="utf-8")
    monkeypatch.setenv("CHISPA_V3_ESPEJO", str(tmp_path))
    ent = tmp_path / "e.json"
    ent.write_text(json.dumps(dict(conjuntos=[dict(nombre="ciberacoso", fuente="FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_1")], exclusiones={})), encoding="utf-8")
    h.huellas_entrenamiento(str(ent), tmp_path / "t", tmp_path / "s", tmp_path / "m")
    hh = {x["h"] for x in leer(tmp_path / "s" / "huellas-entrenamiento.jsonl.gz")}
    assert sol.huella(t)["h"] in hh and sol.huella(t.replace("@pedro_88", "@usuario"))["h"] in hh


def test_fugas_en_log_detecta_ocho_palabras_seguidas():
    log = b.norm("HECHA x soporte=3 y un texto cualquiera: alfa beta gamma delta épsilon zeta eta theta fin")
    n, f = h.fugas_en_log(log, ["nada que ver con el log de esta corrida", "empieza Alfa beta gamma delta épsilon zeta eta theta y sigue", "alfa beta gamma delta épsilon zeta eta"])
    assert (n, f) == (3, 1)


def test_auditar_log_cubre_v2_y_la_muestra_del_entrenamiento(fx, tmp_path):
    sal = tmp_path / "salida"
    d = tmp_path / "t"
    b.construir(fx["manifiesto"], d, sal, huellas=fx["huellas"], solo=["mini-choice"])
    limpio = tmp_path / "limpio.txt"
    limpio.write_text("HECHA mini-choice soporte=24 test=207\n", encoding="utf-8")
    v2 = tmp_path / "v2"
    h.extraer_tgz(fx["dev"], v2)
    assert b.auditar_log(sal, limpio, registro=tmp_path / "ap.jsonl", textos_extra=[v2]) == 0
    sucio = tmp_path / "sucio.txt"
    sucio.write_text("fallo leyendo: " + fx["ref_exacto"] + "\n", encoding="utf-8")
    assert b.auditar_log(sal, sucio, registro=tmp_path / "ap2.jsonl", textos_extra=[v2]) >= 1
    assert b.main(["auditar-log", "--datos", str(sal), "--log", str(sucio), "--textos-extra", str(v2)]) == 1
