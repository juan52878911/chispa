"""Empaquetador y ud de la fase 2 de la batería v3, en seco (sin red, sin AWS, sin subir nada): el HEAD del repo + fixtures inventados."""
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from model.datos.tests.fixtures_v3.generar import generar

RAIZ = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(not shutil.which("git") or not shutil.which("bash") or not shutil.which("shasum"), reason="hace falta git, bash y shasum")


def sh(*cmd, env=None, cwd=RAIZ):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(list(cmd), cwd=str(cwd), env=e, capture_output=True, text=True, timeout=300)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def empaquetado(tmp_path_factory):
    d = tmp_path_factory.mktemp("ud")
    fx = generar(d / "fx")
    head = sh("git", "rev-parse", "--short", "HEAD").stdout.strip()
    if sh("git", "cat-file", "-e", "HEAD:model/datos/aws/bateria_v3.sh").returncode != 0:
        pytest.skip("el ud aún no está commiteado")
    r = sh("bash", "model/datos/aws/empaquetar_bateria_v3.sh", head, str(d / "sal"), env={"CHISPA_UD_DIR": str(d / "ud")})
    return dict(dir=d, fx=fx, head=head, r=r, ud=d / "ud" / ("ud-bateria-v3-%s.sh" % head))


def test_empaquetador_escribe_ud_y_tgz_sin_subir_nada(empaquetado):
    r = empaquetado["r"]
    assert r.returncode == 0, r.stderr[-800:]
    ud = empaquetado["ud"].read_text(encoding="utf-8")
    assert "s3://chispa-pruebas-879381269751/bateria-v3-%s" % empaquetado["head"] in ud and "shutdown -h +$MIN" in ud and "MIN=${CHISPA_MIN:-90}" in ud
    assert not re.search(r"__[A-Z0-9_]*__", ud.replace("__CHISPA_S3_LOCAL", "")), "quedan placeholders"
    assert len(ud.encode("utf-8")) < 15500 and "huellas.tgz" not in ud.replace("# ", "") .split("set -uo")[1] and "HUE_SHA256" not in ud
    assert "COD_SHA256=%s" % sha256(empaquetado["dir"] / "sal" / "codigo.tgz") in ud and not (empaquetado["dir"] / "sal" / "huellas.tgz").exists()
    assert "aws s3 cp" in ud and "s3 sync" not in ud and "s3 rm" not in ud                 # solo copia por nombre exacto: nada de listar ni borrar
    assert not (RAIZ / "model/datos/aws/lanzados").exists() or not list((RAIZ / "model/datos/aws/lanzados").glob("ud-bateria-v3-*"))


def test_ud_fija_nombres_y_sha_de_v2_dev_y_selladas(empaquetado):
    ud = empaquetado["ud"].read_text(encoding="utf-8")
    assert "s3://chispa-pruebas-879381269751/w1-fs8-a6c5293/dev.tgz" in ud and "DEV_SHA256=ce0423dc05a5b393067b6aeb2c1da4aa4d6192d249d1a52061c5f579e8dd3ca9" in ud
    assert "s3://chispa-pruebas-879381269751/r10-apertura-29c441f-a/selladas.tgz" in ud and "SEL_SHA256=a2bb334e5a1e0961a3d79f011d9525dfb872112d5e305a14303dbf16b5f487a0" in ud
    assert ud.index("sha_ok \"$DEV_SHA256\"") < ud.index("fases_main") and ud.index("sha_ok \"$SEL_SHA256\"") < ud.index("fases_main")      # sha comprobado antes de usar nada


def test_empaquetado_trae_el_manifiesto_de_entrenamiento_y_el_modulo_de_huellas(empaquetado):
    import tarfile
    with tarfile.open(empaquetado["dir"] / "sal" / "codigo.tgz") as tf:
        nombres = tf.getnames()
    assert "model/alt/manifiesto-entrenamiento.json" in nombres and "model/datos/bateria_v3_huellas.py" in nombres
    assert not any(n.endswith(("dev.tgz", "selladas.tgz")) or "huellas-" in n for n in nombres)


def test_empaquetador_pide_solo_sha_y_salida(empaquetado, tmp_path):
    assert sh("bash", "model/datos/aws/empaquetar_bateria_v3.sh", empaquetado["head"]).returncode != 0
    assert sh("bash", "model/datos/aws/empaquetar_bateria_v3.sh", "0" * 40, str(tmp_path / "sal"), env={"CHISPA_UD_DIR": str(tmp_path / "ud")}).returncode != 0


def _ud_local(empaquetado, tmp_path, dev=None, selladas=None):
    s3 = tmp_path / "s3"
    s3.mkdir()
    fx = empaquetado["fx"]
    shutil.copy(empaquetado["dir"] / "sal" / "codigo.tgz", s3 / "codigo.tgz")
    dev, selladas = dev or fx["dev"], selladas or fx["selladas"]
    lineas = []
    for l in empaquetado["ud"].read_text(encoding="utf-8").splitlines():
        if l.startswith("COD_SHA256="):
            l = "COD_SHA256=%s; MAN_SHA256=%s; N_TAREAS=13" % (sha256(s3 / "codigo.tgz"), sha256(fx["manifiesto"]))
        elif l.startswith("DEV_URI="):
            l = "DEV_URI=%s; DEV_SHA256=%s" % (dev, sha256(dev))
        elif l.startswith("SEL_URI="):
            l = "SEL_URI=%s; SEL_SHA256=%s" % (selladas, sha256(selladas))
        lineas.append(l)
    local = tmp_path / "ud-local.sh"
    local.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    r = sh("bash", str(local), env=dict(CHISPA_LOCAL="1", CHISPA_S3_LOCAL=str(s3), CHISPA_BASE=str(tmp_path / "base"), CHISPA_PY=sys.executable, CHISPA_V3_ESPEJO=str(fx["espejo"]),
                                        CHISPA_V3_MANIFIESTO=str(fx["manifiesto"]), CHISPA_V3_ENTRENAMIENTO=str(fx["entrenamiento"])), cwd=tmp_path)
    return s3, r


def test_ud_en_local_con_fixtures_inventados(empaquetado, tmp_path):
    assert empaquetado["r"].returncode == 0
    s3, r = _ud_local(empaquetado, tmp_path)
    assert r.returncode == 0, r.stderr[-800:]
    assert "fallo=no" in r.stdout and "AUDITORIA" in r.stdout and "coincidencias=0" in r.stdout and "OK: sellada, sha256 y licencias coinciden" in r.stdout
    assert "coincidencias_v2_y_entrenamiento=0" in r.stdout and "huellas-v2.jsonl.gz" in r.stdout
    for n in ("manifiesto.json", "licencias.tgz", "datos-sellados.tgz", "resumen.json", "log.txt"):
        assert (s3 / n).exists(), n
    assert not (s3 / "huellas.tgz").exists() and not (s3 / "dev.tgz").exists()
    m = json.loads((s3 / "manifiesto.json").read_text(encoding="utf-8"))
    assert {t["estado"] for t in m["tareas"]} == {"construida", "fuera", "pendiente"}
    # huellas dentro del ud: v2 (dev exacto y selladas casi-duplicado) y entrenamiento (exacto) descartan filas de mini-choice
    ch = next(t for t in m["tareas"] if t["nombre"] == "mini-choice")
    assert ch["solapes"]["por_fuente"] == {"v2": 2, "entrenamiento": 1} and ch["solapes"]["descartadas"] == 3
    assert m["solapes"]["huellas_leidas"]["huellas-entrenamiento.jsonl.gz"] == 2 + 5 + 5       # publica: 2 textos largos («corto» no cuenta), tabla: inputs y meta.nota
    eh = m["solapes"]["estado_huellas"]
    assert eh["v2"]["ficheros"] == {"dev": 6, "selladas": 8} and [c["nombre"] for c in eh["entrenamiento"]["comprobados"]] == ["publica", "tabla"]
    motivos = {x["nombre"]: x["motivo"] for x in m["solape_no_comprobado"]}
    assert set(motivos) == {"sindatos", "interna", "Fuente Propia A", "inventada-v1"} and "sin ficheros de datos" in motivos["sindatos"] and "no es una fuente pública" in motivos["interna"]
    r_ = m["resumen"]
    assert r_["por_tipo"]["yesno"]["informativo"] == 1 and r_["por_idioma"]["es"]["n_bajo"] is True
    # el log subido no lleva texto de ninguna fila de test ni de v2 ni del entrenamiento
    import tarfile
    with tarfile.open(s3 / "datos-sellados.tgz") as tf:
        tf.extractall(tmp_path / "x")
    log = (s3 / "log.txt").read_text(encoding="utf-8")
    for f in (tmp_path / "x" / "datos" / "sellada").glob("*.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines()[:50]:
            assert json.loads(l)["texto"][:40] not in log
    for t in (empaquetado["fx"]["ref_exacto"], empaquetado["fx"]["ref_entren"]):
        assert t[:40] not in log
    assert not (tmp_path / "base" / "v2txt").exists() and not (tmp_path / "base" / "muestra").exists()      # los textos de v2 no se quedan en la instancia


def test_ud_se_niega_si_el_sha_de_v2_no_coincide(empaquetado, tmp_path):
    malo = tmp_path / "dev-malo.tgz"
    malo.write_bytes(empaquetado["fx"]["dev"].read_bytes())
    s3 = tmp_path / "s3"
    s3.mkdir()
    shutil.copy(empaquetado["dir"] / "sal" / "codigo.tgz", s3 / "codigo.tgz")
    fx = empaquetado["fx"]
    lineas = []
    for l in empaquetado["ud"].read_text(encoding="utf-8").splitlines():
        if l.startswith("COD_SHA256="):
            l = "COD_SHA256=%s; MAN_SHA256=%s; N_TAREAS=13" % (sha256(s3 / "codigo.tgz"), sha256(fx["manifiesto"]))
        elif l.startswith("DEV_URI="):
            l = "DEV_URI=%s; DEV_SHA256=%s" % (malo, "0" * 64)
        elif l.startswith("SEL_URI="):
            l = "SEL_URI=%s; SEL_SHA256=%s" % (fx["selladas"], sha256(fx["selladas"]))
        lineas.append(l)
    local = tmp_path / "ud-malo.sh"
    local.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    r = sh("bash", str(local), env=dict(CHISPA_LOCAL="1", CHISPA_S3_LOCAL=str(s3), CHISPA_BASE=str(tmp_path / "base"), CHISPA_PY=sys.executable), cwd=tmp_path)
    assert "dev.tgz de v2: descarga o sha256" in r.stdout and not (tmp_path / "base" / "huellas" / "huellas-v2.jsonl.gz").exists() and not (s3 / "manifiesto.json").exists()


def test_ud_falla_si_v2_trae_otro_numero_de_ficheros(empaquetado, tmp_path):
    import tarfile
    pocos = tmp_path / "dev5.tgz"
    with tarfile.open(empaquetado["fx"]["dev"]) as src, tarfile.open(pocos, "w:gz") as dst:
        for m in src.getmembers()[:5]:
            dst.addfile(m, src.extractfile(m))
    s3, r = _ud_local(empaquetado, tmp_path, dev=pocos)
    assert "huellas de v2" in r.stdout and "se esperaban 6" in r.stdout and not (s3 / "manifiesto.json").exists()
