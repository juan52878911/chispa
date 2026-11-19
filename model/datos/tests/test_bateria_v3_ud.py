"""Empaquetador y ud de la fase 2 de la batería v3, en seco (sin red, sin AWS, sin subir nada): el HEAD del repo + fixtures inventados."""
import gzip
import hashlib
import json
import os
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
    r = sh("bash", "model/datos/aws/empaquetar_bateria_v3.sh", head, str(fx["huellas"]), str(d / "sal"), env={"CHISPA_UD_DIR": str(d / "ud")})
    return dict(dir=d, fx=fx, head=head, r=r, ud=d / "ud" / ("ud-bateria-v3-%s.sh" % head))


def test_empaquetador_escribe_ud_y_tgz_sin_subir_nada(empaquetado):
    r = empaquetado["r"]
    assert r.returncode == 0, r.stderr[-800:]
    ud = empaquetado["ud"].read_text(encoding="utf-8")
    assert "s3://chispa-pruebas-879381269751/bateria-v3-%s" % empaquetado["head"] in ud and "shutdown -h +$MIN" in ud and "MIN=${CHISPA_MIN:-75}" in ud
    assert "__COD_SHA256__" not in ud and "__N_TAREAS__" not in ud and len(ud.encode("utf-8")) < 15500
    assert "COD_SHA256=%s" % sha256(empaquetado["dir"] / "sal" / "codigo.tgz") in ud and "HUE_SHA256=%s" % sha256(empaquetado["dir"] / "sal" / "huellas.tgz") in ud
    assert "aws s3 cp" in ud and "s3 sync" not in ud and "s3 rm" not in ud                 # solo copia por nombre exacto: nada de listar ni borrar
    assert not (RAIZ / "model/datos/aws/lanzados").exists() or not list((RAIZ / "model/datos/aws/lanzados").glob("ud-bateria-v3-*"))


def test_empaquetador_se_niega_con_huellas_con_texto(empaquetado, tmp_path):
    malas = tmp_path / "huellas"
    malas.mkdir()
    with gzip.open(malas / "huellas-x.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"h": "a" * 64, "s": [1], "texto": "texto en claro"}) + "\n")
    r = sh("bash", "model/datos/aws/empaquetar_bateria_v3.sh", empaquetado["head"], str(malas), str(tmp_path / "sal"), env={"CHISPA_UD_DIR": str(tmp_path / "ud")})
    assert r.returncode != 0 and "huella" in r.stderr
    (malas / "huellas-x.jsonl.gz").unlink()
    (malas / "otro.txt").write_text("x", encoding="utf-8")
    assert sh("bash", "model/datos/aws/empaquetar_bateria_v3.sh", empaquetado["head"], str(malas), str(tmp_path / "sal"), env={"CHISPA_UD_DIR": str(tmp_path / "ud")}).returncode != 0


def test_ud_en_local_con_fixtures_inventados(empaquetado, tmp_path):
    assert empaquetado["r"].returncode == 0
    s3 = tmp_path / "s3"
    s3.mkdir()
    for n in ("codigo.tgz", "huellas.tgz"):
        shutil.copy(empaquetado["dir"] / "sal" / n, s3 / n)
    man = empaquetado["fx"]["manifiesto"]
    ud = empaquetado["ud"].read_text(encoding="utf-8")
    lineas = []
    for l in ud.splitlines():
        if l.startswith("COD_SHA256="):
            l = "COD_SHA256=%s; HUE_SHA256=%s; MAN_SHA256=%s; N_TAREAS=11" % (sha256(s3 / "codigo.tgz"), sha256(s3 / "huellas.tgz"), sha256(man))
        lineas.append(l)
    local = tmp_path / "ud-local.sh"
    local.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    r = sh("bash", str(local), env=dict(CHISPA_LOCAL="1", CHISPA_S3_LOCAL=str(s3), CHISPA_BASE=str(tmp_path / "base"), CHISPA_PY=sys.executable,
                                        CHISPA_V3_ESPEJO=str(empaquetado["fx"]["espejo"]), CHISPA_V3_MANIFIESTO=str(man)), cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-800:]
    assert "fallo=no" in r.stdout and "AUDITORIA" in r.stdout and "coincidencias=0" in r.stdout and "OK: sellada, sha256 y licencias coinciden" in r.stdout
    for n in ("manifiesto.json", "licencias.tgz", "datos-sellados.tgz", "resumen.json", "log.txt"):
        assert (s3 / n).exists(), n
    m = json.loads((s3 / "manifiesto.json").read_text(encoding="utf-8"))
    assert {t["estado"] for t in m["tareas"]} == {"construida", "fuera", "pendiente"}
    # el log subido no lleva texto de ninguna fila de test
    import tarfile
    with tarfile.open(s3 / "datos-sellados.tgz") as tf:
        tf.extractall(tmp_path / "x")
    log = (s3 / "log.txt").read_text(encoding="utf-8")
    for f in (tmp_path / "x" / "datos" / "sellada").glob("*.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines()[:50]:
            assert json.loads(l)["texto"][:40] not in log
