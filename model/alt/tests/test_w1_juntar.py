import json

import pytest

from model.alt import w1_fixture_b2, w1_juntar, w1_medir


def leer(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


@pytest.fixture()
def b2(tmp_path):
    info = w1_fixture_b2.crear(tmp_path / "fx", 25)
    w1_juntar.main(["extraer", "--res-tgz", str(tmp_path / "fx/resultados.tgz"), "--fasea-tar", str(tmp_path / "fx/faseA.tar"), "--fasea-json", str(tmp_path / "fx/faseA.json"),
                    "--salida", str(tmp_path / "x")])
    return tmp_path, info


def test_extraer_solo_res_y_ent(b2):
    d, _ = b2
    assert (d / "x/res/py").is_dir() and (d / "x/ent/consultas.jsonl").is_file()
    assert not (d / "x/faseA/onnx").exists() and not list((d / "x").rglob("relleno.bin"))


def test_extraer_rechaza_un_faseA_json_de_otro_tar(b2, tmp_path):
    d, _ = b2
    j = json.load(open(d / "fx/faseA.json"))
    j["tar_sha256"] = "0" * 64
    (d / "fx/otro.json").write_text(json.dumps(j))
    with pytest.raises(SystemExit):
        w1_juntar.main(["extraer", "--res-tgz", str(d / "fx/resultados.tgz"), "--fasea-tar", str(d / "fx/faseA.tar"), "--fasea-json", str(d / "fx/otro.json"), "--salida", str(tmp_path / "y")])


def test_faltan_solo_no_validas_o_ausentes_y_finanzas_primero(b2):
    d, info = b2
    w1_juntar.main(["faltan", "--res", str(d / "x/res"), "--ent", str(d / "x/ent"), "--salida", str(d / "f")])
    plan = json.load(open(d / "f/plan.json"))
    assert plan["pendientes"] == info["faltan"] and plan["total"] == 80
    assert plan["orden"] == ["finanzas-documentos", "horeca-resenas", "jql-valor-educativo", "usajobs-nivel"]
    # la fila de finanzas con un aviso en `error` pero válida NO se vuelve a medir
    viejas = leer(d / "x/res/py/finanzas-documentos.jsonl")
    assert viejas[0]["error"] and viejas[0]["valida"] and viejas[0]["id"] not in {r["id"] for r in leer(d / "f/test-demos/finanzas-documentos.jsonl")}
    # mismas ids y mismas demostraciones que el servidor
    demos = {r["id"]: r["ejemplos"] for r in leer(d / "x/res/demos/horeca-resenas.jsonl")}
    assert {r["id"]: r["ejemplos"] for r in leer(d / "f/demos/horeca-resenas.jsonl")} == demos


def test_faltan_se_niega_si_las_demos_no_son_las_del_registro(b2):
    d, _ = b2
    p = d / "x/res/demos/horeca-resenas.jsonl"
    filas = leer(p)
    filas[0]["ejemplos"] = filas[0]["ejemplos"][::-1]
    p.write_text("".join(json.dumps(r) + "\n" for r in filas))
    with pytest.raises(SystemExit, match="registro"):
        w1_juntar.main(["faltan", "--res", str(d / "x/res"), "--ent", str(d / "x/ent"), "--salida", str(d / "f")])


def test_faltan_se_niega_si_go_no_esta_completo(b2):
    d, _ = b2
    p = d / "x/res/e2e-w1-filas.jsonl"
    p.write_text("".join(l for l in open(p) if "horeca" not in l))
    with pytest.raises(SystemExit, match="Go"):
        w1_juntar.main(["faltan", "--res", str(d / "x/res"), "--ent", str(d / "x/ent"), "--salida", str(d / "f")])


def _nuevas(d, tarea, n=None, valida=True):
    filas = leer(d / "f/test-demos" / (tarea + ".jsonl"))[:n]
    out = d / "nuevo"
    out.mkdir(exist_ok=True)
    cons = {r["id"]: r for r in leer(d / "x/ent/consultas.jsonl")}
    with open(out / (tarea + ".jsonl"), "w") as f:
        for r in filas:
            o = cons[r["id"]]["opciones"]
            f.write(json.dumps({"id": r["id"], "probs": {x: 1 / len(o) for x in o} if valida else None, "valida": valida, "error": "" if valida else "x", "ms": 5}) + "\n")


def test_juntar_una_fila_por_id_y_conserva_las_validas(b2):
    d, _ = b2
    w1_juntar.main(["faltan", "--res", str(d / "x/res"), "--ent", str(d / "x/ent"), "--salida", str(d / "f")])
    for t in ("finanzas-documentos", "horeca-resenas", "jql-valor-educativo", "usajobs-nivel"):
        _nuevas(d, t)
    w1_juntar.main(["juntar", "--res", str(d / "x/res"), "--nuevo", str(d / "nuevo"), "--salida", str(d / "py")])
    for t in json.load(open(d / "f/plan.json"))["pendientes"]:
        fin = leer(d / "py" / (t + ".jsonl"))
        assert [r["id"] for r in fin] == [r["id"] for r in leer(d / "x/res/test-demos" / (t + ".jsonl"))] and all(r["valida"] for r in fin)
    # las válidas de B quedan idénticas
    assert [r for r in leer(d / "x/res/py/cfpb-quejas.jsonl")] == leer(d / "py/cfpb-quejas.jsonl")
    viejas = {r["id"]: r for r in leer(d / "x/res/py/finanzas-documentos.jsonl")}
    assert all(r == viejas[r["id"]] for r in leer(d / "py/finanzas-documentos.jsonl") if r["id"] in viejas)


def test_juntar_con_corte_deja_parcial_y_medir_ve_las_dos_lecturas_completas(b2):
    d, _ = b2
    w1_juntar.main(["faltan", "--res", str(d / "x/res"), "--ent", str(d / "x/ent"), "--salida", str(d / "f")])
    _nuevas(d, "finanzas-documentos")
    _nuevas(d, "horeca-resenas", n=10)
    w1_juntar.main(["juntar", "--res", str(d / "x/res"), "--nuevo", str(d / "nuevo"), "--salida", str(d / "py")])
    junta = json.load(open(d / "py/junta.json"))
    assert junta["horeca-resenas"]["validas"] == 10 and junta["horeca-resenas"]["sin_remedir"] == 15 and junta["usajobs-nivel"]["validas"] == 0
    assert len(leer(d / "py/usajobs-nivel.jsonl")) == 25   # las no válidas de B siguen ahí: cobertura < 100 %
    crit = {"id": "t", "criterio": {"tolerancia_macro": 0.005, "cobertura_minima": 1.0, "acuerdo_seleccion_minimo": 0.98}}
    out = w1_medir.cifras(leer(d / "x/ent/consultas.jsonl"), leer(d / "x/res/e2e-w1-filas.jsonl"), d / "py", crit, json.load(open(d / "x/res/seleccion.json")))
    assert out["n"] == 150 and out["cobertura_minima_por_tarea"]["go"] == 1.0 and out["cobertura_minima_por_tarea"]["python"] == 0.0
    assert not out["pasa_todo"]


def test_juntar_rechaza_remedir_una_fila_valida(b2):
    d, _ = b2
    w1_juntar.main(["faltan", "--res", str(d / "x/res"), "--ent", str(d / "x/ent"), "--salida", str(d / "f")])
    ya = leer(d / "x/res/py/cfpb-quejas.jsonl")[0]
    (d / "nuevo").mkdir()
    (d / "nuevo/cfpb-quejas.jsonl").write_text(json.dumps({"id": ya["id"], "probs": {"a": 1.0}, "valida": True}) + "\n")
    with pytest.raises(SystemExit, match="ya eran válidas"):
        w1_juntar.main(["juntar", "--res", str(d / "x/res"), "--nuevo", str(d / "nuevo"), "--salida", str(d / "py")])
