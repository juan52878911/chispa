"""Lectores arreglados en W3c (esquemas reales vistos en la fase 2), fuente por catálogo CKAN y huellas por ficheros. Datos INVENTADOS, sin red."""
import json
from pathlib import Path

import pytest

from model.datos import bateria_v3 as b
from model.datos import bateria_v3_huellas as h
from model.datos import bateria_v3_licencias as lic
from model.datos.bateria_v3_candidatas import CANDIDATAS

POR_NOMBRE = {c["nombre"]: c for c in CANDIDATAS}


def tarea(nombre, **lector):
    """La tarea REAL del repo (su lector tal cual) con datos inventados en <tmp>/datos."""
    t = json.loads(json.dumps(POR_NOMBRE[nombre]))
    t["lector"].update(lector)
    return t


def poner(carpeta, ficheros):
    d = Path(carpeta) / "datos"
    for n, contenido in ficheros.items():
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_text(contenido, encoding="utf-8")


def etiquetas(items):
    return sorted({i["etiqueta"] for i in items})


def test_unam_csv_con_barra_vertical_y_acentos(tmp_path):
    filas = "".join("Resumen número %d de una tesis sobre algo concreto|Título %d|%s\n" % (n, n, "Derecho" if n % 2 else "Medicina") for n in range(60))
    poner(tmp_path, {"tesis.csv": "texto|titulo|carrera\n" + filas})
    items, cif = b.extraer_items(tarea("unam-tesis-carrera"), tmp_path)
    assert len(items) == 60 and etiquetas(items) == ["Derecho", "Medicina"] and items[0]["texto"].startswith("Título")
    # la misma cabecera SIN separador en el lector sale por la regla escrita (una sola columna con «|» dentro)
    t = tarea("unam-tesis-carrera")
    t["lector"].pop("separador")
    assert len(b.extraer_items(t, tmp_path)[0]) == 60


def test_biotech_mayusculas_y_espacios_en_las_columnas(tmp_path):
    cab = "Title,Content,Label 1,Label 2,Label 3,Label 4,Label 5,Target Organization\n"
    poner(tmp_path, {"train.csv": cab + "".join("T%d,contenido de la noticia %d,%s,,,,,Acme\n" % (n, n, "financiación" if n % 2 else "ensayo") for n in range(50))})
    items, _ = b.extraer_items(tarea("biotech-evento"), tmp_path)
    assert len(items) == 50 and etiquetas(items) == ["ensayo", "financiación"]


def test_moral_stories_salta_las_configuraciones_sin_las_columnas(tmp_path):
    cls = "".join(json.dumps({"ID": str(n), "label": "0", "moral_action": "a%d" % n}) + "\n" for n in range(5))
    full = "".join(json.dumps({"ID": str(n), "situation": "s%d" % n, "intention": "i%d" % n, "moral_action": "bien %d" % n, "immoral_action": "mal %d" % n}) + "\n" for n in range(7))
    poner(tmp_path, {"cls-action/train.jsonl": cls, "full/train.jsonl": full})
    items, cif = b.extraer_items(tarea("moral-stories-accion"), tmp_path)
    assert len(items) == 14 and etiquetas(items) == ["no", "si"] and cif["ficheros_omitidos_por_esquema"] == 1
    poner(tmp_path / "solo", {"cls-action/train.jsonl": cls})
    with pytest.raises(b.EsquemaError) as e:                               # si no queda ninguno: nombres y TIPOS de columnas, nunca texto
        b.extraer_items(tarea("moral-stories-accion"), tmp_path / "solo")
    assert "label" in str(e.value) and "str" in str(e.value) and "a1" not in str(e.value)


def test_fake_news_es_con_el_esquema_real(tmp_path):
    cab = "ID,CATEGORY,TOPICS,SOURCE,HEADLINE,TEXT,LINK\n"
    poner(tmp_path, {"train.csv": cab + "".join("%d,%s,Ciencia,Fuente,Titular %d,Cuerpo de la noticia %d,http://x\n" % (n, "Fake" if n % 3 else "True", n, n) for n in range(30))})
    items, _ = b.extraer_items(tarea("fake-news-es"), tmp_path)
    assert len(items) == 30 and etiquetas(items) == ["no", "si"]
    poner(tmp_path / "otro", {"d.csv": cab + "1,Satire,x,s,h,t,l\n"})        # etiqueta fuera del mapa: el motivo lista los valores (etiquetas, no texto)
    with pytest.raises(b.EsquemaError) as e:
        b.extraer_items(tarea("fake-news-es"), tmp_path / "otro")
    assert "Satire" in str(e.value)


def test_wnli_tsv_guardado_como_csv_y_sin_el_test_sin_etiquetas(tmp_path):
    cab = "index\tsentence 1\tsentence 2\tlabel\n"
    tr = "".join("%d\tPrimera frase %d con «comillas\tSegunda %d\t%d\n" % (n, n, n, n % 2) for n in range(20))
    poner(tmp_path, {"wnli-train-es.csv": cab + tr, "wnli-dev-es.csv": cab + tr, "wnli-test-shuffled-es.csv": cab + "".join("%d\ta\tb\t-1\n" % n for n in range(9))})
    items, _ = b.extraer_items(tarea("wnli-es"), tmp_path)
    assert len(items) == 40 and etiquetas(items) == ["no", "si"]


@pytest.mark.parametrize("variante", ["objetos_varias_lineas", "lista_indentada", "objeto_con_data", "jsonl"])
def test_json_de_varias_lineas_ya_no_da_jsondecodeerror(tmp_path, variante):
    regs = [{"question": "¿Pregunta %d?" % n, "answer": bool(n % 2)} for n in range(10)]
    texto = {"objetos_varias_lineas": "".join(json.dumps(r, indent=2) + "\n" for r in regs), "lista_indentada": json.dumps(regs, indent=2),
             "objeto_con_data": json.dumps({"version": "1", "data": regs}, indent=1), "jsonl": "".join(json.dumps(r) + "\n" for r in regs)}[variante]
    poner(tmp_path, {"train.json": texto})
    items, _ = b.extraer_items(tarea("strategyqa-si-no"), tmp_path)
    assert len(items) == 10 and etiquetas(items) == ["no", "si"]


def test_json_de_un_solo_registro_largo_no_se_expande_si_hay_varias_lineas(tmp_path):
    regs = [{"id": n, "context": [{"role": "user", "content": "hola %d" % n}], "ans": "x"} for n in range(3)]
    poner(tmp_path, {"d.jsonl": "".join(json.dumps(r) + "\n" for r in regs)})
    filas = list(b._filas_fichero(tmp_path / "datos" / "d.jsonl", "jsonl", {}))
    assert len(filas) == 3 and filas[0]["context"][0]["content"] == "hola 0"
    assert b._a_texto(filas[0]["context"]) == "hola 0"                 # mensajes de chat -> solo su contenido


def test_resolver_columnas_alternativas_y_normalizacion():
    fila = {"Label 1": "a", "Título": "t", "ctx": {"x": 1}}
    assert b._clave_real("label_1", fila) == "Label 1" and b._clave_real("titulo", fila) == "Título" and b._clave_real("nada||ctx.x", fila) == "ctx.x" and b._clave_real("nada", fila) is None
    l, falta, _ = b._resolver_lector(dict(texto=["titulo", "no_existe"], etiqueta="label1"), [fila])
    assert l["texto"] == ["Título", "no_existe"] and l["etiqueta"] == "Label 1" and falta == ["no_existe"]


def test_columnas_automaticas_con_regla_escrita(tmp_path):
    filas = "".join("%d,Texto bastante largo número %d para que cuente como texto de un ejemplo,%s,es\n" % (n, n, ["pos", "neg", "neu"][n % 3]) for n in range(90))
    poner(tmp_path, {"d.csv": "id,body,sentiment,lang\n" + filas})
    t = tarea("noticias-es-categoria")
    t["lector"] = dict(formato="auto", ficheros=["*.csv"], auto={}, min_por_clase=10)
    items, cif = b.extraer_items(t, tmp_path)
    assert etiquetas(items) == ["neg", "neu", "pos"] and len(items) == 90
    assert t["esquema_elegido"]["d.csv"] == dict(texto=["body"], etiqueta="sentiment", valores_etiqueta=3)       # queda en el manifiesto qué columnas eligió
    poner(tmp_path / "x", {"d.csv": "id,body\n" + "".join("%d,%s\n" % (n, "t" * 50) for n in range(20))})
    with pytest.raises(b.EsquemaError) as e:                                  # sin columna de etiqueta: nombres y tipos
        b.extraer_items(dict(t, lector=t["lector"]), tmp_path / "x")
    assert "ninguna columna de etiqueta" in str(e.value) and "body" in str(e.value)


# ---------------------------------------------------------------------------------------------------------------- catálogo CKAN
def pkg(**kw):
    p = dict(name="p", id="1", title="Storm Events Database", license_id="us-pd", license_title="U.S. Government Work", license_url="", metadata_modified="2026-01-01",
             organization=dict(title="NOAA"), extras=[dict(key="accessLevel", value="public")], resources=[dict(url="https://www.ncei.noaa.gov/pub/data/x/", format="HTML", name="d")])
    p.update(kw)
    return p


@pytest.mark.parametrize("lid,ok,lic_", [("us-pd", True, "dominio-publico"), ("other-pd", True, "dominio-publico"), ("cc-zero", True, "cc0-1.0"), ("cc-by", True, "cc-by-4.0"),
                                         ("cc-by-sa", False, ""), ("cc-nc", False, ""), ("other-open", False, ""), ("license-not-specified", False, ""), ("", False, "")])
def test_licencia_del_catalogo(lid, ok, lic_):
    f = b.ckan_ficha(pkg(license_id=lid), pkg()["resources"][0], {}, "https://x/y")
    r = lic.evaluar_ckan(f)
    assert (r["estado"] == "admitida") == ok and r["licencia"] == lic_ and (ok or r["motivo"]) and r["sha256_texto"]


def test_catalogo_rechaza_acceso_no_publico_sa_y_twitter():
    base = pkg()["resources"][0]
    assert lic.evaluar_ckan(b.ckan_ficha(pkg(extras=[dict(key="accessLevel", value="restricted public")]), base, {}, "u"))["estado"] == "fuera"
    assert "SA" in lic.evaluar_ckan(b.ckan_ficha(pkg(license_title="Creative Commons Attribution Share-Alike"), base, {}, "u"))["motivo"]
    assert "Twitter" in lic.evaluar_ckan(b.ckan_ficha(pkg(title="Tweets de la agencia"), base, {}, "u"))["motivo"]


def espejo_ckan(tmp_path, monkeypatch, paquete, csv_texto=None):
    esp = tmp_path / "esp"
    (esp / "ckan").mkdir(parents=True, exist_ok=True)
    datos = tmp_path / "origen.csv"
    datos.write_text(csv_texto or "narrative,body_part\n" + "".join("relato número %d largo largo,%s\n" % (n, "30" if n % 2 else "75") for n in range(80)), encoding="utf-8")
    if paquete is not None:
        paquete["resources"] = [dict(url=datos.as_uri(), format="CSV", name="d")]
    (esp / "ckan" / "_busqueda.json").write_text(json.dumps(dict(success=True, result=dict(results=[paquete] if paquete else []))), encoding="utf-8")
    monkeypatch.setenv("CHISPA_V3_ESPEJO", str(esp))
    return datos


def tarea_ckan(**fuente):
    t = json.loads(json.dumps(POR_NOMBRE["neiss-parte-cuerpo"]))
    t["fuente"] = dict(tipo="ckan", busqueda="neiss", titulo_regex="neiss", recurso_regex=r"origen\.csv", max_gb=1, **fuente)
    t["lector"] = dict(formato="csv", ficheros=["*.csv"], texto="narrative", etiqueta="body_part", mapa_etiqueta={"30": "hombro", "75": "cabeza"})
    return t


def test_ficha_y_descarga_salen_del_registro_del_catalogo(tmp_path, monkeypatch):
    espejo_ckan(tmp_path, monkeypatch, pkg(title="NEISS injury data"))
    t = tarea_ckan()
    texto = b.bajar_ficha(t, tmp_path / "t")
    f = json.loads(texto)
    assert f["license_id"] == "us-pd" and f["recurso"]["url"] == t["recurso_url"] and f["descarga"] == t["recurso_url"]
    r = b.evaluar_licencia_tarea(t, texto, tmp_path)
    assert r["estado"] == "admitida" and r["licencia"] == "dominio-publico" and t["ckan_license_id"] == "us-pd"
    assert b.bajar_datos(t, tmp_path / "t") and t["rev"].startswith("sha256:")
    items, _ = b.extraer_items(t, tmp_path / "t")
    assert len(items) == 80 and etiquetas(items) == ["cabeza", "hombro"]


def test_catalogo_sin_recurso_o_sin_licencia_admitida_deja_fuera_con_motivo(tmp_path, monkeypatch):
    espejo_ckan(tmp_path, monkeypatch, pkg(title="NEISS injury data", license_id="other-open"))
    t = tarea_ckan()
    r = b.evaluar_licencia_tarea(t, b.bajar_ficha(t, tmp_path / "t"), tmp_path)
    assert r["estado"] == "fuera" and "other-open" in r["motivo"]
    espejo_ckan(tmp_path, monkeypatch, None)
    t = tarea_ckan()
    assert b.bajar_ficha(t, tmp_path / "t2") == "" and "catálogo" in b.evaluar_licencia_tarea(t, "", tmp_path)["motivo"]
    espejo_ckan(tmp_path, monkeypatch, pkg(title="Otra cosa"))                # el título no casa con titulo_regex
    t = tarea_ckan()
    assert b.bajar_ficha(t, tmp_path / "t3") == "" and "ningún paquete" in t["_error_ficha"]


def test_url_de_descarga_propia_solo_si_es_del_dominio_de_un_recurso(tmp_path, monkeypatch):
    espejo_ckan(tmp_path, monkeypatch, pkg(title="NEISS data"))
    t = tarea_ckan(descarga="https://otro.example.org/neiss.csv")
    assert b.bajar_ficha(t, tmp_path / "t") == "" and "dominio" in t["_error_ficha"]
    pk = pkg(title="NEISS data")
    pk["resources"] = []
    espejo_ckan(tmp_path, monkeypatch, pk)
    t = tarea_ckan(descarga="https://www.ncei.noaa.gov/x.csv")
    pk2 = b.ckan_ficha(pkg(), dict(url="https://www.ncei.noaa.gov/pub/", format="HTML"), {}, "u")
    assert pk2["recurso"]["url"].startswith("https://www.ncei.noaa.gov") and b.ckan_descarga(pkg(), dict(url="https://www.ncei.noaa.gov/pub/"), dict(descarga="https://www.ncei.noaa.gov/x.csv")) == "https://www.ncei.noaa.gov/x.csv"


def test_los_ocho_federales_y_el_resto_de_las_tareas_del_catalogo_estan_bien_formados():
    ck = [c for c in CANDIDATAS if c["fuente"]["tipo"] == "ckan"]
    assert {"tormentas-tipo", "nhtsa-componente", "neiss-parte-cuerpo", "nsf-direccion", "cpsc-recall-peligro", "vaers-hospitalizado", "osha-hospitalizado", "ctgov-fase"} <= {c["nombre"] for c in ck}
    for c in ck:
        assert "licencia_regex_ficha" not in c and c["fuente"]["recurso_regex"] and c["fuente"].get("titulo_regex"), c["nombre"]


# ---------------------------------------------------------------------------------------------------------------- huellas de una fuente enorme, fichero a fichero
def test_huellas_por_ficheros_borra_cada_fichero_y_marca_parcial_si_no_hay_cupo(tmp_path, monkeypatch):
    repo = "tasksource/tasksource-jev-typed-decisions"
    esp = tmp_path / "hf" / repo.replace("/", "__")
    for k in range(3):
        (esp / "data").mkdir(parents=True, exist_ok=True)
        (esp / "data" / ("t%d.jsonl" % k)).write_text("".join(json.dumps({"texto": "texto del fichero %d número %d de más de veinte caracteres" % (k, n)}) + "\n" for n in range(5)), encoding="utf-8")
    monkeypatch.setenv("CHISPA_V3_ESPEJO", str(tmp_path))
    ent = tmp_path / "e.json"
    ent.write_text(json.dumps(dict(conjuntos=[dict(nombre="tsjev", fuente=repo)], exclusiones={})), encoding="utf-8")
    visto = []
    orig = b.hf_bajar
    monkeypatch.setattr(b, "hf_bajar", lambda r, rev, pat, dest: (orig(r, rev, pat, dest), visto.append(sorted(p.name for p in Path(dest).rglob("*.jsonl")))))
    n, comp, no = h.huellas_entrenamiento(str(ent), tmp_path / "t", tmp_path / "s", tmp_path / "m")
    assert n == 15 and comp[0]["ficheros"] == 3 and not no and all(len(v) == 1 for v in visto)      # nunca hay más de un fichero en disco
    monkeypatch.setattr(h, "MAX_HUELLAS_POR_FICHEROS", 5)
    n, comp, no = h.huellas_entrenamiento(str(ent), tmp_path / "t2", tmp_path / "s2", tmp_path / "m2")
    assert n == 5 and comp[0]["sin_tiempo"] == 2 and no[0]["parcial"] is True


# ---------------------------------------------------------------------------------------------------------------- contra lo que dejó la fase 2 (fixture real)
FASE2 = Path(__file__).parent / "fixtures_v3" / "fase2-100de63"


@pytest.mark.parametrize("nombre", ["unam-tesis-carrera", "biotech-evento", "moral-stories-accion", "fake-news-es", "wnli-es"])
def test_fichas_reales_de_las_arreglables_siguen_admitidas(nombre):
    r = lic.evaluar_tarjeta((FASE2 / "licencias" / (nombre + ".txt")).read_text(encoding="utf-8"), esperada=POR_NOMBRE[nombre]["licencia_esperada"])
    assert r["estado"] == "admitida", r["motivo"]


@pytest.mark.parametrize("nombre", ["wikinoticias-es-categoria", "argq-calidad", "bbh-deportes", "sts-es"])
def test_fichas_reales_de_las_que_se_quedan_fuera_siguen_fuera(nombre):
    texto = (FASE2 / "licencias" / (nombre + ".txt")).read_text(encoding="utf-8")
    assert lic.evaluar_tarjeta(texto)["estado"] == "fuera" if nombre != "wikinoticias-es-categoria" else lic.evaluar_ficha(texto, r"CC", "cc-by-2.5")["estado"] == "fuera"


@pytest.mark.parametrize("nombre", ["unam-tesis-carrera", "biotech-evento", "moral-stories-accion", "fake-news-es"])
def test_los_lectores_arreglados_aceptan_las_columnas_que_registro_el_log(nombre, tmp_path):
    """Las columnas de la fila que vio la fase 2 (motivo_fuera del manifiesto real) ya no dan «faltan columnas» con el lector nuevo."""
    import re
    man = json.loads((FASE2 / "manifiesto.json").read_text(encoding="utf-8"))
    motivo = next(t["motivo_fuera"] for t in man["tareas"] if t["nombre"] == nombre)
    vistas = re.search(r"la fila trae (\[.*\])", motivo).group(1)
    cols = eval(vistas, {})                          # lista de nombres de columna inventariada por el log (sin datos)
    t = tarea(nombre)
    if nombre == "unam-tesis-carrera":
        assert cols == ["texto|titulo|carrera"]
        cols = ["texto", "titulo", "carrera"]
    l, falta, _ = b._resolver_lector(t["lector"], [{c: "x" for c in cols}])
    if nombre == "moral-stories-accion":
        assert falta                                  # esta fila (ID, label, moral_action) es OTRA configuración: se salta, no se fuerza
    else:
        assert not falta, (nombre, falta)
