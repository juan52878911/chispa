"""Pruebas de model.alt.tareas_nuevas sin red: un hub falso con las tres fuentes en miniatura."""
import csv
import io
import json
import tarfile

from model.alt import tareas_nuevas as tn


def _hub(tmp_path):
    hub = tmp_path / "hub"
    def snap(repo):
        d = hub / ("datasets--" + repo.replace("/", "--")) / "snapshots" / "abc"
        d.mkdir(parents=True)
        return d
    d = snap("BSC-LT/LexBOE")
    for s, n in (("train", 30), ("dev", 2), ("test", 30)):
        (d / (s + ".jsonl")).write_text("".join(json.dumps({"id": "B%s%d" % (s, i), "sentence": "art %s %d " % (s, i) + "x" * (4000 if i == 0 else 5),
                                                            "label": ["Salud", "Cultura"][i % 2]}) + "\n" for i in range(n)))
    d = snap("mariagrandury/fake_news_corpus_spanish")
    with open(d / "test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "CATEGORY", "TOPICS", "SOURCE", "HEADLINE", "TEXT", "LINK"])
        w.writeheader()
        for i in range(10):
            w.writerow({"ID": str(i), "CATEGORY": "FALSE" if i % 2 else "TRUE", "TOPICS": "Sociedad", "SOURCE": "s", "HEADLINE": "h%d" % i, "TEXT": "t", "LINK": ""})
    d = snap("coastalcph/multi_eurlex")
    (d / "data").mkdir()
    with tarfile.open(d / "data" / "multi_eurlex.tar.gz", "w:gz") as tar:
        for s in ("train", "dev", "test"):
            filas = []
            for i in range(60):
                n1 = ["100149"] if i % 3 == 0 else (["100160"] if i % 3 == 1 else ["100149", "100160"])
                filas.append({"celex_id": "%s%d" % (s, i), "text": {"es": "ley %s %d" % (s, i), "en": "law"}, "eurovoc_concepts": {"level_1": n1}})
            filas.append({"celex_id": "rara", "text": {"es": "rara", "en": "x"}, "eurovoc_concepts": {"level_1": ["100999"]}})
            b = "".join(json.dumps(f) + "\n" for f in filas).encode()
            ti = tarfile.TarInfo(s + ".jsonl")
            ti.size = len(b)
            tar.addfile(ti, io.BytesIO(b))
    ev = tmp_path / "eurovoc.json"
    ev.write_text(json.dumps({"100149": {"es": "asuntos sociales"}, "100160": {"es": "industria"}, "100999": {"es": "rara"}}))
    return hub, ev


def test_construir_respeta_reglas(tmp_path):
    hub, ev = _hub(tmp_path)
    por, meta = tn.construir(hub, ev, tope=2000, k=8)
    assert meta["lexboe"]["clases"] == 2 and meta["lexboe"]["n_soporte"] == 16 and meta["lexboe"]["cortados_eval"] == 1
    assert all(len(r["texto"]) <= tn.TOPE_CAR for r in por["lexboe"]["test"])
    # eurlex: solo un descriptor; «rara» (1 en test) fuera por tener < 20
    assert set(meta["eurlex-n1"]["clases_descartadas_menos_de_20_en_test"]) == {"rara"}
    assert meta["eurlex-n1"]["docs_un_descriptor"]["test"] == 41
    assert "eurlex-n1" not in por and meta["eurlex-n1"]["viable"] is False      # 2 clases válidas < 3: no hay tarea
    # fakenews: zero-shot, sin soporte; CATEGORY FALSE = noticia falsa = «true»
    assert meta["fakenews-es"]["n_soporte"] == 0 and meta["fakenews-es"]["n_eval"] == 10
    assert sum(r["etiqueta"] == "true" for r in por["fakenews-es"]["test"]) == 5
    assert all("BOE" in meta[c]["atribucion"] or "©" in meta[c]["atribucion"] or "Posadas" in meta[c]["atribucion"] for c in meta)
