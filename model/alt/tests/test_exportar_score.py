"""exportar_score: el fixture de paridad Python → Go sale de las funciones de lineas_base_v2/v2b (sin red, datos sintéticos)."""
import json

from model.alt import exportar_score as ex
from model.alt import lineas_base_v2 as lb
from model.alt import lineas_base_v2b as vb

NIV = ["1", "2", "3"]


def fila(i, sp, y):
    return {"id": "t:%s:%d" % (sp, i), "tarea": "t", "tipo": "score", "pregunta": "?", "texto": "x", "opciones": NIV, "etiqueta": y, "split": sp}


def pred(r, nivel, sesgo=0.0):
    p = {o: 0.05 for o in NIV}
    p[nivel] = 0.9 + sesgo
    return {"id": r["id"], "probs": p, "valida": True, "ms": 1.0}


def test_fixture_coincide_con_las_funciones_de_python(tmp_path):
    sop = [fila(i, "train", NIV[i % 3]) for i in range(24)]
    ev = [fila(i, "test", NIV[i % 3]) for i in range(30)]
    ps = {r["id"]: pred(r, r["etiqueta"]) for r in sop}
    pt = {r["id"]: pred(r, r["etiqueta"] if i % 4 else "1") for i, r in enumerate(ev)}
    del pt[ev[0]["id"]]                      # fila sin predicción: fallo, sin esperanza
    pt[ev[1]["id"]]["valida"] = False        # inválida: igual
    excl = {sop[0]["id"]}
    out = ex.exportar_score("t", sop, ev, pt, ps, excl)
    assert out["umbrales"] == vb.umbrales(sop, ps, excl)
    assert out["excluidos"] == [sop[0]["id"]]
    assert len(out["umbrales"]) == 2 and out["umbrales"] == sorted(out["umbrales"])
    assert [f["nivel"] for f in out["filas"][:2]] == [None, None] and out["filas"][0]["esperanza"] is None
    for f, r in zip(out["filas"][2:], ev[2:]):
        e = lb.esperanza(NIV, [pt[r["id"]]["probs"][o] for o in NIV])
        assert f["esperanza"] == e and f["nivel"] == lb.top1(pt[r["id"]]["probs"], r, esp=out["umbrales"])
    ok = sum(1 for f in out["filas"] if f["nivel"] == f["etiqueta"])
    assert out["exactitud"] == round(ok / len(ev), 4) and out["n_test"] == 30
    json.dumps(out)                          # serializable


def test_topk_lista_las_candidatas(tmp_path):
    ev = [dict(fila(i, "test", "1"), tipo="choice") for i in range(3)]
    p = {ev[0]["id"]: {"probs": {"b": 0.5, "a": 0.5}}, ev[1]["id"]: {"probs": {}}}
    out = ex.exportar_topk("t", ev, p)
    assert out["filas"][0]["candidatas"] == ["a", "b"] and out["filas"][1]["candidatas"] is None and out["filas"][2]["candidatas"] is None and out["k"] == 2
