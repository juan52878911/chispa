import glob
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from model.etiquetado import banco_pqrs as B

AQUI = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(AQUI, "..", "..", "datos", "fixtures")


class BancoPQRS(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.raiz = os.path.join(self.tmp, "datosgov")
        os.makedirs(self.raiz)
        for f in glob.glob(os.path.join(FIX, "pqrs_*", "*")):   # fixtures inventados, con las columnas de las fuentes reales
            shutil.copy(f, self.raiz)
        self.dir = os.path.join(self.tmp, "banco")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_muestra_reparte_por_entidad_y_sin_plantillas_repetidas(self):
        pares = B.cargar_fuentes(self.raiz)
        filas = B.muestra(pares, 20, semilla=7, min_chars=5)
        self.assertEqual(len(filas), 20)
        por = {}
        for r in filas:
            por[r["entidad"]] = por.get(r["entidad"], 0) + 1
        self.assertEqual(sorted(por.values()), [5, 5, 5, 5])
        textos = [r["texto"] for r in filas]
        self.assertEqual(len(textos), len(set(textos)))
        self.assertEqual(B.muestra(pares, 20, semilla=7, min_chars=5), filas)

    def test_tapar_numeros(self):
        self.assertEqual(B.tapar_numeros("cel 300 123 4567, radicado 2024-001234, año 2023"),
                         "cel [NUM], radicado [NUM], año 2023")

    def test_servidor_a_ciegas_y_kappa(self):
        self.assertEqual(B.main(["muestra", "--dir", self.dir, "--raiz", self.raiz, "--n", "8", "--min-chars", "5"]), 0)
        with open(os.path.join(self.dir, "etiquetadores.json"), "w") as f:
            json.dump({"ana": "ta", "luis": "tl"}, f)
        with open(os.path.join(self.dir, "muestra.jsonl")) as f:
            ids = [json.loads(l)["id"] for l in f]
        with open(os.path.join(self.dir, "sugerencias.jsonl"), "w") as f:
            for i in ids:
                f.write(json.dumps({"id": i, "tipo": "peticion"}) + "\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), B.manejador(self.dir))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d" % srv.server_address[1]

        def post(t, cuerpo):
            req = urllib.request.Request(url + "/api/etiqueta?t=" + t, data=json.dumps(cuerpo).encode(), method="POST")
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                return e.code
        try:
            for j, i in enumerate(ids):
                self.assertEqual(post("ta", {"id": i, "tipo": "peticion", "area": "salud", "plazo": "15_dias"}), 200)
                self.assertEqual(post("tl", {"id": i, "tipo": "peticion" if j % 2 else "queja", "area": "salud",
                                             "plazo": "15_dias"}), 200)
            self.assertEqual(post("ta", {"id": ids[0], "tipo": "inventado"}), 400)
            self.assertEqual(post("nadie", {"id": ids[0], "tipo": "queja"}), 403)
            d = json.loads(urllib.request.urlopen(url + "/api/datos?t=tl").read())
            self.assertTrue(all(set(r) == {"id", "entidad", "texto"} for r in d["filas"]))   # ni original ni sugerencia
            self.assertEqual(set(d["mias"]), set(ids))
            self.assertTrue(all(m["area"] == "salud" for m in d["mias"].values()))
        finally:
            srv.shutdown()
        k = B.kappa(self.dir)
        self.assertEqual(k["campos"]["tipo"]["n"], len(ids))
        self.assertEqual(k["campos"]["area"]["acuerdo"], 1.0)
        self.assertLess(k["campos"]["tipo"]["kappa"], 1.0)
        self.assertIn("kappa_luis_vs_modelo", k["campos"]["tipo"])
        self.assertIsNone(k["campos"]["area"]["kappa_luis_vs_modelo"])   # la sugerencia no trae área


if __name__ == "__main__":
    unittest.main()
