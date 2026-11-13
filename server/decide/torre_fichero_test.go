package decide

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"chispa/format/chispa3"
)

// torre_fichero: la TORR de otro .chispa (p. ej. la int8 del perfil movil-int8) para las torres al vuelo, si es compatible.
func TestTorreFicheroAlterno(t *testing.T) {
	dir := t.TempDir()
	escribe := func(nombre string, torre chispa3.Torre, conTorr bool) string {
		t.Helper()
		secs := []chispa3.Seccion{}
		if conTorr {
			tr, err := chispa3.CodificarTORR(torre)
			if err != nil {
				t.Fatal(err)
			}
			secs = append(secs, chispa3.Seccion{Tipo: chispa3.TORR, Datos: tr})
		} else {
			meta, _ := chispa3.Meta{Version: 1}.Codificar()
			secs = append(secs, chispa3.Seccion{Tipo: chispa3.META, Datos: meta})
		}
		var buf bytes.Buffer
		if err := chispa3.Escribir(&buf, secs); err != nil {
			t.Fatal(err)
		}
		p := filepath.Join(dir, nombre)
		if err := os.WriteFile(p, buf.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	compatible := chispa3.Torre{Dim: 2, MaxLen: 128, Prefijo: "passage: ", ONNX: []byte("onnx-torre-int8")}
	otraDim := compatible
	otraDim.Dim = 3
	casos := []struct {
		nombre, ruta, falla string
	}{
		{"compatible", escribe("int8.chispa", compatible, true), ""},
		{"otra dimensión", escribe("dim3.chispa", otraDim, true), "incompatible"},
		{"sin TORR", escribe("sin.chispa", chispa3.Torre{}, false), "no trae sección TORR"},
		{"no existe", filepath.Join(dir, "no.chispa"), "torre_fichero"},
	}
	for _, c := range casos {
		cfg := Config{CacheEsquemas: 8, TorreFichero: c.ruta}
		alt, err := abrirTorreAlterna(&cfg, conTorreSeccion(t), chispa3.TextoTorreV1)
		if c.falla == "" {
			if err != nil {
				t.Fatalf("%s: %v", c.nombre, err)
			}
			tr, _ := alt.Torre()
			if string(tr.ONNX) != "onnx-torre-int8" {
				t.Fatalf("%s: se usó otra TORR", c.nombre)
			}
			alt.Close()
			continue
		}
		if err == nil || !strings.Contains(err.Error(), c.falla) {
			t.Errorf("%s: err = %v; quiero %q", c.nombre, err, c.falla)
		}
	}
	// En ArmarDesde una torre_fichero incompatible es un error de carga, no una degradación silenciosa.
	cfg := Config{CacheEsquemas: 8, TorreFichero: casos[1].ruta}
	if _, err := ArmarDesde(conTorreSeccion(t), NuevoL0(8), &cfg, tokFalso{}, ConSesionS(sesionFalsa{[][]float32{{3, 0}}})); err == nil || !strings.Contains(err.Error(), "incompatible") {
		t.Fatalf("torre_fichero incompatible debía fallar al armar: %v", err)
	}
}
