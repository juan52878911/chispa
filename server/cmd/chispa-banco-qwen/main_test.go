package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/juan52878911/chispa/server/decide"
)

// El formato que escribe se lee con el cargador del servidor (mismas rutas que usa en producción).
func TestEscribirBancoSeLeeConElCargadorDelServidor(t *testing.T) {
	d := t.TempDir()
	fs := []fila{{Tarea: "t", Etiqueta: "a"}, {Tarea: "t", Etiqueta: "b"}, {Tarea: "t", Etiqueta: "a"}}
	vs := [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0.6, 0.8}}
	base := filepath.Join(d, "banco-t")
	if err := escribirBanco(base, fs, vs); err != nil {
		t.Fatal(err)
	}
	b, err := decide.CargarBancoKNN(base)
	if err != nil {
		t.Fatal(err)
	}
	if b.N() != 3 || b.Dim != 3 || b.NOro != 3 || b.Etiq[1] != "b" || math.Abs(float64(b.X[7])-0.6) > 1e-6 {
		t.Fatalf("%+v", b)
	}
}

func TestEjecutarErroresClaros(t *testing.T) {
	d := t.TempDir()
	sop := filepath.Join(d, "s.jsonl")
	os.WriteFile(sop, []byte(`{"tarea":"t","etiqueta":"a","pregunta":"p","texto":"x"}`+"\n"), 0o644)
	if err := ejecutar("/no/existe.onnx", "/no/existe.bpe", sop, d, "", 1); err == nil {
		t.Fatal("debía fallar")
	}
	os.WriteFile(sop, []byte(`{"tarea":"","etiqueta":"a"}`+"\n"), 0o644)
	if _, err := leerFilas(sop); err == nil {
		t.Fatal("fila sin tarea")
	}
}
