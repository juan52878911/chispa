package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
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
	if err := ejecutar("/no/existe.onnx", "/no/existe.bpe", sop, d, "", 1, -1, semillaPorDefecto, true); err == nil {
		t.Fatal("debía fallar")
	}
	os.WriteFile(sop, []byte(`{"tarea":"","etiqueta":"a"}`+"\n"), 0o644)
	if _, err := leerFilas(sop); err == nil {
		t.Fatal("fila sin tarea")
	}
}

func leerSoporteFixture(t *testing.T) []fila {
	t.Helper()
	fs, err := leerFilas("testdata/soporte.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	return fs
}

// La elección de ejemplos coincide con la de Python (llm_logprobs.elegir_ejemplos con random.Random) byte a byte, con varias semillas (una de más de 32 bits) y k.
func TestEjemplosMismaEleccionQuePython(t *testing.T) {
	fs := leerSoporteFixture(t)
	for _, c := range []struct {
		semilla uint64
		k       int
	}{{20261115, 8}, {20261115, 3}, {7, 8}, {1<<33 + 5, 8}, {0, 5}, {20261115, 40}} {
		want, err := os.ReadFile(fmt.Sprintf("testdata/ejemplos-%d-%d.jsonl", c.semilla, c.k))
		if err != nil {
			t.Fatal(err)
		}
		base := filepath.Join(t.TempDir(), "banco-t")
		n, err := escribirEjemplos(base, fs, c.k, c.semilla)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := os.ReadFile(base + ".ejemplos.jsonl")
		if string(got) != string(want) {
			t.Errorf("semilla %d k %d: difiere de Python\n got %.300s\nwant %.300s", c.semilla, c.k, got, want)
		}
		if n != strings.Count(string(want), "\n") {
			t.Errorf("semilla %d k %d: devuelve %d ejemplos, el fichero tiene %d", c.semilla, c.k, n, strings.Count(string(want), "\n"))
		}
	}
}

// El formato es el del banco de e5 (generar_banco.py): las líneas de un .ejemplos.jsonl de e5 se reescriben idénticas, y el servidor lee lo que escribimos.
func TestEjemplosFormatoDeUnBancoE5(t *testing.T) {
	const e5 = "../../decide/testdata/knn/banco-a"
	b, err := os.ReadFile(e5 + ".ejemplos.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	var re strings.Builder
	var fs []fila
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var f fila
		if err := json.Unmarshal([]byte(l), &f); err != nil {
			t.Fatal(err)
		}
		fs = append(fs, f)
		re.WriteString(lineaEjemplo(f))
	}
	if re.String() != string(b) {
		t.Fatalf("formato distinto del de e5:\n got %s\nwant %s", re.String(), b)
	}
	base := filepath.Join(t.TempDir(), "banco-a")
	if _, err := escribirEjemplos(base, fs, 8, semillaPorDefecto); err != nil {
		t.Fatal(err)
	}
	ej, err := decide.CargarEjemplosFS(base)
	if err != nil || len(ej) != len(fs) {
		t.Fatal(err, len(ej))
	}
	for _, e := range ej { // con las filas del banco de e5 sale el mismo conjunto de ejemplos (solo cambia el orden)
		ok := false
		for _, f := range fs {
			ok = ok || (f.ID == e.ID && f.Texto == e.Texto && f.Etiqueta == e.Etiqueta)
		}
		if !ok {
			t.Errorf("ejemplo %s no está en el original", e.ID)
		}
	}
}

func TestEjemplosAvisosYErrores(t *testing.T) {
	base := filepath.Join(t.TempDir(), "banco-x")
	// sin id/tipo/opciones y k por defecto: sin fichero, sin error
	if n, err := escribirEjemplos(base, []fila{{Tarea: "t", Etiqueta: "a", Texto: "x"}}, -1, 1); err != nil || n != 0 {
		t.Fatal(n, err)
	}
	if _, err := os.Stat(base + ".ejemplos.jsonl"); err == nil {
		t.Fatal("no debía escribir el fichero")
	}
	// k explícito sin esos campos: error claro
	if _, err := escribirEjemplos(base, []fila{{Tarea: "t", Etiqueta: "a", Texto: "x"}}, 8, 1); err == nil || !strings.Contains(err.Error(), "id, tipo y opciones") {
		t.Fatalf("debía explicar qué falta: %v", err)
	}
	// k = 0: nada
	if n, err := escribirEjemplos(base, []fila{{ID: "1", Tipo: "choice", Etiqueta: "a", Opciones: []string{"a", "b"}}}, 0, 1); err != nil || n != 0 {
		t.Fatal(n, err)
	}
	// etiqueta que no es una opción
	if _, err := escribirEjemplos(base, []fila{{ID: "1", Tipo: "choice", Etiqueta: "z", Opciones: []string{"a", "b"}}}, 8, 1); err == nil {
		t.Fatal("etiqueta fuera de opciones")
	}
}

// El formato pool lleva todo el oro con su `fila` y marca las k fijas con su posición: quitando esas columnas, las fijas son las del formato viejo y en su orden.
func TestEjemplosFormatoPool(t *testing.T) {
	b, err := os.ReadFile("testdata/soporte.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	var fs []fila
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var f fila
		if err := json.Unmarshal([]byte(l), &f); err != nil {
			t.Fatal(err)
		}
		fs = append(fs, f)
	}
	base := filepath.Join(t.TempDir(), "banco-t")
	if _, err := escribirEjemplosFmt(base, fs, 8, semillaPorDefecto, true); err != nil {
		t.Fatal(err)
	}
	if _, err := escribirEjemplos(base+"v", fs, 8, semillaPorDefecto); err != nil {
		t.Fatal(err)
	}
	viejo, _ := decide.CargarEjemplosFS(base + "v")
	pool, err := decide.CargarEjemplosFS(base)
	if err != nil {
		t.Fatal(err)
	}
	oro := 0
	for _, f := range fs {
		if f.Split == "" || f.Split == "train" {
			oro++
		}
	}
	if len(pool) != oro || len(viejo) != 8 {
		t.Fatalf("pool %d (oro %d), fijas %d", len(pool), oro, len(viejo))
	}
	fijas := make([]string, 8)
	for _, e := range pool {
		if e.Fila == nil || fs[*e.Fila].ID != e.ID || fs[*e.Fila].Etiqueta != e.Etiqueta {
			t.Fatalf("fila mal: %+v", e)
		}
		if e.Fs > 0 {
			fijas[e.Fs-1] = e.ID
		}
	}
	for i, e := range viejo {
		if fijas[i] != e.ID {
			t.Fatalf("la fija %d es %q y en el formato viejo %q", i+1, fijas[i], e.ID)
		}
	}
}
