package decide

import (
	"encoding/json"
	"os"
	"testing"
)

type filaRegla struct {
	Tipo     string    `json:"tipo"`
	Opciones []string  `json:"opciones"`
	Probs    []float64 `json:"probs"`
	Esperada *string   `json:"esperada"`
}

// Paridad fila a fila con lineas_base_v2.top1(esp=False) sobre un fixture sintético (model/alt/w1_fixture_score.py): empates, colas en 0,5 exactas y
// opciones fuera de orden. Reproduce, con probabilidades sintéticas, el mecanismo del 0,4700 (regla) frente al 0,4609 (argmax) medidos en dev: las
// filas guardadas de L2-qwen3emb no están en el repo, así que lo que se comprueba es que Go y Python dan la misma respuesta en cada fila.
func TestReglaKNNParidadConPython(t *testing.T) {
	b, err := os.ReadFile("../testdata/w1_score_regla.json")
	if err != nil {
		t.Fatal(err)
	}
	var filas []filaRegla
	if err := json.Unmarshal(b, &filas); err != nil {
		t.Fatal(err)
	}
	var difArgmax, invalidas, score int
	for i, f := range filas {
		got, ok := RespuestaKNN(f.Tipo, f.Opciones, f.Probs)
		if f.Esperada == nil {
			invalidas++
			if ok {
				t.Fatalf("fila %d: Python da None y Go %q", i, got)
			}
			continue
		}
		if !ok || got != *f.Esperada {
			t.Fatalf("fila %d (%s %v %v): Go %q (ok=%v), Python %q", i, f.Tipo, f.Opciones, f.Probs, got, ok, *f.Esperada)
		}
		if f.Tipo == KindScore {
			score++
			if ops := f.Opciones; ops[argmaxF(f.Probs)] != got {
				difArgmax++
			}
		}
	}
	if len(filas) < 400 || invalidas == 0 || score < 200 || difArgmax == 0 {
		t.Fatalf("fixture degenerado: %d filas, %d inválidas, %d score, %d donde la regla difiere del argmax", len(filas), invalidas, score, difArgmax)
	}
}

func TestReglaKNNCasosConocidos(t *testing.T) {
	for _, c := range []struct {
		kind string
		ops  []string
		p    []float64
		want string
	}{
		{KindScore, []string{"1", "2", "3"}, []float64{0.5, 0.25, 0.25}, "1"}, // cola de 2 = 0,5 exacta: no sube
		{KindScore, []string{"1", "2", "3"}, []float64{0.4, 0.2, 0.4}, "2"},   // la mediana está en medio aunque el argmax sea 1
		{KindScore, []string{"10", "2", "1"}, []float64{0.5, 0.3, 0.2}, "2"},  // orden por entero: 1 < 2 < 10 (por texto saldría otro)
		{KindChoice, []string{"a", "b", "c"}, []float64{0.4, 0.4, 0.2}, "a"},  // empate: la primera
		{KindNoul, []string{"false", "true"}, []float64{0.1, 0.9}, "true"},
	} {
		if got, ok := RespuestaKNN(c.kind, c.ops, c.p); !ok || got != c.want {
			t.Errorf("%s %v %v: %q, quería %q", c.kind, c.ops, c.p, got, c.want)
		}
	}
	if _, ok := RespuestaKNN(KindScore, []string{"a", "b"}, []float64{0.5, 0.5}); ok {
		t.Error("ids no enteros en score: debe dejar al argmax")
	}
	if _, ok := RespuestaKNN(KindChoice, []string{"a", "b"}, []float64{0, 0}); ok {
		t.Error("suma 0: inválida")
	}
}
