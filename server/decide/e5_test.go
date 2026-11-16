package decide

import (
	"math"
	"testing"
)

func TestConEspacioFinalComoHF(t *testing.T) {
	ids := []int64{0, 41, 1294, 2}
	if got := conEspacioFinal("hola ", ids, 256); len(got) != 5 || got[3] != idEspacioE5 || got[4] != 2 || len(ids) != 4 {
		t.Errorf("espacio final: %v", got)
	}
	for _, tx := range []string{"hola", "", "hola x"} {
		if got := conEspacioFinal(tx, ids, 256); len(got) != 4 {
			t.Errorf("%q no debe cambiar: %v", tx, got)
		}
	}
	if got := conEspacioFinal("hola\n", ids, 256); len(got) != 5 {
		t.Errorf("salto de línea final: %v", got)
	}
	if got := conEspacioFinal("hola ", ids, 4); len(got) != 4 {
		t.Errorf("ya en el tope de tokens: HF trunca antes del ▁: %v", got)
	}
}

func TestMediaNormalizadaE5(t *testing.T) {
	// dos tokens (1,0,0) y (0,1,0): media (0,5, 0,5, 0) → norma 1
	v := mediaNormalizada([]float32{1, 0, 0, 0, 1, 0}, 2, 3)
	if math.Abs(float64(v[0])-math.Sqrt2/2) > 1e-6 || math.Abs(float64(v[1])-math.Sqrt2/2) > 1e-6 || v[2] != 0 {
		t.Errorf("%v", v)
	}
	if z := mediaNormalizada([]float32{0, 0}, 1, 2); z[0] != 0 || z[1] != 0 {
		t.Errorf("vector nulo: %v", z)
	}
}
