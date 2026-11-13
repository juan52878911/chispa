package chispa3

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// Una CALIB sin tau_escalate se leía como τ=0 y L1 lo aceptaba todo: ahora es un error de carga.
func TestCalibSinTauSeRechaza(t *testing.T) {
	for _, j := range []string{`{"tipos":{}}`, `{"tipos":{},"tau_escalate":null}`} {
		if _, err := DecodificarCalib([]byte(j)); !errors.Is(err, ErrCalibSinTau) || !errors.Is(err, ErrContenido) {
			t.Errorf("%s: err = %v; quiero ErrCalibSinTau", j, err)
		}
	}
	if c, err := DecodificarCalib([]byte(`{"tipos":{},"tau_escalate":0}`)); err != nil || c.TauEscalate != 0 {
		t.Errorf("τ = 0 explícito es válido: %v %v", c, err)
	}
}

func TestCalibracionPorCapa(t *testing.T) {
	j := `{"tipos":{},"tau_escalate":0.7,"calibracion":{"L0":{"metodo":"platt","a":1,"b":0},"L1":{"metodo":"isotonica","x":[0.2,0.6,1],"y":[0,0.5,0.9]}}}`
	c, err := DecodificarCalib([]byte(j))
	if err != nil {
		t.Fatal(err)
	}
	// Platt con a=1, b=0 es la identidad.
	if v := c.Calibracion["L0"].Aplicar(0.83); math.Abs(v-0.83) > 1e-9 {
		t.Errorf("platt identidad: %v", v)
	}
	iso := c.Calibracion["L1"]
	for _, k := range []struct{ c, quiere float64 }{{0, 0}, {0.1, 0}, {0.2, 0}, {0.4, 0.25}, {0.6, 0.5}, {0.8, 0.7}, {1, 0.9}} {
		if v := iso.Aplicar(k.c); math.Abs(v-k.quiere) > 1e-9 {
			t.Errorf("isotonica(%v) = %v; quiero %v", k.c, v, k.quiere)
		}
	}
	// Ida y vuelta.
	b, err := c.Codificar()
	if err != nil {
		t.Fatal(err)
	}
	if d, err := DecodificarCalib(b); err != nil || len(d.Calibracion) != 2 {
		t.Fatalf("ida y vuelta: %v %v", d, err)
	}
	malos := map[string]string{
		"capa desconocida":       `{"L9":{"metodo":"platt","a":1,"b":0}}`,
		"método desconocido":     `{"L1":{"metodo":"beta"}}`,
		"platt sin b":            `{"L1":{"metodo":"platt","a":1}}`,
		"platt a negativo":       `{"L1":{"metodo":"platt","a":-1,"b":0}}`,
		"isotonica x no crece":   `{"L1":{"metodo":"isotonica","x":[0.5,0.5],"y":[0,1]}}`,
		"isotonica y decrece":    `{"L1":{"metodo":"isotonica","x":[0,1],"y":[1,0]}}`,
		"isotonica fuera de 0,1": `{"L1":{"metodo":"isotonica","x":[0,2],"y":[0,1]}}`,
		"isotonica corta":        `{"L1":{"metodo":"isotonica","x":[0],"y":[0]}}`,
	}
	for n, m := range malos {
		_, err := DecodificarCalib([]byte(`{"tipos":{},"tau_escalate":0.7,"calibracion":` + m + `}`))
		if !errors.Is(err, ErrContenido) || !strings.Contains(err.Error(), "calibracion") {
			t.Errorf("%s: err = %v", n, err)
		}
	}
}
