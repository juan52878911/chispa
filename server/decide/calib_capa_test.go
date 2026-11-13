package decide

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"chispa/format/chispa3"
)

// La calibración por capa se aplica a la confianza antes del τ, y la respuesta dice si la capa que contestó está calibrada.
func TestCascadaCalibracionPorCapa(t *testing.T) {
	r := Respuesta{Answer: "a", Probs: map[string]float64{"a": 0.85, "b": 0.15}, Conf: 0.85}
	armar := func(cal map[string]chispa3.CalibCapa, calibradas map[string]bool) *Cascada {
		c, err := NuevaCascada([]Capa{{"L0", 0.8}, {"L1", 0.8}},
			[]Decisor{&falso{n: "L0", r: r}, &falso{n: "L1", r: r}}, NuevasMetricas())
		if err != nil {
			t.Fatal(err)
		}
		if cal != nil || calibradas != nil {
			c.ConCalibracion(cal, calibradas)
		}
		return c
	}
	p := Pregunta{ID: "q", Kind: KindChoice}
	// Sin ConCalibracion: como antes, contesta L0 y no sale «calibrada».
	res := armar(nil, nil).Decidir(context.Background(), p)
	if res.Layer != "L0" || res.Calibrada != nil {
		t.Fatalf("sin calibración: %+v", res)
	}
	b, _ := json.Marshal(res)
	if strings.Contains(string(b), "calibrada") {
		t.Fatalf("sin ConCalibracion no debe salir «calibrada»: %s", b)
	}
	// L0 sin calibrar: contesta por encima de su τ pero queda marcada.
	res = armar(map[string]chispa3.CalibCapa{}, map[string]bool{}).Decidir(context.Background(), p)
	if res.Layer != "L0" || res.Calibrada == nil || *res.Calibrada {
		t.Fatalf("L0 sin calibrar debe salir con calibrada=false: %+v", res)
	}
	if b, _ := json.Marshal(res); !strings.Contains(string(b), `"calibrada":false`) {
		t.Fatalf("JSON sin calibrada=false: %s", b)
	}
	// L0 calibrada a la baja (isotónica 0,85 → 0,5): no llega a su τ y contesta L1, calibrada por la identidad.
	a, bb := 1.0, 0.0
	cal := map[string]chispa3.CalibCapa{
		"L0": {Metodo: chispa3.CalibIsotonica, X: []float64{0, 1}, Y: []float64{0, 0.588}},
		"L1": {Metodo: chispa3.CalibPlatt, A: &a, B: &bb},
	}
	res = armar(cal, map[string]bool{"L0": true, "L1": true}).Decidir(context.Background(), p)
	if res.Layer != "L1" || res.Calibrada == nil || !*res.Calibrada {
		t.Fatalf("L0 calibrada bajo τ debe pasar a L1 calibrada: %+v", res)
	}
	// Escalada: no lleva «calibrada».
	baja := map[string]chispa3.CalibCapa{"L0": cal["L0"], "L1": cal["L0"]}
	res = armar(baja, map[string]bool{"L0": true, "L1": true}).Decidir(context.Background(), p)
	if !res.Escalate || res.Calibrada != nil {
		t.Fatalf("escalada: %+v", res)
	}
}
