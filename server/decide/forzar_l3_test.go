package decide

import (
	"context"
	"testing"
)

type decisorContadorL3 struct {
	nombre string
	r      Respuesta
	n      int
}

func (d *decisorContadorL3) Nombre() string { return d.nombre }
func (d *decisorContadorL3) Decidir(context.Context, Pregunta) (Respuesta, error) {
	d.n++
	return d.r, nil
}

func TestCascadaSoloL3SaltaLasDemasCapasYAceptaConfBaja(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -1.0, "B": -1.1}} // casi 50/50: bajo cualquier τ razonable
	l3 := nuevoLP(t, f, nil)
	l0 := &decisorContadorL3{nombre: "L0", r: Respuesta{Answer: "op1", Probs: map[string]float64{"op0": 0, "op1": 1}, Conf: 1}}
	capas := []Capa{{"L0", 0.8}, {"L3", 0.9}}
	ds := []Decisor{l0, l3}
	p := preguntaLP(2)
	normal, err := NuevaCascada(capas, ds, NuevasMetricas())
	if err != nil {
		t.Fatal(err)
	}
	if r := normal.Decidir(context.Background(), p); r.Layer != "L0" || l0.n != 1 {
		t.Fatalf("sin forzar contesta L0: %+v", r)
	}
	forz, _ := NuevaCascada(capas, ds, NuevasMetricas())
	forz.SoloL3()
	r := forz.Decidir(context.Background(), p)
	if l0.n != 1 || r.Layer != "L3" || r.Escalate || r.Answer != "op0" || len(r.Probs) != 2 {
		t.Errorf("forzado: L0 no debía correr y L3 contestar aun con confianza < τ: %+v (L0 llamadas %d)", r, l0.n)
	}
}
