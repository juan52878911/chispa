package decide

import (
	"context"
	"testing"
)

// Una pregunta score sale «experimental» salvo que su tarea tenga umbrales calibrados en el banco (T3b).
func TestScoreExperimentalSoloSinCalibrar(t *testing.T) {
	c, err := NuevaCascada([]Capa{{"L3", 0.5}}, []Decisor{NuevoL3()}, NuevasMetricas())
	if err != nil {
		t.Fatal(err)
	}
	p := Pregunta{ID: "x", Kind: KindScore, Texto: "calibrada", Opciones: []Opcion{{ID: "1"}, {ID: "2"}, {ID: "3"}}}
	if !c.Decidir(context.Background(), p).Experimental {
		t.Fatal("sin ConScoreCalibrado, score debe salir experimental")
	}
	c.ConScoreCalibrado(func(q string) bool { return q == "calibrada" })
	if c.Decidir(context.Background(), p).Experimental {
		t.Fatal("con umbrales calibrados, score no debe salir experimental")
	}
	p.Texto = "otra"
	if !c.Decidir(context.Background(), p).Experimental {
		t.Fatal("una tarea score sin umbrales sigue experimental")
	}
	p.Kind = KindChoice
	if c.Decidir(context.Background(), p).Experimental {
		t.Fatal("choice nunca es experimental")
	}
}
