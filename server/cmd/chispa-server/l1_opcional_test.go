package main

import (
	"context"
	"testing"

	"chispa/format/chispa3"
	"github.com/juan52878911/chispa/server/decide"
)

// L1 es opcional: con un .chispa sin sección L1 la cascada se salta la capa (no hay stub que nunca opina) y /healthz la da inactiva.
func TestSinSeccionL1LaCascadaSeLaSalta(t *testing.T) {
	const modelo = "../../testdata/caida/riesgo.chispa"
	f, err := chispa3.Abrir(modelo)
	if err != nil {
		t.Fatal(err)
	}
	tieneL1 := f.Tiene(chispa3.L1)
	f.Close()
	if tieneL1 {
		t.Fatal("el fixture debía venir sin L1")
	}
	a, err := armarModelo(decide.NuevasMetricas(), "", modelo, "")
	if err != nil {
		t.Fatal(err)
	}
	if a.Cerrar != nil {
		defer a.Cerrar()
	}
	if _, ok := a.Cascada.Taus()["L1"]; ok {
		t.Fatalf("la cascada no debía tener L1: %v", a.Cascada.Taus())
	}
	if cp := a.Cascada.Capacidades(); cp.Capas["L1"] || len(cp.Capas) != 4 {
		t.Fatalf("capacidades: %v", cp.Capas)
	}
	r := a.Cascada.Decidir(context.Background(), decide.Pregunta{ID: "q", Kind: decide.KindChoice, Texto: "¿riesgo?", Estado: "hola",
		Opciones: []decide.Opcion{{ID: "a"}, {ID: "b"}}})
	if r.Layer == "L1" || r.Incompleto {
		t.Fatalf("respuesta: %+v", r)
	}
}
