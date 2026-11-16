package main

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/juan52878911/chispa/server/decide"
)

// Con `knn` en la config, armarModelo monta L2 = kNN por tarea y la cascada es L2→L3 (L0 y L1 se saltan); con un banco que no existe, no arranca.
func TestKNNDesdeConfig(t *testing.T) {
	banco, _ := filepath.Abs("../../decide/testdata/knn/banco-b")
	cfg := `{"knn": {"umbral": 0.5, "tareas": {"b": {"banco": "` + banco + `", "preguntas": ["¿b?"]}}}}`
	a, err := armarModelo(decide.NuevasMetricas(), escribeCfg(t, cfg), "", "")
	if err != nil {
		t.Fatal(err)
	}
	if a.Cerrar != nil {
		defer a.Cerrar()
	}
	// sin vector inyectado y sin L3 el kNN no opina: la cascada escala (no hay codificador e5 en el servidor)
	r := a.Cascada.Decidir(context.Background(), decide.Pregunta{ID: "q", Estado: "hola", Texto: "¿b?", Kind: "choice", Opciones: []decide.Opcion{{ID: "p"}, {ID: "q"}}})
	if !r.Escalate {
		t.Fatalf("%+v", r)
	}
	mal := `{"knn": {"umbral": 0.5, "tareas": {"b": {"banco": "/no/existe"}}}}`
	if _, err := armarModelo(decide.NuevasMetricas(), escribeCfg(t, mal), "", ""); err == nil || !strings.Contains(err.Error(), "knn") {
		t.Fatalf("un banco que no existe debía fallar: %v", err)
	}
}
