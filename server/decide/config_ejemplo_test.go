package decide

import (
	"testing"
	time_ "time"
)

// deploy/config.ejemplo.json (la config del producto: knn + e5 + L3 por logprobs con fs8 y los topes de R10) es una config válida.
func TestConfigEjemploDelProductoEsValida(t *testing.T) {
	c, err := CargarConfig("../deploy/config.ejemplo.json")
	if err != nil {
		t.Fatal(err)
	}
	if c.KNN == nil || c.KNN.E5ONNX == "" || c.L3 == nil || c.L3.Modo != L3ModoLogprobs || c.L3.Ejemplos == nil || *c.L3.Ejemplos != 8 {
		t.Fatalf("config de ejemplo sin knn, e5 o fs8: %+v", c)
	}
	if time := c.L3.TimeoutFila; time != Duracion(30*time_.Second) || c.L3.MaxConcurrente != 32 || c.L3.MaxCola == nil || *c.L3.MaxCola != 64 {
		t.Errorf("topes de R10: timeout_fila %v max_concurrente %d max_cola %v", time, c.L3.MaxConcurrente, c.L3.MaxCola)
	}
}
