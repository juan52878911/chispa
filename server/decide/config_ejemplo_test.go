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

// Las configs de los dos modos del servidor general (W1) son válidas y dicen lo que prometen; las configs anteriores siguen en el modo de siempre.
func TestConfigsDeLosDosModos(t *testing.T) {
	gpu, err := CargarConfig("../deploy/config.modo-gpu.json")
	if err != nil {
		t.Fatal(err)
	}
	if gpu.KNN == nil || gpu.KNN.Modo != ModoKNNGPU || gpu.L3 == nil || gpu.L3.Modo != L3ModoLogprobs || gpu.L3.Ejemplos == nil || *gpu.L3.Ejemplos != 8 || gpu.KNN.ConservarL0L1 {
		t.Fatalf("modo gpu: %+v", gpu)
	}
	cpu, err := CargarConfig("../deploy/config.modo-cpu.json")
	if err != nil {
		t.Fatal(err)
	}
	if cpu.KNN == nil || cpu.KNN.Modo != ModoKNNCPU || cpu.L3 != nil || cpu.KNN.NombreCodificador() != CodificadorE5Small || cpu.KNN.E5ONNX == "" || cpu.KNN.ConservarL0L1 {
		t.Fatalf("modo cpu: %+v", cpu)
	}
	// modo gpu sin l3 no arranca; con un modo desconocido tampoco
	mala := gpu
	mala.L3 = nil
	if err := mala.Validar(); err == nil {
		t.Error("modo gpu sin l3 debía fallar")
	}
	// las configs anteriores: modo clásico (el escalado por margen de siempre), sin examples ni tope nuevo cambiados
	for _, f := range []string{"config.ejemplo.json", "config.ejemplo-qwen3emb.json"} {
		c, err := CargarConfig("../deploy/" + f)
		if err != nil {
			t.Fatal(f, err)
		}
		if c.KNN.Modo != ModoKNNClasico || c.KNN.CacheVectoresMB != 0 || c.Limites.MaxEjemplos != 0 && c.Limites.MaxEjemplos != MaxEjemplosPregunta {
			t.Errorf("%s cambió de modo: %+v", f, c.KNN)
		}
	}
	// y sin la clave limites.max_ejemplos el servidor usa los topes por defecto
	l := normalizaLimites(Limites{})
	if l.MaxEjemplos != MaxEjemplosPregunta || l.MaxEjemploTexto != MaxEjemploTextoByte {
		t.Errorf("límites por defecto: %+v", l)
	}
}
