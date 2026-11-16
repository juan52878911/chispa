package decide

import "testing"

// El recorte a top_k y la guía de capas previas siguen existiendo, pero con `knn` y sin `l3.top_k` están apagados (L3 ve todas las opciones,
// como llm_logprobs.py); con l3.top_k explícito mandan; sin knn no cambia nada.
func TestL3TopKConKNN(t *testing.T) {
	topK := func(knn bool, k int) int {
		c := Config{L3: &ConfigL3{URL: "http://127.0.0.1:1/v1", Modelo: "m", TopK: k}}
		if knn {
			c.KNN = &ConfigKNN{}
		}
		d, err := c.DecisorL3()
		if err != nil {
			t.Fatal(err)
		}
		return d.(*L3LLM).cfg.TopK
	}
	if k := topK(true, 0); k != L3TopKSinRecorte {
		t.Errorf("con knn y sin top_k: %d", k)
	}
	if k := topK(true, 5); k != 5 {
		t.Errorf("con knn y top_k explícito: %d", k)
	}
	if k := topK(false, 0); k != 0 { // 0 = el de por defecto (8), como siempre
		t.Errorf("sin knn: %d", k)
	}
}
