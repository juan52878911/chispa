package decide

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// Los histogramas llegan hasta 30 s (antes saturaban en 1 s) y cada respuesta de L3 lleva su espera y su tiempo en cabeceras.
func TestL3CabecerasPorPeticionYBucketsHasta30s(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -0.1, "B": -3}, espera: 150 * time.Millisecond}
	l, m, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxConcurrente = 1 })
	casc, err := NuevaCascada([]Capa{{"L3", 0.5}}, []Decisor{l}, m)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NuevoServidor(casc, m, 0).Handler())
	defer srv.Close()
	cuerpo, _ := json.Marshal(map[string]any{"state": "hola", "questions": []any{map[string]any{"id": "q", "question": "¿?", "kind": "choice",
		"options": []any{map[string]any{"id": "a"}, map[string]any{"id": "b"}}}}})
	var wg sync.WaitGroup
	esp, tl := make([]float64, 3), make([]float64, 3)
	for i := range esp {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := http.Post(srv.URL+"/v1/decide", "application/json", bytes.NewReader(cuerpo))
			if err != nil {
				t.Error(err)
				return
			}
			r.Body.Close()
			esp[i], _ = strconv.ParseFloat(r.Header.Get("X-Chispa-L3-Espera-Ms"), 64)
			tl[i], _ = strconv.ParseFloat(r.Header.Get("X-Chispa-L3-Ms"), 64)
		}()
	}
	wg.Wait()
	var maxEsp float64
	for i := range esp {
		if tl[i] < 140 {
			t.Errorf("L3-Ms de la petición %d: %v (el vLLM falso tarda 150 ms)", i, tl[i])
		}
		maxEsp = max(maxEsp, esp[i])
	}
	if maxEsp < 250 { // con un hueco y 3 peticiones, la última espera ≈ 2 × 150 ms
		t.Errorf("la espera en el semáforo de la última debía ser ≈ 300 ms: %v", esp)
	}
	txt := metricasTexto(m)
	for _, s := range []string{`chispa_l3_espera_segundos_bucket{le="30"}`, `chispa_l3_latencia_segundos_bucket{le="10"}`, `chispa_capa_latencia_segundos_bucket{capa="L3",le="30"}`} {
		if !strings.Contains(txt, s) {
			t.Errorf("falta %s", s)
		}
	}
	// una espera de 3 s ya no cae en +Inf ni en el bucket de 1 s
	var h histo
	h.observa(3 * time.Second)
	var b strings.Builder
	h.escribe(&b, "x")
	if !strings.Contains(b.String(), `x_bucket{le="2"} 0`) || !strings.Contains(b.String(), `x_bucket{le="5"} 1`) {
		t.Errorf("3 s debe caer en el bucket de 5 s:\n%s", b.String())
	}
}
