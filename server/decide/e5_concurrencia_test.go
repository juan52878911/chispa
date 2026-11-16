//go:build ort

package decide

import (
	"context"
	"os"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	"chispa/runtime/tok"
)

// Latencia del codificador e5 con 16 clientes concurrentes según el tamaño del pool de sesiones ORT (T5c). Informativo: no afirma nada, solo mide.
//
//	CHISPA_E5_ONNX=e5.onnx CHISPA_E5_TOK=e5.chispa-tok CHISPA_CONSULTAS_TEXTO=consultas.jsonl CHISPA_PARIDAD_DIR=banco-paridad CHISPA_ORT_LIB=… [CHISPA_E5_SESIONES=1,2,4] [CHISPA_E5_CLIENTES=16] [CHISPA_E5_N=400]
//	go test -tags ort ./decide -run E5Concurrencia -v
func TestE5ConcurrenciaLatenciaPorSesiones(t *testing.T) {
	onnx, tk, textos := os.Getenv("CHISPA_E5_ONNX"), os.Getenv("CHISPA_E5_TOK"), os.Getenv("CHISPA_CONSULTAS_TEXTO")
	if onnx == "" || tk == "" || textos == "" {
		t.Skip("falta CHISPA_E5_ONNX, CHISPA_E5_TOK o CHISPA_CONSULTAS_TEXTO")
	}
	if lib := os.Getenv("CHISPA_ORT_LIB"); lib != "" {
		RutaORT = lib
	}
	cs, _, _ := leerConsultasKNNDe(t, os.Getenv("CHISPA_PARIDAD_DIR"), textos)
	atoi := func(k string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
			return v
		}
		return def
	}
	clientes, total := atoi("CHISPA_E5_CLIENTES", 16), atoi("CHISPA_E5_N", 400)
	pool := []int{1, 2, 4}
	if v := os.Getenv("CHISPA_E5_SESIONES"); v != "" {
		pool = nil
		for _, p := range splitComas(v) {
			n, _ := strconv.Atoi(p)
			pool = append(pool, n)
		}
	}
	tt, err := tok.Abrir(tk)
	if err != nil {
		t.Fatal(err)
	}
	modelo, err := os.ReadFile(onnx)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range pool {
		enc, err := NuevoCodificadorE5ORTN(modelo, NuevoTokenizadorE5(tt).ConMaxLen(E5MaxTokBanco, 128), n)
		if err != nil {
			t.Fatal(err)
		}
		var mu sync.Mutex
		var lat []float64
		var wg sync.WaitGroup
		sig := make(chan int, total)
		for i := 0; i < total; i++ {
			sig <- (i * 10) % len(cs)
		}
		close(sig)
		t0 := time.Now()
		for c := 0; c < clientes; c++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range sig {
					a := time.Now()
					if _, err := enc.Codificar(context.Background(), cs[i].Texto); err != nil {
						t.Error(err)
						return
					}
					d := float64(time.Since(a).Microseconds()) / 1000
					mu.Lock()
					lat = append(lat, d)
					mu.Unlock()
				}
			}()
		}
		wg.Wait()
		seg := time.Since(t0).Seconds()
		sort.Float64s(lat)
		q := func(p float64) float64 { return lat[int(p*float64(len(lat)-1))] }
		t.Logf("sesiones %d, %d clientes, %d consultas: %.1f consultas/s; latencia ms p50 %.1f p95 %.1f p99 %.1f", enc.Sesiones(), clientes, total, float64(total)/seg, q(0.5), q(0.95), q(0.99))
		enc.Close()
	}
}

func splitComas(s string) []string {
	var out []string
	cur := ""
	for _, c := range s {
		if c == ',' {
			out = append(out, cur)
			cur = ""
			continue
		}
		cur += string(c)
	}
	return append(out, cur)
}
