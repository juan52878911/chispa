//go:build ort

package decide

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func ortQwenLib(t *testing.T) {
	t.Helper()
	lib := os.Getenv("CHISPA_ORT_LIB")
	if lib == "" {
		t.Skip("falta CHISPA_ORT_LIB")
	}
	RutaORT = lib
}

type filaFixtureQwen struct{ ID, Tipo, Pregunta, Texto string }

func leerFixtureQwen(t *testing.T) []filaFixtureQwen {
	t.Helper()
	f, err := os.Open(dirQwenTok + "fixture_paridad.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []filaFixtureQwen
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<22), 1<<22)
	for sc.Scan() {
		var r filaFixtureQwen
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		out = append(out, r)
	}
	return out
}

func cosenoQwen(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// Paridad del codificador Go (tokenizador BPE + ORT + pooling del último token) con transformers/torch sobre el Qwen3 en miniatura
// (testdata/qwen: modelo aleatorio de la arquitectura real exportado con exportar_qwen3emb_onnx.py; vectores de qwen3emb_referencia.py con relleno a la izquierda en lote):
// coseno ≥ 0,999 en los 200 textos del fixture, 20 de ellos truncados a 320 tokens. Con el modelo real lo repite el ud (qwen_paridad_real_test.go).
func TestQwenParidadMiniFP32(t *testing.T) {
	ortQwenLib(t)
	b := bpeMini(t)
	c, err := NuevoCodificadorQwenORT("testdata/qwen/qwen3emb-fp32.onnx", b, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ref, n, dim, err := leerNpyF32("testdata/qwen/vectores_mini.npy")
	if err != nil {
		t.Fatal(err)
	}
	fx := leerFixtureQwen(t)
	if n != len(fx) || n != 200 {
		t.Fatalf("%d vectores y %d filas", n, len(fx))
	}
	min := 1.0
	for i, r := range fx {
		var v []float32
		if r.Tipo == "consulta" {
			v, err = c.CodificarConsulta(context.Background(), r.Pregunta, r.Texto)
		} else {
			v, err = c.Codificar(context.Background(), r.Texto)
		}
		if err != nil {
			t.Fatalf("%s: %v", r.ID, err)
		}
		if len(v) != dim {
			t.Fatalf("dim %d vs %d", len(v), dim)
		}
		if cs := cosenoQwen(v, ref[i*dim:(i+1)*dim]); cs < min {
			min = cs
		}
		if cs := cosenoQwen(v, ref[i*dim:(i+1)*dim]); cs < 0.999 {
			t.Errorf("%s: coseno %.6f", r.ID, cs)
		}
	}
	t.Logf("coseno mínimo fp32 sobre %d textos: %.8f", n, min)
}

func TestQwenParidadMiniInt8Informativa(t *testing.T) {
	ortQwenLib(t)
	c, err := NuevoCodificadorQwenORT("testdata/qwen/qwen3emb-int8.onnx", bpeMini(t), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ref, _, dim, _ := leerNpyF32("testdata/qwen/vectores_mini.npy")
	min := 1.0
	for i, r := range leerFixtureQwen(t)[:60] {
		v, err := c.CodificarConsulta(context.Background(), r.Pregunta, r.Texto)
		if err != nil {
			t.Fatal(err)
		}
		min = math.Min(min, cosenoQwen(v, ref[i*dim:(i+1)*dim]))
	}
	if min < 0.95 {
		t.Fatalf("int8: coseno mínimo %.4f", min)
	}
	t.Logf("coseno mínimo int8 (60 textos, modelo aleatorio): %.5f", min)
}

func TestQwenErroresClarosAlAbrir(t *testing.T) {
	ortQwenLib(t)
	b := bpeMini(t)
	// otro ONNX (el de L2 sintético): falta chispa.pooling
	if _, err := NuevoCodificadorQwenORT("testdata/l2_sintetico_ml256.onnx", b, 1); err == nil || !strings.Contains(err.Error(), "chispa.pooling") {
		t.Fatalf("ONNX ajeno: %v", err)
	}
	// falta el fichero de pesos que va al lado
	d := t.TempDir()
	datos, _ := os.ReadFile("testdata/qwen/qwen3emb-fp32.onnx")
	os.WriteFile(filepath.Join(d, "qwen3emb-fp32.onnx"), datos, 0o600)
	if _, err := NuevoCodificadorQwenORT(filepath.Join(d, "qwen3emb-fp32.onnx"), b, 1); err == nil {
		t.Fatal("sin el .data debía fallar")
	} else {
		t.Logf("sin .data: %v", err)
	}
	if _, err := NuevoCodificadorQwenORT(filepath.Join(d, "no-existe.onnx"), b, 1); err == nil || !strings.Contains(err.Error(), "no-existe") {
		t.Fatalf("falta el ONNX: %v", err)
	}
}

// Pool de 2 sesiones con 8 goroutines: mismos vectores que en serie y sin condiciones de carrera (go test -race).
func TestQwenConcurrencia(t *testing.T) {
	ortQwenLib(t)
	c, err := NuevoCodificadorQwenORT("testdata/qwen/qwen3emb-fp32.onnx", bpeMini(t), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if c.Sesiones() != 2 {
		t.Fatalf("sesiones %d", c.Sesiones())
	}
	fx := leerFixtureQwen(t)[:24]
	serie := make([][]float32, len(fx))
	for i, r := range fx {
		serie[i], _ = c.CodificarConsulta(context.Background(), r.Pregunta, r.Texto)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, r := range fx {
				v, err := c.CodificarConsulta(context.Background(), r.Pregunta, r.Texto)
				if err != nil || cosenoQwen(v, serie[i]) < 0.999999 {
					t.Errorf("%s: %v", r.ID, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	// con el ctx cancelado y las sesiones libres puede ir bien o dar ctx.Err: no debe colgarse
	c.Codificar(ctx, "x")
}
