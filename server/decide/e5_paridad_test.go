//go:build ort

package decide

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"chispa/runtime/tok"
)

// Paridad del codificador e5 en Go con los vectores de Python (criterio congelado: model/alt/crit-r10-t5b.json).
//
//	CHISPA_PARIDAD_DIR=…/banco-paridad  CHISPA_CONSULTAS_TEXTO=consultas.jsonl (con texto; model/alt/consultas_con_texto.py)
//	CHISPA_E5_ONNX=e5.onnx  CHISPA_E5_TOK=e5.chispa-tok  CHISPA_ORT_LIB=libonnxruntime.so|.dylib  [CHISPA_E5_N=N]
//	go test -tags ort ./decide -run E5Paridad -v
//
// Mide, sobre las consultas del fixture: coseno entre el vector de Go y el de Python (consultas.npy) —mín, p01, p50—, coincidencia del top-1
// del kNN con el vector de Go frente al de Python, y (informativo) coincidencia de escala y motivo; y la latencia del codificador.
// Decide el criterio: coseno p01 ≥ 0,999 y top-1 ≥ 99,5 %.
func TestE5ParidadVectoresYDecision(t *testing.T) {
	dir, onnx, tk := os.Getenv("CHISPA_PARIDAD_DIR"), os.Getenv("CHISPA_E5_ONNX"), os.Getenv("CHISPA_E5_TOK")
	if dir == "" || onnx == "" || tk == "" {
		t.Skip("falta CHISPA_PARIDAD_DIR, CHISPA_E5_ONNX o CHISPA_E5_TOK")
	}
	if lib := os.Getenv("CHISPA_ORT_LIB"); lib != "" {
		RutaORT = lib
	}
	textos := os.Getenv("CHISPA_CONSULTAS_TEXTO")
	if textos == "" {
		textos = filepath.Join(dir, "consultas.jsonl")
	}
	cs, x, dim := leerConsultasKNNDe(t, dir, textos)
	if len(cs) == 0 || cs[0].Texto == "" {
		t.Fatal("el jsonl no trae `texto` (CHISPA_CONSULTAS_TEXTO)")
	}
	n := len(cs)
	if v := os.Getenv("CHISPA_E5_N"); v != "" {
		var m int
		for _, c := range v {
			m = m*10 + int(c-'0')
		}
		if m > 0 && m < n {
			n = m
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
	enc, err := NuevoCodificadorE5ORT(modelo, NuevoTokenizadorE5(tt).ConMaxLen(E5MaxTokBanco, 128))
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	// tareas del fixture
	var tareas []string
	fs, _ := filepath.Glob(filepath.Join(dir, "banco-*.npy"))
	for _, f := range fs {
		b := filepath.Base(f)
		tareas = append(tareas, b[len("banco-"):len(b)-len(".npy")])
	}
	ref := leerReferenciaKNN(t, dir)
	k := knnDeDir(t, dir, tareas, ref.U, ref.Siempre)
	ctx := context.Background()
	cos := make([]float64, 0, n)
	lat := make([]float64, 0, n)
	var top1, escala, motivo int
	for i := 0; i < n; i++ {
		c := cs[i]
		t0 := time.Now()
		v, err := enc.Codificar(ctx, c.Texto)
		lat = append(lat, float64(time.Since(t0).Microseconds())/1000)
		if err != nil {
			t.Fatalf("%s: %v", c.ID, err)
		}
		if len(v) != dim {
			t.Fatalf("dim %d, fixture %d", len(v), dim)
		}
		var d float64
		for j := range v {
			d += float64(v[j]) * float64(x[i*dim+j])
		}
		cos = append(cos, d)
		dec, err := k.Evaluar(c.Tarea, v, c.Opciones)
		if err != nil {
			t.Fatal(err)
		}
		if dec.Answer == c.L2 {
			top1++
		}
		if dec.Escala == c.Escala {
			escala++
		}
		if dec.Motivo == motivoPy(c.Motivo) {
			motivo++
		}
	}
	if os.Getenv("CHISPA_E5_PEORES") != "" { // diagnóstico: las filas con peor coseno
		idx := make([]int, len(cos))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return cos[idx[a]] < cos[idx[b]] })
		for _, i := range idx[:min(25, len(idx))] {
			tx := []rune(cs[i].Texto)
			t.Logf("peor: %s cos %.5f letras %d tokens %d  %q", cs[i].ID, cos[i], len(tx), len(NuevoTokenizadorE5(tt).ConMaxLen(E5MaxTokBanco, 128).IDs(NormalizarMayusculas(cs[i].Texto))), string(tx[:min(70, len(tx))]))
		}
	}
	sort.Float64s(cos)
	sort.Float64s(lat)
	q := func(v []float64, p float64) float64 {
		return v[int(math.Min(float64(len(v)-1), math.Floor(p*float64(len(v)-1))))]
	}
	pc := func(a int) float64 { return 100 * float64(a) / float64(n) }
	t.Logf("n=%d  coseno Go/Python: mín %.6f  p01 %.6f  p05 %.6f  p50 %.6f", n, cos[0], q(cos, 0.01), q(cos, 0.05), q(cos, 0.5))
	t.Logf("kNN con el vector de Go frente a Python: top-1 %.2f %%  (criterio ≥ 99,5)  escala %.2f %%  motivo %.2f %%  (informativo, objetivo ≥ 99)", pc(top1), pc(escala), pc(motivo))
	t.Logf("latencia del codificador por consulta (lote 1, ms): p50 %.2f  p95 %.2f  p99 %.2f  máx %.2f", q(lat, 0.5), q(lat, 0.95), q(lat, 0.99), lat[len(lat)-1])
	if q(cos, 0.01) < 0.999 {
		t.Errorf("criterio: coseno p01 %.6f < 0,999", q(cos, 0.01))
	}
	if pc(top1) < 99.5 {
		t.Errorf("criterio: top-1 %.2f %% < 99,5 %%", pc(top1))
	}
}
