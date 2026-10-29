//go:build ort

package decide

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"chispa/runtime/tok"
)

// Coseno del vector Go (ORT) con el de transformers fp32 sobre el fixture de 200 textos y el modelo REAL, y latencia del codificador (informativa).
// CHISPA_QWEN_ONNX, CHISPA_QWEN_TOK, CHISPA_QWEN_REF, CHISPA_ORT_LIB, CHISPA_QWEN_CONDICION (fp32|int8, para el nombre del JSON), [CHISPA_QWEN_SALIDA].
func TestQwenCosenoReal(t *testing.T) {
	onnx, rutaTok, ref, cond := os.Getenv("CHISPA_QWEN_ONNX"), os.Getenv("CHISPA_QWEN_TOK"), os.Getenv("CHISPA_QWEN_REF"), os.Getenv("CHISPA_QWEN_CONDICION")
	if onnx == "" || rutaTok == "" || ref == "" {
		t.Skip("falta CHISPA_QWEN_ONNX, CHISPA_QWEN_TOK o CHISPA_QWEN_REF")
	}
	if cond == "" {
		cond = "fp32"
	}
	if lib := os.Getenv("CHISPA_ORT_LIB"); lib != "" {
		RutaORT = lib
	}
	b, err := tok.AbrirBPE(rutaTok)
	if err != nil {
		t.Fatal(err)
	}
	c, err := NuevoCodificadorQwenORT(onnx, b, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	refIDs := leerRefIDsQwen(t, filepath.Join(ref, "ids.jsonl"))
	vref, n, dim, err := leerNpyF32(filepath.Join(ref, "vectores.npy"))
	if err != nil || n != len(refIDs) {
		t.Fatalf("vectores de referencia: %v (%d vectores, %d textos)", err, n, len(refIDs))
	}
	cs, ms := make([]float64, 0, n), make([]float64, 0, n)
	for i, r := range refIDs {
		// la entrada de referencia ya es la consulta completa: se codifica tal cual con la misma ruta que usa el servidor
		t0 := time.Now()
		v, err := c.codificar(context.Background(), r.Entrada)
		ms = append(ms, float64(time.Since(t0).Microseconds())/1000)
		if err != nil {
			t.Fatalf("%s: %v", r.ID, err)
		}
		cs = append(cs, cosenoQwen(v, vref[i*dim:(i+1)*dim]))
	}
	orden := append([]float64(nil), cs...)
	sort.Float64s(orden)
	lat := append([]float64(nil), ms...)
	sort.Float64s(lat)
	bajo := 0
	for _, x := range cs {
		if x < 0.999 {
			bajo++
		}
	}
	tokens := compararTokensQwen(t, b, refIDs)
	res := map[string]any{"condicion": cond, "onnx": filepath.Base(onnx), "n": n, "dim": dim, "tokens": tokens,
		"coseno":                  map[string]any{"min": orden[0], "p01": orden[len(orden)/100], "p50": orden[len(orden)/2], "n_bajo_0.999": bajo, "pasa": bajo == 0},
		"latencia_codificador_ms": map[string]any{"p50": lat[len(lat)/2], "p95": lat[len(lat)*95/100], "max": lat[len(lat)-1]}}
	escribirJSONQwen(t, "paridad-"+cond+".json", res)
	t.Logf("%s: coseno mín %.6f p50 %.6f (%d bajo 0,999); tokens %d/%d; latencia p50 %.0f ms", cond, orden[0], orden[len(orden)/2], bajo, tokens.Iguales, tokens.N, lat[len(lat)/2])
	if cond == "fp32" && (bajo != 0 || !tokens.Veredicto) {
		t.Fatalf("criterio: %d textos con coseno < 0,999; tokens %d/%d", bajo, tokens.Iguales, tokens.N)
	}
}
