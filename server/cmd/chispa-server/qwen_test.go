package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/juan52878911/chispa/server/decide"
)

const binQwen = "../../../runtime/tok/testdata/qwen/mini.chispa-bpe"

// knn.codificador = "qwen3-emb": faltan rutas → error de config que las nombra; con rutas que no existen, el error dice cuál (tokenizador u ONNX).
func TestQwenDesdeConfigErroresClaros(t *testing.T) {
	banco, _ := filepath.Abs("../../decide/testdata/knn/banco-b")
	bpe, _ := filepath.Abs(binQwen)
	cfg := func(extra string) string {
		return `{"knn": {"umbral": 0, "codificador": "qwen3-emb", ` + extra + `"tareas": {"b": {"banco": "` + banco + `", "preguntas": ["¿b?"]}}}}`
	}
	if _, err := armarModelo(decide.NuevasMetricas(), escribeCfg(t, cfg(`"qwen_onnx": "/x.onnx", `)), "", ""); err == nil || !strings.Contains(err.Error(), "qwen_tok") {
		t.Fatalf("falta qwen_tok: %v", err)
	}
	if _, err := armarModelo(decide.NuevasMetricas(), escribeCfg(t, cfg(`"qwen_onnx": "/x.onnx", "qwen_tok": "/no/existe.chispa-bpe", `)), "", ""); err == nil || !strings.Contains(err.Error(), "knn.qwen_tok") {
		t.Fatalf("tokenizador inexistente: %v", err)
	}
	if _, err := armarModelo(decide.NuevasMetricas(), escribeCfg(t, cfg(`"qwen_onnx": "/no/existe.onnx", "qwen_tok": "`+bpe+`", `)), "", ""); err == nil || !strings.Contains(err.Error(), "knn.qwen_onnx") {
		t.Fatalf("ONNX inexistente: %v", err)
	}
}
