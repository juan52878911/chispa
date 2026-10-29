//go:build ort

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"chispa/runtime/tok"
	"github.com/juan52878911/chispa/server/decide"
)

func escribirBancoNpy(t *testing.T, base string, etiquetas []string, vs [][]float32) {
	t.Helper()
	cab := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", len(vs), len(vs[0]))
	for (10+len(cab)+1)%64 != 0 {
		cab += " "
	}
	cab += "\n"
	buf := append([]byte("\x93NUMPY\x01\x00"), 0, 0)
	binary.LittleEndian.PutUint16(buf[8:], uint16(len(cab)))
	buf = append(buf, cab...)
	var js string
	for i, v := range vs {
		for _, x := range v {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(x))
		}
		js += fmt.Sprintf("{\"etiqueta\": %q, \"es_oro\": true}\n", etiquetas[i])
	}
	if err := os.WriteFile(base+".npy", buf, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(base+".jsonl", []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
}

// El servidor con Qwen (modelo en miniatura de la arquitectura real): arranca por config, contesta con el kNN del banco construido con el MISMO codificador,
// y un banco de otra dimensión (e5, 8 en el fixture) se rechaza al arrancar.
func TestQwenServidorDesdeConfig(t *testing.T) {
	lib := os.Getenv("CHISPA_ORT_LIB")
	if lib == "" {
		t.Skip("falta CHISPA_ORT_LIB")
	}
	decide.RutaORT = lib
	bpe, _ := filepath.Abs(binQwen)
	onnx, _ := filepath.Abs("../../decide/testdata/qwen/qwen3emb-fp32.onnx")
	tk, err := tok.AbrirBPE(bpe)
	if err != nil {
		t.Fatal(err)
	}
	c, err := decide.NuevoCodificadorQwenORT(onnx, tk, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	textos := map[string]string{"a": "hola buenos días", "b": "mal servicio, nunca más", "c": "¿pueden cambiar la dirección?"}
	var et []string
	var vs [][]float32
	for _, e := range []string{"a", "b", "c"} {
		v, err := c.CodificarConsulta(context.Background(), "¿Qué es?", textos[e])
		if err != nil {
			t.Fatal(err)
		}
		et, vs = append(et, e), append(vs, v)
	}
	dir := t.TempDir()
	escribirBancoNpy(t, filepath.Join(dir, "banco-t"), et, vs)
	cfg := `{"knn": {"umbral": 0, "codificador": "qwen3-emb", "qwen_onnx": "` + onnx + `", "qwen_tok": "` + bpe + `", "vecinos": 8,
	  "tareas": {"t": {"banco": "` + filepath.Join(dir, "banco-t") + `", "preguntas": ["¿Qué es?"]}}}}`
	a, err := armarModelo(decide.NuevasMetricas(), escribeCfg(t, cfg), "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Cerrar()
	// con el texto de un ejemplo, el vecino más cercano es él mismo (similitud 1): contesta su etiqueta, sin escalar
	for e, txt := range textos {
		r := a.Cascada.Decidir(context.Background(), decide.Pregunta{ID: "q", Estado: txt, Texto: "¿Qué es?", Kind: "choice", Opciones: []decide.Opcion{{ID: "a"}, {ID: "b"}, {ID: "c"}}})
		if r.Escalate || r.Answer != e || r.Layer != "L2" {
			t.Fatalf("texto %q: %+v (se esperaba %q)", txt, r, e)
		}
	}
	// banco de otra dimensión
	banco8, _ := filepath.Abs("../../decide/testdata/knn/banco-b")
	mal := `{"knn": {"umbral": 0, "codificador": "qwen3-emb", "qwen_onnx": "` + onnx + `", "qwen_tok": "` + bpe + `", "tareas": {"b": {"banco": "` + banco8 + `", "preguntas": ["¿b?"]}}}}`
	if _, err := armarModelo(decide.NuevasMetricas(), escribeCfg(t, mal), "", ""); err == nil || !strings.Contains(err.Error(), "dimensión") {
		t.Fatalf("banco de otra dimensión: %v", err)
	}
}
