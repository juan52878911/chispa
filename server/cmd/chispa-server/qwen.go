package main

import (
	"fmt"

	"chispa/runtime/tok"
	"github.com/juan52878911/chispa/server/decide"
)

// montarCodificadorQwen abre el tokenizador BPE y el ONNX de Qwen3-Embedding (knn.qwen_tok y knn.qwen_onnx). Errores claros si falta alguno de los dos.
func montarCodificadorQwen(k *decide.ConfigKNN) (*decide.CodificadorQwen, error) {
	t, err := tok.AbrirBPE(k.QwenTok)
	if err != nil {
		return nil, fmt.Errorf("knn.qwen_tok: %w", err)
	}
	c, err := decide.NuevoCodificadorQwenORT(k.QwenONNX, t, k.QwenSesiones)
	if err != nil {
		return nil, fmt.Errorf("knn.qwen_onnx: %w", err)
	}
	return c, nil
}

// vecinosKNN son los vecinos efectivos del kNN (0 en la config = el valor por defecto).
func vecinosKNN(k *decide.ConfigKNN) int {
	if k.Vecinos <= 0 {
		return decide.KNNVecinosPorDefecto
	}
	return k.Vecinos
}
