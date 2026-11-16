package main

import (
	"fmt"
	"os"

	"chispa/runtime/tok"
	"github.com/juan52878911/chispa/server/decide"
)

// montarCodificadorE5 abre el ONNX de e5 (knn.e5_onnx) con el tokenizador -tok, truncado a los 256 tokens de los bancos del kNN.
func montarCodificadorE5(onnx, rutaTok string) (*decide.CodificadorE5, error) {
	tk, err := tok.Abrir(rutaTok)
	if err != nil {
		return nil, fmt.Errorf("knn.e5_onnx: tokenizador: %w", err)
	}
	b, err := os.ReadFile(onnx)
	if err != nil {
		return nil, fmt.Errorf("knn.e5_onnx: %w", err)
	}
	c, err := decide.NuevoCodificadorE5ORT(b, decide.NuevoTokenizadorE5(tk).ConMaxLen(decide.E5MaxTokBanco, tok.MaxTok))
	if err != nil {
		return nil, fmt.Errorf("knn.e5_onnx: %w", err)
	}
	return c, nil
}
