//go:build !ort

package decide

import "context"

// CodificadorE5 no existe sin ONNX Runtime (compila con -tags ort); este tipo solo mantiene la API.
type CodificadorE5 struct{}

// NuevoCodificadorE5ORT falla sin la etiqueta ort.
func NuevoCodificadorE5ORT(onnx []byte, tok Tokenizador) (*CodificadorE5, error) {
	return nil, ErrSinORTE5
}

func (c *CodificadorE5) Close() error { return nil }

func (c *CodificadorE5) Codificar(ctx context.Context, texto string) ([]float32, error) {
	return nil, ErrSinORTE5
}
