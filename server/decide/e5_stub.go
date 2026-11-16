//go:build !ort

package decide

import (
	"context"
	"runtime"
)

// CodificadorE5 no existe sin ONNX Runtime (compila con -tags ort); este tipo solo mantiene la API.
type CodificadorE5 struct{}

// NuevoCodificadorE5ORT falla sin la etiqueta ort.
func NuevoCodificadorE5ORT(onnx []byte, tok Tokenizador) (*CodificadorE5, error) {
	return nil, ErrSinORTE5
}

// NuevoCodificadorE5ORTN falla sin la etiqueta ort.
func NuevoCodificadorE5ORTN(onnx []byte, tok Tokenizador, n int) (*CodificadorE5, error) {
	return nil, ErrSinORTE5
}

// E5SesionesPorDefecto: min(4, núcleos), como en el binario con ort.
func E5SesionesPorDefecto() int { return max(1, min(4, runtime.NumCPU())) }

func (c *CodificadorE5) Sesiones() int { return 0 }

func (c *CodificadorE5) Close() error { return nil }

func (c *CodificadorE5) Codificar(ctx context.Context, texto string) ([]float32, error) {
	return nil, ErrSinORTE5
}
