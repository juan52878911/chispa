//go:build !ort

package decide

import "chispa/runtime/tok"

// NuevoCodificadorQwenORT falla sin la etiqueta ort.
func NuevoCodificadorQwenORT(rutaONNX string, t *tok.BPE, sesiones int) (*CodificadorQwen, error) {
	return nil, ErrSinORTQwen
}
