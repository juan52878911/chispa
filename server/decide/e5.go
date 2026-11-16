package decide

import (
	"errors"
	"math"
	"unicode"
)

// E5MaxTokBanco es el truncado (tokens, con <s> y </s>) con el que se calcularon los vectores de los bancos del kNN (Codificador(max_tok=256)).
const E5MaxTokBanco = 256

// ErrSinORTE5 lo da NuevoCodificadorE5ORT en un binario compilado sin la etiqueta ort.
var ErrSinORTE5 = errors.New("e5: binario compilado sin la etiqueta ort")

// idEspacioE5 es el id del token «▁» de XLM-R (sentencepiece) en e5.
const idEspacioE5 = 6

// conEspacioFinal reproduce una diferencia del tokenizador de Go (runtime/tok, que sigue a sentencepiece y descarta los espacios finales)
// con el rápido de HuggingFace con el que se calcularon los vectores de los bancos: este deja un token «▁» (id 6) antes de </s> cuando
// el texto acaba en espacio (hatecheck-es trae «No la soporto. » con espacio final: 11 tokens en HF frente a 10 en Go; el coseno del
// vector caía a 0,96). ids lleva <s> … </s>; si ya está en maxTok (truncado en HF antes del «▁») no se toca.
func conEspacioFinal(texto string, ids []int64, maxTok int) []int64 {
	if len(ids) < 3 || len(ids) >= maxTok {
		return ids
	}
	r := []rune(texto)
	if len(r) == 0 || !unicode.IsSpace(r[len(r)-1]) {
		return ids
	}
	out := append(append(make([]int64, 0, len(ids)+1), ids[:len(ids)-1]...), idEspacioE5)
	return append(out, ids[len(ids)-1])
}

// mediaNormalizada es `F.normalize((H * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)` para un lote de 1 sin relleno: la media de los T estados
// de token (T × d, fila a fila) y su norma L2 (eps 1e-12 como PyTorch).
func mediaNormalizada(h []float32, t, d int) []float32 {
	acc := make([]float64, d)
	for i := 0; i < t; i++ {
		row := h[i*d : (i+1)*d]
		for j, v := range row {
			acc[j] += float64(v)
		}
	}
	var n2 float64
	for j := range acc {
		acc[j] /= float64(max(t, 1))
		n2 += acc[j] * acc[j]
	}
	inv := 1 / math.Max(math.Sqrt(n2), 1e-12)
	out := make([]float32, d)
	for j := range acc {
		out[j] = float32(acc[j] * inv)
	}
	return out
}
