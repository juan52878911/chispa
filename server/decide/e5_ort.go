//go:build ort

package decide

import (
	"context"
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/juan52878911/chispa/server/torre"
)

// CodificadorE5 es el Codificador del kNN de L2 con el ONNX de multilingual-e5-small SIN ajustar (salida last_hidden_state, ver
// model/alt/exportar_e5_onnx.py) y ONNX Runtime. Hace lo que `model/l2/linea_base_knn_e5.py:vectores` con `model/l2/modelo.py:Codificador`:
// normalización de mayúsculas del mensaje, «query: » + texto, tokenizador de e5 (-tok) truncado a maxTok (256 en los bancos), media de los
// estados de token (con máscara) y norma L2. Una consulta por llamada (lote 1): sin relleno, así que no hay máscara que aplicar.
type CodificadorE5 struct {
	sem    chan struct{}
	s      *ort.DynamicAdvancedSession
	tok    Tokenizador
	maxTok int // truncado del tokenizador (256 en los bancos)
}

// NuevoCodificadorE5ORT abre el ONNX (fp32 o int8). tok debe truncar la consulta a la longitud con la que se calcularon los vectores del
// banco (TokenizadorE5.ConMaxLen(256, …)).
func NuevoCodificadorE5ORT(onnx []byte, tok Tokenizador) (*CodificadorE5, error) {
	if err := torre.IniciarORT(RutaORT); err != nil {
		return nil, err
	}
	s, err := ort.NewDynamicAdvancedSessionWithONNXData(onnx, []string{"input_ids", "attention_mask"}, []string{"last_hidden_state"}, nil)
	if err != nil {
		return nil, err
	}
	return &CodificadorE5{s: s, sem: make(chan struct{}, 1), tok: tok, maxTok: E5MaxTokBanco}, nil
}

// Close destruye la sesión (espera a que termine la consulta en curso).
func (c *CodificadorE5) Close() error {
	c.sem <- struct{}{}
	defer func() { <-c.sem }()
	if c.s == nil {
		return nil
	}
	err := c.s.Destroy()
	c.s = nil
	return err
}

// Codificar devuelve el vector e5 normalizado (384) del texto.
func (c *CodificadorE5) Codificar(ctx context.Context, texto string) ([]float32, error) {
	texto = NormalizarMayusculas(texto)
	ids := conEspacioFinal(texto, c.tok.IDs(texto), c.maxTok)
	n := int64(len(ids))
	if n == 0 {
		return nil, errors.New("e5: texto sin tokens")
	}
	mask := make([]int64, n)
	for i := range mask {
		mask[i] = 1
	}
	tid, err := ort.NewTensor(ort.NewShape(1, n), ids)
	if err != nil {
		return nil, err
	}
	defer tid.Destroy()
	tm, err := ort.NewTensor(ort.NewShape(1, n), mask)
	if err != nil {
		return nil, err
	}
	defer tm.Destroy()
	select {
	case c.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if c.s == nil {
		<-c.sem
		return nil, errors.New("e5: sesión cerrada")
	}
	salida := []ort.Value{nil}
	err = c.s.Run([]ort.Value{tid, tm}, salida)
	<-c.sem
	if err != nil {
		return nil, err
	}
	defer salida[0].Destroy()
	t, ok := salida[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("e5: salida de tipo %T", salida[0])
	}
	sh := t.GetShape() // (1, T, d)
	if len(sh) != 3 || sh[0] != 1 || sh[1] != n {
		return nil, fmt.Errorf("e5: forma de last_hidden_state %v para %d tokens", sh, n)
	}
	return mediaNormalizada(t.GetData(), int(sh[1]), int(sh[2])), nil
}
