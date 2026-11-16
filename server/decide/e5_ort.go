//go:build ort

package decide

import (
	"context"
	"errors"
	"fmt"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/juan52878911/chispa/server/torre"
)

// CodificadorE5 es el Codificador del kNN de L2 con el ONNX de multilingual-e5-small SIN ajustar (salida last_hidden_state, ver
// model/alt/exportar_e5_onnx.py) y ONNX Runtime. Hace lo que `model/l2/linea_base_knn_e5.py:vectores` con `model/l2/modelo.py:Codificador`:
// normalización de mayúsculas del mensaje, «query: » + texto, tokenizador de e5 (-tok) truncado a maxTok (256 en los bancos), media de los
// estados de token (con máscara) y norma L2. Una consulta por llamada (lote 1): sin relleno, así que no hay máscara que aplicar.
//
// Concurrencia: un POOL de sesiones ORT (knn.e5_sesiones; por defecto min(4, núcleos)). Una sola sesión serializada era el cuello de botella
// con 16 clientes (e5 p50 202 ms, p95 1,3 s frente a 8,6 ms en serie): cada consulta espera su turno de sesión, y con N sesiones corren N a la
// vez. Cada sesión carga su copia de los pesos (≈ 470 MB en fp32) y usa núcleos/N hilos intra-operador, para no sobresuscribir la CPU.
type CodificadorE5 struct {
	libres chan *ort.DynamicAdvancedSession // sesiones libres; vacío cuando todas están en uso
	todas  []*ort.DynamicAdvancedSession
	cerr   chan struct{} // se cierra en Close
	tok    Tokenizador
	maxTok int // truncado del tokenizador (256 en los bancos)
}

// E5SesionesPorDefecto es el tamaño del pool sin configurar: min(4, núcleos).
func E5SesionesPorDefecto() int { return max(1, min(4, runtime.NumCPU())) }

// NuevoCodificadorE5ORT abre el ONNX (fp32 o int8) con el pool por defecto. tok debe truncar la consulta a la longitud con la que se calcularon
// los vectores del banco (TokenizadorE5.ConMaxLen(256, …)).
func NuevoCodificadorE5ORT(onnx []byte, tok Tokenizador) (*CodificadorE5, error) {
	return NuevoCodificadorE5ORTN(onnx, tok, 0)
}

// NuevoCodificadorE5ORTN abre un pool de n sesiones (n ≤ 0 = E5SesionesPorDefecto).
func NuevoCodificadorE5ORTN(onnx []byte, tok Tokenizador, n int) (*CodificadorE5, error) {
	if n <= 0 {
		n = E5SesionesPorDefecto()
	}
	if err := torre.IniciarORT(RutaORT); err != nil {
		return nil, err
	}
	c := &CodificadorE5{libres: make(chan *ort.DynamicAdvancedSession, n), cerr: make(chan struct{}), tok: tok, maxTok: E5MaxTokBanco}
	hilos := max(1, runtime.NumCPU()/n)
	for i := 0; i < n; i++ {
		so, err := ort.NewSessionOptions()
		if err == nil {
			err = so.SetIntraOpNumThreads(hilos)
		}
		var s *ort.DynamicAdvancedSession
		if err == nil {
			s, err = ort.NewDynamicAdvancedSessionWithONNXData(onnx, []string{"input_ids", "attention_mask"}, []string{"last_hidden_state"}, so)
		}
		if so != nil {
			so.Destroy()
		}
		if err != nil {
			for _, x := range c.todas {
				x.Destroy()
			}
			return nil, err
		}
		c.todas = append(c.todas, s)
		c.libres <- s
	}
	return c, nil
}

// Sesiones es el tamaño del pool.
func (c *CodificadorE5) Sesiones() int { return len(c.todas) }

// Close destruye las sesiones (espera a que terminen las consultas en curso).
func (c *CodificadorE5) Close() error {
	select {
	case <-c.cerr:
		return nil
	default:
		close(c.cerr)
	}
	var errs []error
	for range c.todas {
		s := <-c.libres // espera a que vuelva cada sesión
		errs = append(errs, s.Destroy())
	}
	return errors.Join(errs...)
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
	var s *ort.DynamicAdvancedSession
	select {
	case s = <-c.libres:
	case <-c.cerr:
		return nil, errors.New("e5: codificador cerrado")
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	salida := []ort.Value{nil}
	err = s.Run([]ort.Value{tid, tm}, salida)
	c.libres <- s
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
