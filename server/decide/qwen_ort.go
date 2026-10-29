//go:build ort

package decide

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"

	ort "github.com/yalue/onnxruntime_go"

	"chispa/runtime/tok"
	"github.com/juan52878911/chispa/server/torre"
)

// QwenSesionesPorDefecto: una sola sesión. Cada sesión ORT carga su copia de los pesos (≈ 2,4 GB en fp32, ≈ 1,2 GB en int8): en una máquina de 8 GB
// un pool como el de e5 no cabe. La única sesión usa todos los núcleos en cada consulta.
const QwenSesionesPorDefecto = 1

// ortQwen es el ejecutorQwen con ONNX Runtime: un pool de sesiones sobre el ONNX abierto POR RUTA (los pesos están en un fichero .data al lado: pasan de 2 GB).
type ortQwen struct {
	libres chan *ort.DynamicAdvancedSession
	todas  []*ort.DynamicAdvancedSession
	cerr   chan struct{}
}

// NuevoCodificadorQwenORT abre el ONNX de Qwen3-Embedding (fp32 o int8, de exportar_qwen3emb_onnx.py) con un pool de `sesiones` (≤ 0: QwenSesionesPorDefecto).
// Comprueba los metadatos del ONNX (chispa.pooling = last_token y chispa.max_len = QwenMaxTok) para no abrir por error otro modelo.
func NuevoCodificadorQwenORT(rutaONNX string, t *tok.BPE, sesiones int) (*CodificadorQwen, error) {
	if sesiones <= 0 {
		sesiones = QwenSesionesPorDefecto
	}
	if _, err := os.Stat(rutaONNX); err != nil {
		return nil, fmt.Errorf("qwen3-emb: ONNX: %w", err)
	}
	if err := torre.IniciarORT(RutaORT); err != nil {
		return nil, err
	}
	o := &ortQwen{libres: make(chan *ort.DynamicAdvancedSession, sesiones), cerr: make(chan struct{})}
	hilos := max(1, runtime.NumCPU()/sesiones)
	for i := 0; i < sesiones; i++ {
		so, err := ort.NewSessionOptions()
		if err == nil {
			err = so.SetIntraOpNumThreads(hilos)
		}
		var s *ort.DynamicAdvancedSession
		if err == nil {
			s, err = ort.NewDynamicAdvancedSession(rutaONNX, []string{"input_ids", "attention_mask"}, []string{"last_hidden_state"}, so)
		}
		if so != nil {
			so.Destroy()
		}
		if err == nil && i == 0 {
			err = comprobarMetadatosQwen(s)
		}
		if err != nil {
			if s != nil {
				s.Destroy()
			}
			o.Close()
			return nil, fmt.Errorf("qwen3-emb: %s: %w", rutaONNX, err)
		}
		o.todas = append(o.todas, s)
		o.libres <- s
	}
	return nuevoCodificadorQwen(t, o), nil
}

func comprobarMetadatosQwen(s *ort.DynamicAdvancedSession) error {
	m, err := s.GetModelMetadata()
	if err != nil {
		return err
	}
	defer m.Destroy()
	for _, p := range [][2]string{{"chispa.pooling", "last_token"}, {"chispa.max_len", strconv.Itoa(QwenMaxTok)}} { // en este orden: el error es siempre el mismo
		k, esperado := p[0], p[1]
		v, ok, err := m.LookupCustomMetadataMap(k)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("el ONNX no trae el metadato %s (¿es el de model/alt/exportar_qwen3emb_onnx.py?)", k)
		}
		if v != esperado {
			return fmt.Errorf("metadato %s = %q, se espera %q", k, v, esperado)
		}
	}
	return nil
}

func (o *ortQwen) Sesiones() int { return len(o.todas) }

func (o *ortQwen) Close() error {
	select {
	case <-o.cerr:
		return nil
	default:
		close(o.cerr)
	}
	var errs []error
	for range o.todas {
		errs = append(errs, (<-o.libres).Destroy()) // espera a que vuelva cada sesión
	}
	return errors.Join(errs...)
}

func (o *ortQwen) Correr(ctx context.Context, ids []int64) ([]float32, int, int, error) {
	n := int64(len(ids))
	mask := make([]int64, n)
	for i := range mask {
		mask[i] = 1
	}
	tid, err := ort.NewTensor(ort.NewShape(1, n), ids)
	if err != nil {
		return nil, 0, 0, err
	}
	defer tid.Destroy()
	tm, err := ort.NewTensor(ort.NewShape(1, n), mask)
	if err != nil {
		return nil, 0, 0, err
	}
	defer tm.Destroy()
	var s *ort.DynamicAdvancedSession
	select {
	case s = <-o.libres:
	case <-o.cerr:
		return nil, 0, 0, errors.New("qwen3-emb: codificador cerrado")
	case <-ctx.Done():
		return nil, 0, 0, ctx.Err()
	}
	salida := []ort.Value{nil}
	err = s.Run([]ort.Value{tid, tm}, salida)
	o.libres <- s
	if err != nil {
		return nil, 0, 0, err
	}
	defer salida[0].Destroy()
	t, ok := salida[0].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, 0, fmt.Errorf("qwen3-emb: salida de tipo %T", salida[0])
	}
	sh := t.GetShape() // (1, T, d)
	if len(sh) != 3 || sh[0] != 1 || sh[1] != n {
		return nil, 0, 0, fmt.Errorf("qwen3-emb: forma de last_hidden_state %v para %d tokens", sh, n)
	}
	h := append([]float32(nil), t.GetData()...) // el tensor se destruye al volver
	return h, int(sh[1]), int(sh[2]), nil
}
