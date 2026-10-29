package decide

import (
	"context"
	"errors"
	"fmt"
	"math"

	"chispa/runtime/tok"
)

// Codificador Qwen3-Embedding-0.6B para el kNN de L2 (alternativa a e5-small, config `knn.codificador: "qwen3-emb"`). Reproduce lo que midió R11
// (model/alt/alternativas_r11.py, EmbQwen y consulta_emb) y nada más:
//   - revisión 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 del modelo; ONNX de model/alt/exportar_qwen3emb_onnx.py (salida last_hidden_state);
//   - consulta de soporte y de test: «Instruct: Classify the text to answer this question: <pregunta>\nQuery:<texto>»; las descripciones de etiqueta, sin instrucción;
//   - tokenizador BPE byte-level de transformers (runtime/tok/bpe.go) truncado a 320 tokens, con los especiales que ponga su plantilla;
//   - pooling del ÚLTIMO token (con relleno a la izquierda en lote la última posición es siempre la del último token; aquí el lote es de 1, sin relleno) y norma L2.
// A diferencia de e5, NO se normalizan las mayúsculas del mensaje (R11 no lo hacía con Qwen).

const (
	// QwenMaxTok es el truncado (tokens, con el especial final de la plantilla) de los bancos y de las consultas de Qwen (MAX_TOK_EMB de R11).
	QwenMaxTok = 320
	// QwenRevision es la revisión de Qwen/Qwen3-Embedding-0.6B medida en R11.
	QwenRevision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
	// QwenInstruccion es INSTR_EMB de R11, con la pregunta al final.
	QwenInstruccion = "Classify the text to answer this question: "
)

// Nombres del codificador del kNN (`knn.codificador`).
const (
	CodificadorE5Small  = "e5-small"
	CodificadorQwen3Emb = "qwen3-emb"
)

// ErrSinORTQwen lo da NuevoCodificadorQwenORT en un binario compilado sin la etiqueta ort.
var ErrSinORTQwen = errors.New("qwen3-emb: binario compilado sin la etiqueta ort")

// QwenConsulta es la entrada del modelo para una consulta (soporte o test): `consulta_emb` de R11.
func QwenConsulta(pregunta, texto string) string {
	return "Instruct: " + QwenInstruccion + pregunta + "\nQuery:" + texto
}

// CodificadorConsulta es un Codificador que necesita además el texto de la pregunta (Qwen3-Embedding lleva la instrucción en la consulta). L2KNN lo usa
// en lugar de Codificar cuando el codificador lo implementa.
type CodificadorConsulta interface {
	Codificador
	CodificarConsulta(ctx context.Context, pregunta, texto string) ([]float32, error)
}

// ejecutorQwen corre el ONNX: input_ids (1×T) → last_hidden_state (T×d, fila a fila). Lo da el binario con ort; los tests ponen uno falso.
type ejecutorQwen interface {
	Correr(ctx context.Context, ids []int64) (h []float32, t, d int, err error)
	Sesiones() int
	Close() error
}

// CodificadorQwen: ver el comentario del fichero. Es seguro para uso concurrente si lo es el ejecutor.
type CodificadorQwen struct {
	tok *tok.BPE
	ej  ejecutorQwen
}

func nuevoCodificadorQwen(t *tok.BPE, ej ejecutorQwen) *CodificadorQwen {
	return &CodificadorQwen{tok: t, ej: ej}
}

// Sesiones es el tamaño del pool de sesiones ORT.
func (c *CodificadorQwen) Sesiones() int { return c.ej.Sesiones() }

// Close destruye las sesiones.
func (c *CodificadorQwen) Close() error { return c.ej.Close() }

// Codificar da el vector del texto SIN instrucción (descripciones de etiqueta).
func (c *CodificadorQwen) Codificar(ctx context.Context, texto string) ([]float32, error) {
	return c.codificar(ctx, texto)
}

// CodificarConsulta da el vector de una consulta con la instrucción de R11.
func (c *CodificadorQwen) CodificarConsulta(ctx context.Context, pregunta, texto string) ([]float32, error) {
	return c.codificar(ctx, QwenConsulta(pregunta, texto))
}

func (c *CodificadorQwen) codificar(ctx context.Context, entrada string) ([]float32, error) {
	i32 := c.tok.IDs(entrada, QwenMaxTok)
	if len(i32) == 0 {
		return nil, errors.New("qwen3-emb: texto sin tokens")
	}
	ids := make([]int64, len(i32))
	for i, v := range i32 {
		ids[i] = int64(v)
	}
	h, t, d, err := c.ej.Correr(ctx, ids)
	if err != nil {
		return nil, err
	}
	if t != len(ids) || d <= 0 || len(h) != t*d {
		return nil, fmt.Errorf("qwen3-emb: last_hidden_state de %d valores para %d tokens (d=%d)", len(h), len(ids), d)
	}
	return ultimoTokenNormalizado(h, t, d), nil
}

// ultimoTokenNormalizado es `F.normalize(last_hidden_state[:, -1], dim=-1)` de EmbQwen para un lote de 1 (sin relleno): la fila de la última
// posición (T × d, fila a fila) con norma L2 (eps 1e-12 como PyTorch). Con relleno a la izquierda en lote la última posición también es la del último token.
func ultimoTokenNormalizado(h []float32, t, d int) []float32 {
	fila := h[(t-1)*d : t*d]
	var n2 float64
	for _, v := range fila {
		n2 += float64(v) * float64(v)
	}
	inv := 1 / math.Max(math.Sqrt(n2), 1e-12)
	out := make([]float32, d)
	for j, v := range fila {
		out[j] = float32(float64(v) * inv)
	}
	return out
}
