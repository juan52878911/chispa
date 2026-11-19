package decide

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
)

// codHash es un codificador falso determinista: el vector sale del sha256 del texto (normalizado a norma 1) y cuenta las llamadas.
type codHash struct {
	dim    int
	llamas atomic.Int64
}

func (c *codHash) Codificar(_ context.Context, t string) ([]float32, error) {
	c.llamas.Add(1)
	v := make([]float32, c.dim)
	var n float64
	for i := range v {
		h := sha256.Sum256([]byte(t + string(rune(i))))
		v[i] = float32(int32(binary.LittleEndian.Uint32(h[:4]))) / (1 << 31)
		n += float64(v[i]) * float64(v[i])
	}
	for i := range v {
		v[i] /= float32(math.Sqrt(n))
	}
	return v, nil
}

var reMsW1 = regexp.MustCompile(`"ms":[0-9.eE+-]+`)

func sinMsW1(s string) string { return reMsW1.ReplaceAllString(s, `"ms":0`) }

func postDecide(t *testing.T, s *Servidor, body string) (int, string) {
	t.Helper()
	rr := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/v1/decide", strings.NewReader(body))
	s.Handler().ServeHTTP(rr, r)
	return rr.Code, sinMsW1(strings.TrimSpace(rr.Body.String()))
}

func servidorKNNFalso(t *testing.T, cfg ConfigKNN, cod Codificador) (*Servidor, *L2KNN) {
	t.Helper()
	k, err := NuevoL2KNN(cfg, cod)
	if err != nil {
		t.Fatal(err)
	}
	casc, err := NuevaCascada([]Capa{{"L2", 0.1}}, []Decisor{k}, NuevasMetricas())
	if err != nil {
		t.Fatal(err)
	}
	return NuevoServidor(casc, NuevasMetricas(), 16), k
}

func TestW1Golden(t *testing.T) {
	cfg := ConfigKNN{Umbral: 0.0, Tareas: map[string]TareaKNN{"a": {Banco: "testdata/knn/banco-a", Preguntas: []string{"¿a?"}}}}
	b, _ := CargarBancoKNN("testdata/knn/banco-a")
	s, _ := servidorKNNFalso(t, cfg, &codHash{dim: b.Dim})
	_, got := postDecide(t, s, `{"state":"hola mundo","questions":[{"id":"q1","question":"¿a?","kind":"choice","options":[{"id":"x"},{"id":"y"}]},{"id":"q2","question":"¿a?","kind":"noul"}]}`)
	t.Logf("GOLDEN %s", got)
}
