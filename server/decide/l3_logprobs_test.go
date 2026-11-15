package decide

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// lpFalso es un vLLM falso: la llamada principal (logprobs) devuelve top; la de relleno (prompt_logprobs) devuelve relleno[etiqueta].
type lpFalso struct {
	top      map[string]float64 // token → logprob del top 20 (los tokens pueden llevar espacio: " A")
	relleno  map[string]float64 // etiqueta → logprob por prompt_logprobs
	esperaP  time.Duration      // espera en la llamada principal
	esperaR  time.Duration      // espera en las de relleno
	cuerpos  []map[string]any
	mu       sync.Mutex
	enVuelo  atomic.Int64
	maxVuelo atomic.Int64
	fallos   atomic.Int64 // cuántas llamadas iniciales cortan la conexión
	llamadas atomic.Int64
}

func (f *lpFalso) arranca(t *testing.T) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.llamadas.Add(1)
		n := f.enVuelo.Add(1)
		defer f.enVuelo.Add(-1)
		for {
			m := f.maxVuelo.Load()
			if n <= m || f.maxVuelo.CompareAndSwap(m, n) {
				break
			}
		}
		if f.fallos.Add(-1) >= 0 {
			c, _, _ := w.(http.Hijacker).Hijack()
			c.Close()
			return
		}
		b, _ := io.ReadAll(r.Body)
		var c map[string]any
		json.Unmarshal(b, &c)
		f.mu.Lock()
		f.cuerpos = append(f.cuerpos, c)
		f.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if _, ok := c["prompt_logprobs"]; ok {
			time.Sleep(f.esperaR)
			msgs := c["messages"].([]any)
			cod := msgs[len(msgs)-1].(map[string]any)["content"].(string)
			lp, ok := f.relleno[cod]
			if !ok {
				fmt.Fprint(w, `{"choices":[{"message":{"content":""}}],"prompt_logprobs":[null,{"1":{"logprob":-1,"decoded_token":"zz"}}]}`)
				return
			}
			fmt.Fprintf(w, `{"choices":[{"message":{"content":""}}],"prompt_logprobs":[null,{"1":{"logprob":-0.5,"decoded_token":"<|im_start|>"}},{"2":{"logprob":%v,"decoded_token":%q}}]}`, lp, cod)
			return
		}
		time.Sleep(f.esperaP)
		var tops []map[string]any
		for tok, lp := range f.top {
			tops = append(tops, map[string]any{"token": tok, "logprob": lp})
		}
		out, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"logprobs": map[string]any{"content": []any{map[string]any{"token": "A", "top_logprobs": tops}}}}}})
		w.Write(out)
	}))
	t.Cleanup(s.Close)
	return s
}

func (f *lpFalso) cuerpo(i int) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.cuerpos[i]
}

func nuevoLP(t *testing.T, f *lpFalso, mod func(*ConfigL3)) *L3LLM {
	t.Helper()
	s := f.arranca(t)
	cfg := ConfigL3{URL: s.URL, Modelo: "qwen", Modo: L3ModoLogprobs}
	if mod != nil {
		mod(&cfg)
	}
	l, err := NuevoL3LLM(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func preguntaLP(n int) Pregunta {
	ops := make([]Opcion, n)
	for i := range ops {
		ops[i] = Opcion{ID: "op" + strconv.Itoa(i)}
	}
	return Pregunta{ID: "q", Texto: "¿cuál?", Kind: KindChoice, Opciones: ops, Estado: "hola"}
}

func softmaxTest(lps ...float64) []float64 {
	m := math.Inf(-1)
	for _, v := range lps {
		m = math.Max(m, v)
	}
	var s float64
	out := make([]float64, len(lps))
	for i, v := range lps {
		out[i] = math.Exp(v - m)
		s += out[i]
	}
	for i := range out {
		out[i] /= s
	}
	return out
}

func cerca(t *testing.T, got, want float64, que string) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, esperado %v", que, got, want)
	}
}

func TestL3LogprobsParidadConPython(t *testing.T) {
	b, err := os.ReadFile("testdata/l3_logprobs_paridad.json")
	if err != nil {
		t.Fatal(err)
	}
	var casos []struct {
		Fila struct {
			Tipo        string            `json:"tipo"`
			Pregunta    string            `json:"pregunta"`
			Opciones    []string          `json:"opciones"`
			Descripcion map[string]string `json:"descripciones"`
			Texto       string            `json:"texto"`
		} `json:"fila"`
		Mensajes  []msgL3  `json:"mensajes"`
		Etiquetas []string `json:"etiquetas"`
	}
	if err := json.Unmarshal(b, &casos); err != nil || len(casos) < 4 {
		t.Fatal(err, len(casos))
	}
	for _, c := range casos {
		p := Pregunta{Texto: c.Fila.Pregunta, Kind: c.Fila.Tipo, Estado: c.Fila.Texto}
		for _, o := range c.Fila.Opciones {
			p.Opciones = append(p.Opciones, Opcion{ID: o, Description: c.Fila.Descripcion[o]})
		}
		ord := p.Opciones
		if p.Kind == KindScore {
			ord, _ = opcionesNiveles(p.Opciones)
		}
		msgs, cods, err := mensajesLogprobsL3(p, ord)
		if err != nil {
			t.Fatal(err)
		}
		for i := range msgs {
			if msgs[i] != c.Mensajes[i] {
				t.Errorf("%s: mensaje %d difiere\n go: %q\n py: %q", c.Fila.Pregunta, i, msgs[i].Content, c.Mensajes[i].Content)
			}
		}
		if fmt.Sprint(cods) != fmt.Sprint(c.Etiquetas) {
			t.Errorf("%s: etiquetas %v, python %v", c.Fila.Pregunta, cods, c.Etiquetas)
		}
	}
}

func TestL3LogprobsSoftmaxYVariantesConEspacio(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -1, " A": -1.5, "B": -2, "C": -3, "x": -9}}
	l := nuevoLP(t, f, nil)
	r, err := l.Decidir(context.Background(), preguntaLP(3))
	if err != nil {
		t.Fatal(err)
	}
	a := math.Log(math.Exp(-1) + math.Exp(-1.5))
	want := softmaxTest(a, -2, -3)
	for i, id := range []string{"op0", "op1", "op2"} {
		cerca(t, r.Probs[id], want[i], id)
	}
	if r.Answer != "op0" || f.llamadas.Load() != 1 {
		t.Errorf("answer %q, llamadas %d", r.Answer, f.llamadas.Load())
	}
	c := f.cuerpo(0)
	if c["max_tokens"] != 1.0 || c["logprobs"] != true || c["top_logprobs"] != 20.0 || c["temperature"] != 0.0 {
		t.Errorf("cuerpo principal: %v", c)
	}
}

func TestL3LogprobsRellenoPorPromptLogprobsYPiso(t *testing.T) {
	// 6 opciones, A en el top; max_extra por defecto = 4: B,C,D,E por prefijo; F recibe el piso (mínimo conocido − 1).
	f := &lpFalso{top: map[string]float64{"A": -0.5, "z": -8}, relleno: map[string]float64{"B": -3, "C": -4, "D": -5, "E": -6}}
	l := nuevoLP(t, f, nil)
	r, err := l.Decidir(context.Background(), preguntaLP(6))
	if err != nil {
		t.Fatal(err)
	}
	if f.llamadas.Load() != 5 { // 1 principal + 4 rellenos
		t.Errorf("llamadas = %d, esperadas 5", f.llamadas.Load())
	}
	want := softmaxTest(-0.5, -3, -4, -5, -6, -8-1) // piso: mínimo conocido (el token z del top, −8) − 1
	for i := 0; i < 6; i++ {
		cerca(t, r.Probs["op"+strconv.Itoa(i)], want[i], "op"+strconv.Itoa(i))
	}
	c := f.cuerpo(1)
	if c["continue_final_message"] != true || c["add_generation_prompt"] != false || c["prompt_logprobs"] != 0.0 {
		t.Errorf("cuerpo de relleno: %v", c)
	}
	ms := c["messages"].([]any)
	if len(ms) != 3 || ms[2].(map[string]any)["role"] != "assistant" || ms[2].(map[string]any)["content"] != "B" {
		t.Errorf("la conversación de relleno debe terminar en el asistente con la etiqueta: %v", ms)
	}
}

func TestL3LogprobsMaxExtraCero(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"B": -1}}
	cero := 0
	l := nuevoLP(t, f, func(c *ConfigL3) { c.MaxExtra = &cero })
	r, err := l.Decidir(context.Background(), preguntaLP(3))
	if err != nil || f.llamadas.Load() != 1 {
		t.Fatal(err, f.llamadas.Load())
	}
	want := softmaxTest(-2, -1, -2) // piso = −1 − 1
	cerca(t, r.Probs["op1"], want[1], "op1")
	cerca(t, r.Probs["op0"], want[0], "op0")
	if r.Answer != "op1" {
		t.Error(r.Answer)
	}
}

func TestL3LogprobsSinEtiquetasEsInvalida(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"hola": -0.1, "x": -3}} // ni etiquetas en el top ni relleno que sirva
	l := nuevoLP(t, f, nil)
	r, err := l.Decidir(context.Background(), preguntaLP(3))
	if err != nil {
		t.Fatal(err)
	}
	if !r.Incompleto || r.Probs != nil || r.Conf != 0 {
		t.Errorf("debía ser inválida: %+v", r)
	}
}

func TestL3LogprobsMasDe62OpcionesEsInvalida(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -1}}
	l := nuevoLP(t, f, nil)
	r, err := l.Decidir(context.Background(), preguntaLP(63))
	if err != nil || !r.Incompleto || r.Probs != nil {
		t.Errorf("63 opciones: %+v %v", r, err)
	}
	if f.llamadas.Load() != 0 {
		t.Error("no debía llamar al LLM")
	}
	r, err = l.Decidir(context.Background(), preguntaLP(62))
	if err != nil || r.Incompleto || len(r.Probs) != 62 {
		t.Errorf("62 opciones: %v %v", len(r.Probs), err)
	}
}

func TestL3LogprobsEnableThinkingSoloSiSeConfigura(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -1, "B": -2}}
	l := nuevoLP(t, f, nil)
	if _, err := l.Decidir(context.Background(), preguntaLP(2)); err != nil {
		t.Fatal(err)
	}
	if _, ok := f.cuerpo(0)["chat_template_kwargs"]; ok {
		t.Error("sin configurar no debe mandarse chat_template_kwargs")
	}
	f2 := &lpFalso{top: map[string]float64{"A": -1}, relleno: map[string]float64{"B": -2}}
	l2 := nuevoLP(t, f2, func(c *ConfigL3) { c.SinPensar = true })
	if _, err := l2.Decidir(context.Background(), preguntaLP(2)); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ { // la principal y el relleno
		k, _ := f2.cuerpo(i)["chat_template_kwargs"].(map[string]any)
		if k["enable_thinking"] != false {
			t.Errorf("llamada %d: chat_template_kwargs = %v", i, k)
		}
	}
}

func TestL3LogprobsTopeDeTiempoSinNadaEsSinL3(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -1}, esperaP: 400 * time.Millisecond}
	l := nuevoLP(t, f, func(c *ConfigL3) { c.TimeoutFila = Duracion(80 * time.Millisecond) })
	t0 := time.Now()
	_, err := l.Decidir(context.Background(), preguntaLP(3))
	if !errors.Is(err, ErrSinL3) {
		t.Fatalf("esperaba ErrSinL3, llegó %v", err)
	}
	if time.Since(t0) > 300*time.Millisecond {
		t.Errorf("el tope no cortó: %v", time.Since(t0))
	}
}

func TestL3LogprobsTopeDeTiempoEnElRellenoDaParciales(t *testing.T) {
	// la principal es rápida; el relleno tarda más que el tope de la fila: se devuelve lo del top 20 con piso.
	f := &lpFalso{top: map[string]float64{"A": -0.2, "B": -1.2}, relleno: map[string]float64{"C": -2}, esperaR: 400 * time.Millisecond}
	l := nuevoLP(t, f, func(c *ConfigL3) { c.TimeoutFila = Duracion(120 * time.Millisecond) })
	r, err := l.Decidir(context.Background(), preguntaLP(3))
	if err != nil || r.Incompleto {
		t.Fatalf("esperaba parciales, llegó %+v %v", r, err)
	}
	want := softmaxTest(-0.2, -1.2, -1.2-1)
	cerca(t, r.Probs["op2"], want[2], "op2 con piso")
	cerca(t, r.Probs["op0"], want[0], "op0")
}

func TestL3LogprobsLimiteDeConcurrencia(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -1, "B": -2}, esperaP: 60 * time.Millisecond}
	l := nuevoLP(t, f, func(c *ConfigL3) { c.MaxConcurrente = 2 })
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := l.Decidir(context.Background(), preguntaLP(2)); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if m := f.maxVuelo.Load(); m != 2 {
		t.Errorf("máximo en vuelo = %d, esperado exactamente 2", m)
	}
}

func TestL3LogprobsReintentaUnaVezEnErrorDeRed(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -1, "B": -2}}
	f.fallos.Store(1)
	l := nuevoLP(t, f, nil)
	r, err := l.Decidir(context.Background(), preguntaLP(2))
	if err != nil || r.Incompleto || f.llamadas.Load() != 2 {
		t.Fatalf("1 fallo + 1 reintento: %+v %v llamadas=%d", r, err, f.llamadas.Load())
	}
	f2 := &lpFalso{top: map[string]float64{"A": -1}}
	f2.fallos.Store(5)
	l2 := nuevoLP(t, f2, nil)
	if _, err := l2.Decidir(context.Background(), preguntaLP(2)); !errors.Is(err, ErrSinL3) || f2.llamadas.Load() != 2 {
		t.Errorf("dos fallos: %v llamadas=%d (solo 1 reintento)", err, f2.llamadas.Load())
	}
}

func TestL3LogprobsScoreUsaElNivelComoEtiqueta(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"1": -3, "2": -1, "3": -0.5}}
	l := nuevoLP(t, f, nil)
	p := Pregunta{ID: "q", Texto: "nivel", Kind: KindScore, Estado: "x", Opciones: []Opcion{{ID: "3"}, {ID: "1"}, {ID: "2"}}}
	r, err := l.Decidir(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	want := softmaxTest(-3, -1, -0.5)
	cerca(t, r.Probs["1"], want[0], "nivel 1")
	cerca(t, r.Probs["3"], want[2], "nivel 3")
	if r.Answer != "2" { // mediana: 0,1+0,?… acumulado ≥ 0,5 en el nivel 2 o 3 según masa
		var acc float64
		exp := ""
		for i, id := range []string{"1", "2", "3"} {
			if acc += want[i]; acc >= 0.5 && exp == "" {
				exp = id
			}
		}
		if r.Answer != exp {
			t.Errorf("answer %q, esperado %q", r.Answer, exp)
		}
	}
}

func TestL3ModoPorDefectoSigueSiendoJSON(t *testing.T) {
	if err := (ConfigL3{URL: "http://127.0.0.1:1", Modelo: "m", Modo: "otro"}).validar(); err == nil {
		t.Error("modo desconocido debe rechazarse")
	}
	if err := (ConfigL3{URL: "http://127.0.0.1:1", Modelo: "m"}).validar(); err != nil {
		t.Error(err)
	}
}
