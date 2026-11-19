package decide

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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

const goldenSinExamples = `{"results":[{"id":"q1","kind":"choice","answer":"y","probs":{"x":0.030579183842907347,"y":0.9694208161570926},"layer":"L2","escalate":false,"ms":0},{"id":"q2","kind":"noul","answer":"false","probs":{"false":0.5,"true":0.5},"layer":"L2","escalate":false,"ms":0}]}`

func cfgW1(extra func(*ConfigKNN)) ConfigKNN {
	c := ConfigKNN{Umbral: 0.0, Tareas: map[string]TareaKNN{"a": {Banco: "testdata/knn/banco-a", Preguntas: []string{"¿a?"}}}}
	if extra != nil {
		extra(&c)
	}
	return c
}

func dimBancoA(t *testing.T) int {
	b, err := CargarBancoKNN("testdata/knn/banco-a")
	if err != nil {
		t.Fatal(err)
	}
	return b.Dim
}

// Una petición sin examples responde igual que antes de existir el campo (bytes, salvo ms), con el banco de la tarea.
func TestW1SinExamplesRespondeComoAntes(t *testing.T) {
	s, _ := servidorKNNFalso(t, cfgW1(nil), &codHash{dim: dimBancoA(t)})
	code, got := postDecide(t, s, `{"state":"hola mundo","questions":[{"id":"q1","question":"¿a?","kind":"choice","options":[{"id":"x"},{"id":"y"}]},{"id":"q2","question":"¿a?","kind":"noul"}]}`)
	if code != 200 || got != goldenSinExamples {
		t.Fatalf("%d\n got %s\nwant %s", code, got, goldenSinExamples)
	}
	// `examples` vacío o ausente es lo mismo
	_, got2 := postDecide(t, s, `{"state":"hola mundo","questions":[{"id":"q1","question":"¿a?","kind":"choice","options":[{"id":"x"},{"id":"y"}],"examples":[]},{"id":"q2","question":"¿a?","kind":"noul"}]}`)
	if got2 != goldenSinExamples {
		t.Fatalf("examples vacío: %s", got2)
	}
}

func cuerpoEj(state, kind, opts, ejs string) string {
	return `{"state":` + strconvQ(state) + `,"questions":[{"id":"q","question":"¿libre?","kind":"` + kind + `","options":` + opts + `,"examples":` + ejs + `}]}`
}

func strconvQ(s string) string { return `"` + s + `"` }

const opsXY = `[{"id":"x"},{"id":"y"}]`

func TestW1ExamplesSustituyenAlBancoYNoNecesitanTarea(t *testing.T) {
	cod := &codHash{dim: dimBancoA(t)}
	s, _ := servidorKNNFalso(t, cfgW1(nil), cod)
	ejs := `[{"text":"alfa","label":"x"},{"text":"alfa dos","label":"x"},{"text":"beta","label":"y"},{"text":"beta dos","label":"y"}]`
	// pregunta que NO está en knn.tareas: contesta L2 con los ejemplos de la petición
	for texto, esp := range map[string]string{"alfa": "x", "beta": "y"} {
		code, got := postDecide(t, s, cuerpoEj(texto, "choice", opsXY, ejs))
		if code != 200 || !strings.Contains(got, `"answer":"`+esp+`"`) || !strings.Contains(got, `"layer":"L2"`) {
			t.Fatalf("%q: %d %s", texto, code, got)
		}
	}
	// pregunta de una tarea con banco: con examples sustituyen al banco (el banco daba y para "hola mundo"; aquí el ejemplo idéntico dice x)
	body := `{"state":"hola mundo","questions":[{"id":"q1","question":"¿a?","kind":"choice","options":[{"id":"x"},{"id":"y"}],"examples":[{"text":"hola mundo","label":"x"},{"text":"adiós","label":"y"}]}]}`
	if _, got := postDecide(t, s, body); !strings.Contains(got, `"answer":"x"`) {
		t.Fatalf("los examples deben sustituir al banco: %s", got)
	}
}

func TestW1ExamplesValidacion(t *testing.T) {
	s, _ := servidorKNNFalso(t, cfgW1(nil), &codHash{dim: dimBancoA(t)})
	grande := strings.Repeat("a", MaxEjemploTextoByte+1)
	var muchos strings.Builder
	muchos.WriteString("[")
	for i := 0; i <= MaxEjemplosPregunta; i++ {
		if i > 0 {
			muchos.WriteString(",")
		}
		muchos.WriteString(`{"text":"t","label":"x"}`)
	}
	muchos.WriteString("]")
	for _, c := range []struct {
		nombre, body, code string
		status             int
	}{
		{"etiqueta fuera de opciones", cuerpoEj("s", "choice", opsXY, `[{"text":"a","label":"z"}]`), ErrEjemploEtiqueta, 400},
		{"texto vacío", cuerpoEj("s", "choice", opsXY, `[{"text":"  ","label":"x"}]`), ErrEjemploInvalido, 400},
		{"sin etiqueta", cuerpoEj("s", "choice", opsXY, `[{"text":"a"}]`), ErrEjemploInvalido, 400},
		{"texto largo", cuerpoEj("s", "choice", opsXY, `[{"text":"`+grande+`","label":"x"}]`), ErrCampoLargo, 400},
		{"demasiados", cuerpoEj("s", "choice", opsXY, muchos.String()), ErrDemasiadosEjem, 400},
		{"campo desconocido", cuerpoEj("s", "choice", opsXY, `[{"text":"a","label":"x","peso":1}]`), ErrJSONInvalido, 400},
		{"campo repetido", cuerpoEj("s", "choice", opsXY, `[{"text":"a","text":"b","label":"x"}]`), ErrJSONInvalido, 400},
		{"multi", cuerpoEj("s", "multi", opsXY, `[{"text":"a","label":"x"}]`), ErrEjemploInvalido, 400},
		{"no es una lista", cuerpoEj("s", "choice", opsXY, `{"text":"a"}`), ErrJSONInvalido, 400},
	} {
		code, got := postDecide(t, s, c.body)
		if code != c.status || !strings.Contains(got, `"`+c.code+`"`) {
			t.Errorf("%s: %d %s (quería %s)", c.nombre, code, got, c.code)
		}
	}
	// el mensaje dice qué ejemplo y qué opciones
	_, got := postDecide(t, s, cuerpoEj("s", "choice", opsXY, `[{"text":"a","label":"x"},{"text":"b","label":"z"}]`))
	if !strings.Contains(got, "examples[1].label") || !strings.Contains(got, `"question_id":"q"`) {
		t.Errorf("mensaje poco claro: %s", got)
	}
	// noul: etiquetas true/false
	if code, got := postDecide(t, s, cuerpoEj("s", "noul", `[]`, `[{"text":"a","label":"true"},{"text":"b","label":"false"}]`)); code != 200 {
		t.Errorf("noul con examples: %d %s", code, got)
	}
}

func TestW1CacheDeVectoresYDesalojo(t *testing.T) {
	cod := &codHash{dim: dimBancoA(t)}
	s, k := servidorKNNFalso(t, cfgW1(nil), cod)
	ejs := `[{"text":"alfa","label":"x"},{"text":"beta","label":"y"},{"text":"gamma","label":"y"}]`
	postDecide(t, s, cuerpoEj("alfa", "choice", opsXY, ejs))
	if n := cod.llamas.Load(); n != 4 { // consulta + 3 ejemplos
		t.Fatalf("primera petición: %d codificaciones", n)
	}
	postDecide(t, s, cuerpoEj("alfa", "choice", opsXY, ejs))
	if n := cod.llamas.Load(); n != 5 { // solo la consulta: los ejemplos salen de la caché
		t.Fatalf("segunda petición: %d codificaciones (los ejemplos debían venir de la caché)", n)
	}
	// la clave lleva el codificador y la revisión: otro texto preparado es otra clave, el mismo texto con otra etiqueta es el mismo vector
	if k.claveVector("¿p?", "t") == k.claveVector("¿p?", "u") {
		t.Fatal("textos distintos, misma clave")
	}
	if !strings.HasSuffix(k.claveVector("¿p?", "t"), "|"+CodificadorE5Small+"|"+E5Revision) {
		t.Fatalf("clave sin codificador+revisión: %s", k.claveVector("¿p?", "t"))
	}
	// tope de memoria: con sitio para ~2 vectores de cada fragmento no se acumulan más de lo que cabe
	c := &cacheVectores{lru: NewLRUBytes(fragmentos*(4*64+sobreEntrada+80)*2, func(v []float32) int { return 4 * len(v) }), en: map[string]*vueloVector{}}
	for i := 0; i < 2000; i++ {
		c.da(context.Background(), "k"+strconv.Itoa(i), func() ([]float32, error) { return make([]float32, 64), nil })
	}
	if b := c.lru.Bytes(); b > int64(fragmentos*(4*64+sobreEntrada+80)*2) || c.lru.Len() >= 2000 || c.lru.Len() == 0 {
		t.Fatalf("la caché no respeta el tope: %d bytes, %d entradas", b, c.lru.Len())
	}
	llamas := 0
	c.da(context.Background(), "k1999", func() ([]float32, error) { llamas++; return make([]float32, 64), nil })
	if llamas != 0 {
		t.Fatal("la última entrada debía seguir en la caché (LRU)")
	}
	c.da(context.Background(), "k0", func() ([]float32, error) { llamas++; return make([]float32, 64), nil })
	if llamas != 1 {
		t.Fatal("la primera debía haberse desalojado")
	}
}

func TestW1CacheDeduplicaCodificacionesSimultaneas(t *testing.T) {
	c := nuevaCacheVectores(1)
	var n atomic.Int64
	puerta := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.da(context.Background(), "k", func() ([]float32, error) { n.Add(1); <-puerta; return []float32{1}, nil })
		}()
	}
	time.Sleep(50 * time.Millisecond)
	close(puerta)
	wg.Wait()
	if n.Load() != 1 {
		t.Fatalf("%d codificaciones de la misma clave", n.Load())
	}
}

func TestW1NonceNoMezclaListasDeExamples(t *testing.T) {
	s, _ := servidorKNNFalso(t, cfgW1(nil), &codHash{dim: dimBancoA(t)})
	mk := func(label string) string {
		return `{"state":"alfa","questions":[{"id":"q","question":"¿libre?","kind":"choice","nonce":"n1","options":` + opsXY + `,"examples":[{"text":"alfa","label":"` + label + `"},{"text":"otro","label":"` + map[string]string{"x": "y", "y": "x"}[label] + `"}]}]}`
	}
	_, a := postDecide(t, s, mk("x"))
	_, b := postDecide(t, s, mk("y"))
	if !strings.Contains(a, `"answer":"x"`) || !strings.Contains(b, `"answer":"y"`) {
		t.Fatalf("la caché por nonce devolvió la respuesta de otros examples:\n%s\n%s", a, b)
	}
}

func TestW1ModoCPU(t *testing.T) {
	cod := &codHash{dim: dimBancoA(t)}
	s, _ := servidorKNNFalso(t, cfgW1(func(c *ConfigKNN) { c.Modo = ModoKNNCPU; c.Umbral = 0.99; c.Siempre = []string{"a"} }), cod)
	// sin escalado: contesta L2 aunque el margen no llegue a u y la tarea esté en «siempre»
	_, got := postDecide(t, s, `{"state":"hola mundo","questions":[{"id":"q1","question":"¿a?","kind":"choice","options":[{"id":"x"},{"id":"y"}]}]}`)
	if !strings.Contains(got, `"layer":"L2"`) || !strings.Contains(got, `"escalate":false`) {
		t.Fatalf("modo cpu debe contestar L2: %s", got)
	}
	// una pregunta sin banco y sin examples se rechaza con un error claro
	code, got := postDecide(t, s, `{"state":"x","questions":[{"id":"q9","question":"¿otra?","kind":"choice","options":[{"id":"x"},{"id":"y"}]}]}`)
	if code != 400 || !strings.Contains(got, ErrEjemplosFalta) || !strings.Contains(got, `"question_id":"q9"`) {
		t.Fatalf("sin examples ni banco: %d %s", code, got)
	}
	// y con examples contesta
	if code, got := postDecide(t, s, cuerpoEj("alfa", "choice", opsXY, `[{"text":"alfa","label":"x"},{"text":"beta","label":"y"}]`)); code != 200 || !strings.Contains(got, `"layer":"L2"`) {
		t.Fatalf("con examples: %d %s", code, got)
	}
}

func TestW1ModoGPUEscalaTodoYNoCodifica(t *testing.T) {
	cod := &codHash{dim: dimBancoA(t)}
	cfg := cfgW1(func(c *ConfigKNN) { c.Modo = ModoKNNGPU })
	k, err := NuevoL2KNN(cfg, cod)
	if err != nil {
		t.Fatal(err)
	}
	l3 := &decisorContadorL3{nombre: "L3", r: Respuesta{Answer: "x", Probs: map[string]float64{"x": 0.9, "y": 0.1}, Conf: 0.9}}
	casc, _ := NuevaCascada([]Capa{{"L2", 0.1}, {"L3", 0.5}}, []Decisor{k, l3}, NuevasMetricas())
	p := Pregunta{ID: "q", Estado: "hola", Texto: "¿a?", Kind: KindChoice, Opciones: []Opcion{{ID: "x"}, {ID: "y"}}}
	if r := casc.Decidir(context.Background(), p); r.Layer != "L3" || l3.n != 1 || cod.llamas.Load() != 0 {
		t.Fatalf("modo gpu: %+v, L3 %d veces, %d codificaciones", r, l3.n, cod.llamas.Load())
	}
	// con más de 20 opciones el kNN sí corre (top-k de L3) y devuelve el desempate por centroide
	var ops []Opcion
	for i := 0; i < L3TopKKNNDesde+3; i++ {
		ops = append(ops, Opcion{ID: strconv.Itoa(i)})
	}
	p.Opciones = ops
	p.Ejemplos = []EjemploPeticion{{"hola", "3"}, {"adiós", "4"}}
	r, err := k.Decidir(context.Background(), p)
	if err != nil || r.MotivoKNN != MotivoKNNSiempre || len(r.RankKNN) != len(ops) || cod.llamas.Load() == 0 {
		t.Fatalf("top-k en modo gpu: %+v %v", r, err)
	}
}

func TestW1ConfigValidaciones(t *testing.T) {
	for _, c := range []ConfigKNN{
		{Tareas: map[string]TareaKNN{"a": {Banco: "x"}}, Modo: "raro"},
		{Tareas: map[string]TareaKNN{"a": {Banco: "x"}}, CacheVectoresMB: -1},
		{Modo: ModoKNNCPU}, // sin tareas y sin codificador en el servidor
	} {
		if err := c.validar(); err == nil {
			t.Errorf("debía fallar: %+v", c)
		}
	}
	// sin tareas pero con codificador (solo examples de la petición): vale
	if err := (ConfigKNN{Modo: ModoKNNCPU, E5ONNX: "/e5.onnx"}).validar(); err != nil {
		t.Error(err)
	}
}

func TestW1DemosDePeticion(t *testing.T) {
	p := Pregunta{Texto: "¿q?", Kind: KindChoice, Opciones: []Opcion{{ID: "x"}, {ID: "y"}, {ID: "z"}},
		Ejemplos: []EjemploPeticion{{"a1", "x"}, {"a2", "x"}, {"a3", "x"}, {"b1", "y"}, {"c1", "z"}, {"c2", "z"}}}
	d := DemosDePeticion(p, 5)
	var got []string
	for _, e := range d {
		got = append(got, e.Texto)
	}
	if strings.Join(got, ",") != "a1,b1,c1,a2,c2" { // una vuelta por clase, en orden de aparición
		t.Fatalf("demos %v", got)
	}
	if d[0].Pregunta != "¿q?" || d[0].Tipo != KindChoice || len(d[0].Opciones) != 3 {
		t.Fatalf("%+v", d[0])
	}
	if _, err := turnosEjemplosL3(d); err != nil {
		t.Fatal(err)
	}
	if DemosDePeticion(p, 0) != nil || DemosDePeticion(Pregunta{}, 8) != nil {
		t.Fatal("sin demos")
	}
}

// L3 usa los examples de la petición como demostraciones fs, y el prompt es el de siempre si no hay (y no hay banco).
func TestW1L3UsaLosExamplesComoDemos(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"A": -0.1, "B": -2}}
	l := nuevoLP(t, f, nil)
	p := Pregunta{ID: "q", Texto: "¿cuál?", Kind: KindChoice, Estado: "caso nuevo", Opciones: []Opcion{{ID: "x"}, {ID: "y"}}}
	if _, err := l.Decidir(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	sin := len(f.cuerpo(0)["messages"].([]any))
	p.Ejemplos = []EjemploPeticion{{"demo uno", "x"}, {"demo dos", "y"}}
	if _, err := l.Decidir(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	ms := f.cuerpo(1)["messages"].([]any)
	if len(ms) != sin+4 || !strings.Contains(ms[1].(map[string]any)["content"].(string), "demo uno") || !strings.Contains(ms[3].(map[string]any)["content"].(string), "demo dos") {
		t.Fatalf("mensajes con demos: %d (sin: %d)", len(ms), sin)
	}
	// l3.ejemplos = 0 las apaga
	cero := 0
	l0 := nuevoLP(t, f, func(c *ConfigL3) { c.Ejemplos = &cero })
	if _, err := l0.Decidir(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	if n := len(f.cuerpo(len(f.cuerpos) - 1)["messages"].([]any)); n != sin {
		t.Fatalf("ejemplos 0: %d mensajes", n)
	}
}

// codMapa da vectores fijos por texto (norma 1), para fabricar probabilidades del kNN con la mediana distinta del argmax.
type codMapa map[string][]float32

func (c codMapa) Codificar(_ context.Context, t string) ([]float32, error) {
	v, ok := c[t]
	if !ok {
		return nil, errors.New("texto sin vector: " + t)
	}
	return v, nil
}

// En score, la respuesta del kNN por HTTP es la mediana CORAL de las `probs` que salen en la misma respuesta (y en choice, el argmax).
func TestW1ScoreRespondeLaMedianaPorHTTP(t *testing.T) {
	n := func(v ...float32) []float32 {
		var s float64
		for _, x := range v {
			s += float64(x * x)
		}
		for i := range v {
			v[i] /= float32(math.Sqrt(s))
		}
		return v
	}
	cod := codMapa{"n1": n(1, 0, 0, 0), "n2": n(0, 1, 0, 0), "n3": n(0, 0, 1, 0), "q": n(1, 0.2, 1, 0)}
	s, _ := servidorKNNFalso(t, cfgW1(func(c *ConfigKNN) { c.Vecinos = 3; c.Temperatura = 0.5 }), cod)
	ejs := `[{"text":"n1","label":"1"},{"text":"n2","label":"2"},{"text":"n3","label":"3"}]`
	ops := `[{"id":"3"},{"id":"1"},{"id":"2"}]`
	_, got := postDecide(t, s, cuerpoEj("q", "score", ops, ejs))
	var out struct {
		Results []struct {
			Answer string
			Probs  map[string]float64
		}
	}
	if err := json.Unmarshal([]byte(got), &out); err != nil || len(out.Results) != 1 {
		t.Fatal(err, got)
	}
	r := out.Results[0]
	p := []float64{r.Probs["3"], r.Probs["1"], r.Probs["2"]}
	want, ok := RespuestaKNN(KindScore, []string{"3", "1", "2"}, p)
	if !ok || r.Answer != want {
		t.Fatalf("answer %q, la regla da %q con %v", r.Answer, want, p)
	}
	if am := []string{"3", "1", "2"}[argmaxF(p)]; am == r.Answer {
		t.Fatalf("el caso debía separar la mediana (%s) del argmax (%s): probs %v", r.Answer, am, p)
	}
}

// Con l3.top_k_knn = 0 el modo gpu no usa el kNN ni con más de 20 opciones: no codifica nada.
func TestW1ModoGPUSinTopKNoCodifica(t *testing.T) {
	cod := &codHash{dim: dimBancoA(t)}
	k, _ := NuevoL2KNN(cfgW1(func(c *ConfigKNN) { c.Modo = ModoKNNGPU }), cod)
	k.ConTopK(false)
	var ops []Opcion
	for i := 0; i < L3TopKKNNDesde+3; i++ {
		ops = append(ops, Opcion{ID: strconv.Itoa(i)})
	}
	r, err := k.Decidir(context.Background(), Pregunta{ID: "q", Estado: "hola", Texto: "¿a?", Kind: KindChoice, Opciones: ops})
	if err != nil || r.MotivoKNN != MotivoKNNSiempre || r.RankKNN != nil || cod.llamas.Load() != 0 {
		t.Fatalf("%+v %v %d", r, err, cod.llamas.Load())
	}
	cero := 0
	c := Config{KNN: &ConfigKNN{}, L3: &ConfigL3{TopKKNN: &cero}}
	d := Config{KNN: &ConfigKNN{}, L3: &ConfigL3{}}
	if c.TopKKNNActivo() || !d.TopKKNNActivo() || (Config{}).TopKKNNActivo() {
		t.Fatal("TopKKNNActivo")
	}
}
