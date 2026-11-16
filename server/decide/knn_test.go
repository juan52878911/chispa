package decide

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type consultaKNN struct {
	ID       string    `json:"id"`
	Tarea    string    `json:"tarea"`
	Opciones []string  `json:"opciones"`
	Etiqueta *string   `json:"etiqueta"`
	L3       *string   `json:"l3"`
	L2       string    `json:"l2"`
	L2Probs  []float64 `json:"l2_probs"`
	Margen   float64   `json:"margen"`
	SimOro   float64   `json:"sim_oro"`
	Escala   bool      `json:"escala"`
	Motivo   string    `json:"motivo"`
	Final    *string   `json:"final"`
}

func leerConsultasKNN(t *testing.T, dir string) ([]consultaKNN, []float32, int) {
	t.Helper()
	x, n, dim, err := leerNpyF32(filepath.Join(dir, "consultas.npy"))
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(dir, "consultas.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var cs []consultaKNN
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var c consultaKNN
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			t.Fatal(err)
		}
		cs = append(cs, c)
	}
	if len(cs) != n {
		t.Fatalf("%d consultas y %d vectores", len(cs), n)
	}
	return cs, x, dim
}

// knnDeDir arma L2KNN con un banco por tarea (banco-<tarea>.npy/.jsonl en dir).
func knnDeDir(t *testing.T, dir string, tareas []string, u float64, siempre []string) *L2KNN {
	t.Helper()
	cfg := ConfigKNN{Tareas: map[string]TareaKNN{}, Umbral: u, Siempre: siempre}
	for _, n := range tareas {
		cfg.Tareas[n] = TareaKNN{Banco: filepath.Join(dir, "banco-"+n)}
	}
	k, err := NuevoL2KNN(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

type paridadKNN struct {
	n, l2ok, escalaOK, motivoOK int
	maxDif                      float64
	acc                         map[string][2]float64 // tarea → [aciertos, n] de la cascada con la l3 del fixture
	accL2                       map[string][2]float64
	porMotivo                   map[string]int
}

func motivoPy(m string) string {
	if m == "auroc" {
		return "siempre"
	}
	return m
}

// medirParidadKNN decide cada consulta del fixture con el kNN de Go y la compara con la decisión de Python.
func medirParidadKNN(t *testing.T, dir string, tareas []string, u float64, siempre []string) paridadKNN {
	t.Helper()
	cs, x, dim := leerConsultasKNN(t, dir)
	k := knnDeDir(t, dir, tareas, u, siempre)
	p := paridadKNN{acc: map[string][2]float64{}, accL2: map[string][2]float64{}, porMotivo: map[string]int{}}
	for i, c := range cs {
		d, err := k.Evaluar(c.Tarea, x[i*dim:(i+1)*dim], c.Opciones)
		if err != nil {
			t.Fatal(err)
		}
		p.n++
		if d.Answer == c.L2 {
			p.l2ok++
		} else if p.n-p.l2ok <= 5 {
			t.Logf("L2 distinto en %s: go %s py %s (margen py %.4f go %.4f)", c.ID, d.Answer, c.L2, c.Margen, d.Sen.Margen)
		}
		if d.Escala == c.Escala {
			p.escalaOK++
		}
		if d.Motivo == motivoPy(c.Motivo) {
			p.motivoOK++
		}
		p.porMotivo[d.Motivo]++
		for j := range c.L2Probs {
			p.maxDif = math.Max(p.maxDif, math.Abs(d.Probs[j]-c.L2Probs[j]))
		}
		// cascada con la respuesta de L3 del fixture (nil = L3 inválida: cuenta como fallo)
		var fin *string
		if d.Escala {
			fin = c.L3
		} else {
			a := d.Answer
			fin = &a
		}
		ok := 0.0
		if fin != nil && c.Etiqueta != nil && *fin == *c.Etiqueta {
			ok = 1
		}
		a := p.acc[c.Tarea]
		p.acc[c.Tarea] = [2]float64{a[0] + ok, a[1] + 1}
		ok2 := 0.0
		if c.Etiqueta != nil && d.Answer == *c.Etiqueta {
			ok2 = 1
		}
		b := p.accL2[c.Tarea]
		p.accL2[c.Tarea] = [2]float64{b[0] + ok2, b[1] + 1}
	}
	return p
}

func macroSinHC(m map[string][2]float64) float64 {
	var s float64
	var n int
	for t, v := range m {
		if t == "hatecheck-es" {
			continue
		}
		s += v[0] / v[1]
		n++
	}
	return s / float64(n)
}

// Fixture sintético (testdata/knn, lo genera generar.py con una reimplementación literal de escalado.l2 en Python): corre siempre.
func TestKNNSinteticoParidadConPython(t *testing.T) {
	p := medirParidadKNN(t, "testdata/knn", []string{"a", "b", "c"}, 0.50, []string{"c"})
	if p.l2ok != p.n || p.escalaOK != p.n || p.motivoOK != p.n {
		t.Fatalf("de %d filas: L2 %d, escala %d, motivo %d", p.n, p.l2ok, p.escalaOK, p.motivoOK)
	}
	if p.maxDif > 1e-5 {
		t.Errorf("|Δp| máximo %g", p.maxDif)
	}
	for _, m := range []string{"margen", "sin_oro", "siempre"} {
		if p.porMotivo[m] == 0 {
			t.Errorf("el fixture no ejercita el motivo %q", m)
		}
	}
}

// Paridad con el fixture real de Python (4 244 consultas, 8 tareas): CHISPA_PARIDAD_DIR=…/banco-paridad. Sin la variable, se salta.
func TestKNNParidadFixtureReal(t *testing.T) {
	dir := os.Getenv("CHISPA_PARIDAD_DIR")
	if dir == "" {
		t.Skip("CHISPA_PARIDAD_DIR sin definir")
	}
	var ref struct {
		Regla struct {
			U float64 `json:"u"`
		} `json:"regla"`
		U       float64  `json:"u"`
		Siempre []string `json:"siempre"`
		Macro   struct {
			Cascada float64 `json:"cascada"`
			L2      float64 `json:"l2"`
			L3      float64 `json:"l3"`
		} `json:"macro_sin_hc_B"`
	}
	b, err := os.ReadFile(filepath.Join(dir, "referencia.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	u := ref.U
	if u == 0 {
		u = ref.Regla.U
	}
	var tareas []string
	fs, _ := filepath.Glob(filepath.Join(dir, "banco-*.npy"))
	for _, f := range fs {
		tareas = append(tareas, strings.TrimSuffix(strings.TrimPrefix(filepath.Base(f), "banco-"), ".npy"))
	}
	p := medirParidadKNN(t, dir, tareas, u, ref.Siempre)
	pc := func(a int) float64 { return 100 * float64(a) / float64(p.n) }
	mc, ml2 := macroSinHC(p.acc), macroSinHC(p.accL2)
	t.Logf("n=%d u=%.5f siempre=%v tareas=%v", p.n, u, ref.Siempre, tareas)
	t.Logf("coincidencia con Python: L2 top-1 %.2f %%, escala %.2f %%, motivo %.2f %%; |Δp| máx %.2e", pc(p.l2ok), pc(p.escalaOK), pc(p.motivoOK), p.maxDif)
	t.Logf("macro sin hatecheck: cascada Go %.4f (Python %.4f), L2 Go %.4f (Python %.4f); motivos Go %v", mc, ref.Macro.Cascada, ml2, ref.Macro.L2, p.porMotivo)
	for tarea, v := range p.acc {
		t.Logf("  %-14s cascada %.4f  L2 %.4f  (n=%d)", tarea, v[0]/v[1], p.accL2[tarea][0]/p.accL2[tarea][1], int(v[1]))
	}
	if pc(p.l2ok) < 99 || pc(p.escalaOK) < 99 || pc(p.motivoOK) < 99 {
		t.Errorf("coincidencia < 99 %%")
	}
	if math.Abs(mc-ref.Macro.Cascada) > 0.005 {
		t.Errorf("macro de la cascada Go %.4f a más de 0,5 puntos de Python %.4f", mc, ref.Macro.Cascada)
	}
}

func TestKNNReglaUmbralDeOroYUniforme(t *testing.T) {
	// banco: 2 oro (etiqueta a) pegados a la consulta, 6 fondo (etiqueta b) algo más lejos pero más numerosos
	q := []float32{1, 0}
	mk := func(c float64) []float32 { return []float32{float32(c), float32(math.Sqrt(1 - c*c))} }
	b := &BancoKNN{Dim: 2, NOro: 2}
	add := func(et string, c float64) { b.X = append(b.X, mk(c)...); b.Etiq = append(b.Etiq, et) }
	add("a", 0.95)
	add("a", 0.93)
	for i := 0; i < 6; i++ {
		add("b", 0.985) // más cerca que el oro, pero el oro ≥ 0,90 manda y solo vota él
	}
	pr, sen, err := b.Votar(q, []string{"a", "b"}, ParamsKNN{})
	if err != nil || !sen.SoloOro || pr[0] != 1 || sen.Margen != 1 {
		t.Fatalf("regla de umbral: %v %+v %v", pr, sen, err)
	}
	// sin oro por encima de 0,90: votan todos y gana el fondo
	b2 := &BancoKNN{Dim: 2, NOro: 2}
	b2.X = append(b2.X, mk(0.5)...)
	b2.X = append(b2.X, mk(0.4)...)
	b2.Etiq = []string{"a", "a"}
	for i := 0; i < 3; i++ {
		b2.X = append(b2.X, mk(0.99)...)
		b2.Etiq = append(b2.Etiq, "b")
	}
	pr, sen, _ = b2.Votar(q, []string{"a", "b"}, ParamsKNN{})
	if sen.SoloOro || pr[1] < 0.99 {
		t.Fatalf("sin oro cercano vota todo el banco: %v %+v", pr, sen)
	}
	// votos solo entre las opciones de la fila; sin votos, uniforme
	pr, sen, _ = b2.Votar(q, []string{"x", "y", "z"}, ParamsKNN{})
	if math.Abs(pr[0]-1.0/3) > 1e-12 || sen.Margen != 0 {
		t.Fatalf("uniforme: %v %+v", pr, sen)
	}
	if _, _, err := b.Votar([]float32{1, 0, 0}, []string{"a"}, ParamsKNN{}); err == nil {
		t.Error("dimensión distinta debe fallar")
	}
}

func TestKNNBancoRechazaFormatoYOroDespuesDelFondo(t *testing.T) {
	d := t.TempDir()
	if err := os.WriteFile(filepath.Join(d, "x.npy"), []byte("no soy un npy"), 0o600); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(d, "x.jsonl"), nil, 0o600)
	if _, err := CargarBancoKNN(filepath.Join(d, "x")); err == nil {
		t.Error("un .npy inválido debe fallar")
	}
	// oro después del fondo
	b, _ := os.ReadFile("testdata/knn/banco-c.npy")
	os.WriteFile(filepath.Join(d, "y.npy"), b, 0o600)
	var ls []string
	f, _ := os.ReadFile("testdata/knn/banco-c.jsonl")
	ls = strings.Split(strings.TrimSpace(string(f)), "\n")
	ls[0], ls[len(ls)-1] = ls[len(ls)-1], ls[0]
	os.WriteFile(filepath.Join(d, "y.jsonl"), []byte(strings.Join(ls, "\n")+"\n"), 0o600)
	if _, err := CargarBancoKNN(filepath.Join(d, "y")); err == nil || !strings.Contains(err.Error(), "oro debe ir antes") {
		t.Errorf("oro después del fondo: %v", err)
	}
}

// Integración: la cascada L2(kNN)→L3 con el vector inyectado en el contexto, los motivos y las métricas.
func TestKNNEnLaCascadaMotivosYMetricas(t *testing.T) {
	cfg := ConfigKNN{Umbral: 0.5, Siempre: []string{"c"}, Tareas: map[string]TareaKNN{
		"a": {Banco: "testdata/knn/banco-a", Preguntas: []string{"¿a?"}},
		"b": {Banco: "testdata/knn/banco-b", Preguntas: []string{"¿b?"}},
		"c": {Banco: "testdata/knn/banco-c", Preguntas: []string{"¿c?"}},
	}}
	k, err := NuevoL2KNN(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	cs, x, dim := leerConsultasKNN(t, "testdata/knn")
	l3 := &decisorContadorL3{nombre: "L3", r: Respuesta{Answer: "L3", Probs: map[string]float64{"L3": 1}, Conf: 1}}
	m := NuevasMetricas()
	casc, err := NuevaCascada([]Capa{{"L0", 0.8}, {"L1", 0.8}, {"L2", 0.1}, {"L3", 0.5}}, []Decisor{Stub{"L0"}, Stub{"L1"}, k, l3}, m)
	if err != nil {
		t.Fatal(err)
	}
	casc.Saltar("L0", "L1")
	for i, c := range cs {
		p := Pregunta{ID: "q", Estado: "texto", Texto: "¿" + c.Tarea + "?", Kind: KindChoice}
		for _, o := range c.Opciones {
			p.Opciones = append(p.Opciones, Opcion{ID: o})
		}
		antes := l3.n
		r := casc.Decidir(ConVectorConsulta(context.Background(), x[i*dim:(i+1)*dim]), p)
		if c.Escala {
			if l3.n != antes+1 || r.Layer != "L3" {
				t.Fatalf("%s debía escalar a L3: %+v", c.ID, r)
			}
		} else if l3.n != antes || r.Layer != "L2" || r.Answer != c.L2 {
			t.Fatalf("%s debía contestar L2: %+v", c.ID, r)
		}
	}
	txt := metricasTexto(m)
	for _, s := range []string{`chispa_knn_filas_total{tarea="a"} 14`, `chispa_knn_escaladas_total{tarea="b",motivo="sin_oro"} 14`, `chispa_knn_escaladas_total{tarea="c",motivo="siempre"} 14`, `chispa_knn_escaladas_total{tarea="a",motivo="margen"} 4`} {
		if !strings.Contains(txt, s) {
			t.Errorf("falta %q en /metrics", s)
		}
	}
	// sin vector: no opina y sigue L3 (se cuenta)
	p := Pregunta{ID: "q", Estado: "t", Texto: "¿a?", Kind: KindChoice, Opciones: []Opcion{{ID: "x"}, {ID: "y"}}}
	if r := casc.Decidir(context.Background(), p); r.Layer != "L3" || !strings.Contains(metricasTexto(m), "chispa_knn_sin_vector_total 1") {
		t.Errorf("sin vector: %+v", r)
	}
	// pregunta de una tarea sin banco: tampoco opina
	p.Texto = "¿otra?"
	if r := casc.Decidir(ConVectorConsulta(context.Background(), x[:dim]), p); r.Layer != "L3" {
		t.Errorf("tarea desconocida: %+v", r)
	}
}

// El motivo «incompleta»: el kNN escaló y L3 falló.
func TestKNNMotivoIncompleta(t *testing.T) {
	cfg := ConfigKNN{Umbral: 0.5, Tareas: map[string]TareaKNN{"b": {Banco: "testdata/knn/banco-b", Preguntas: []string{"¿b?"}}}}
	k, err := NuevoL2KNN(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := NuevasMetricas()
	casc, _ := NuevaCascada([]Capa{{"L2", 0.1}, {"L3", 0.5}}, []Decisor{k, errL3Falso{}}, m)
	_, x, dim := leerConsultasKNN(t, "testdata/knn")
	p := Pregunta{ID: "q", Estado: "t", Texto: "¿b?", Kind: KindChoice, Opciones: []Opcion{{ID: "p"}, {ID: "q"}}}
	r := casc.Decidir(ConVectorConsulta(context.Background(), x[14*dim:15*dim]), p)
	if !r.Escalate || r.Answer == "" {
		t.Fatalf("L3 falló: debía escalar con la pista del kNN: %+v", r)
	}
	if !strings.Contains(metricasTexto(m), `chispa_knn_escaladas_total{tarea="b",motivo="incompleta"} 1`) {
		t.Error("falta el motivo incompleta")
	}
}

type errL3Falso struct{}

func (errL3Falso) Nombre() string { return "L3" }
func (errL3Falso) Decidir(context.Context, Pregunta) (Respuesta, error) {
	return Respuesta{}, ErrSinL3
}

func TestKNNConfigValida(t *testing.T) {
	ok := ConfigKNN{Umbral: 0.1, Tareas: map[string]TareaKNN{"a": {Banco: "x", Preguntas: []string{"p"}}}}
	if err := ok.validar(); err != nil {
		t.Fatal(err)
	}
	for nombre, mod := range map[string]func(*ConfigKNN){
		"sin tareas":        func(c *ConfigKNN) { c.Tareas = nil },
		"umbral > 1":        func(c *ConfigKNN) { c.Umbral = 2 },
		"banco vacío":       func(c *ConfigKNN) { c.Tareas = map[string]TareaKNN{"a": {}} },
		"siempre inexiste":  func(c *ConfigKNN) { c.Siempre = []string{"zz"} },
		"pregunta repetida": func(c *ConfigKNN) { c.Tareas["b"] = TareaKNN{Banco: "y", Preguntas: []string{"p"}} },
	} {
		c := ok
		c.Tareas = map[string]TareaKNN{"a": ok.Tareas["a"]}
		mod(&c)
		if c.validar() == nil {
			t.Errorf("%s debía fallar", nombre)
		}
	}
}
