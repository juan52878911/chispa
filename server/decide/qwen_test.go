package decide

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"chispa/runtime/tok"
)

const dirQwenTok = "../../runtime/tok/testdata/qwen/"

func bpeMini(t *testing.T) *tok.BPE {
	t.Helper()
	b, err := tok.AbrirBPE(dirQwenTok + "mini.chispa-bpe")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// ejecutor falso: guarda los ids que recibe y devuelve T filas conocidas (la fila i es (i+1, -(i+1), 2·(i+1)) repetida d/3 veces… aquí d = 3).
type ejFalso struct {
	ids   [][]int64
	err   error
	forma func(t int) (h []float32, tt, d int)
}

func (e *ejFalso) Correr(_ context.Context, ids []int64) ([]float32, int, int, error) {
	e.ids = append(e.ids, ids)
	if e.err != nil {
		return nil, 0, 0, e.err
	}
	if e.forma != nil {
		h, t, d := e.forma(len(ids))
		return h, t, d, nil
	}
	h := make([]float32, 0, len(ids)*3)
	for i := range ids {
		f := float32(i + 1)
		h = append(h, f, -f, 2*f)
	}
	return h, len(ids), 3, nil
}
func (e *ejFalso) Sesiones() int { return 1 }
func (e *ejFalso) Close() error  { return nil }

// Las consultas se arman como `consulta_emb` de R11: las entradas del fixture (generadas por referencia_bpe.py) son byte a byte las de QwenConsulta.
func TestQwenConsultaIgualQueRef(t *testing.T) {
	f, err := os.Open(dirQwenTok + "fixture_paridad.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ref := map[string]string{}
	g, _ := os.Open(dirQwenTok + "ids_mini.jsonl")
	defer g.Close()
	sc := bufio.NewScanner(g)
	sc.Buffer(make([]byte, 1<<22), 1<<22)
	for sc.Scan() {
		var r struct{ ID, Entrada string }
		json.Unmarshal(sc.Bytes(), &r)
		ref[r.ID] = r.Entrada
	}
	sc = bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<22), 1<<22)
	n, desc := 0, 0
	for sc.Scan() {
		var r struct{ ID, Tipo, Pregunta, Texto string }
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		got := r.Texto
		if r.Tipo == "consulta" {
			got = QwenConsulta(r.Pregunta, r.Texto)
		} else {
			desc++
		}
		if got != ref[r.ID] {
			t.Fatalf("%s: %q ≠ %q", r.ID, got, ref[r.ID])
		}
		n++
	}
	if n != 200 || desc == 0 {
		t.Fatalf("%d filas, %d descripciones", n, desc)
	}
	if QwenConsulta("¿P?", "x") != "Instruct: Classify the text to answer this question: ¿P?\nQuery:x" {
		t.Fatal("formato de la instrucción")
	}
}

// Pooling del último token y L2, y entrada exacta: ids = los del tokenizador con la instrucción, truncados a 320 y SIN normalizar mayúsculas.
func TestQwenCodificadorPoolingYEntrada(t *testing.T) {
	b := bpeMini(t)
	ej := &ejFalso{}
	c := nuevoCodificadorQwen(b, ej)
	var cq CodificadorConsulta = c // el tipo cumple la interfaz que usa L2KNN
	v, err := cq.CodificarConsulta(context.Background(), "¿Qué emoción?", "ESTO ES UN GRITO ÑANDÚ")
	if err != nil {
		t.Fatal(err)
	}
	want := b.IDs(QwenConsulta("¿Qué emoción?", "ESTO ES UN GRITO ÑANDÚ"), QwenMaxTok)
	got := ej.ids[0]
	if len(got) != len(want) {
		t.Fatalf("ids %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != int64(want[i]) {
			t.Fatalf("id %d: %d vs %d", i, got[i], want[i])
		}
	}
	if low := b.IDs(QwenConsulta("¿Qué emoción?", strings.ToLower("ESTO ES UN GRITO ÑANDÚ")), QwenMaxTok); len(low) == len(want) && reflect.DeepEqual(low, want) {
		t.Fatal("el test no distingue mayúsculas de minúsculas")
	}
	// la última fila es (T, -T, 2T): normalizada = (1, -1, 2)/√6
	r6 := float32(math.Sqrt(6))
	if len(v) != 3 || math.Abs(float64(v[0]-1/r6)) > 1e-6 || math.Abs(float64(v[1]+1/r6)) > 1e-6 || math.Abs(float64(v[2]-2/r6)) > 1e-6 {
		t.Fatalf("vector %v", v)
	}
	// descripción: sin instrucción
	if _, err := c.Codificar(context.Background(), "ñandú"); err != nil {
		t.Fatal(err)
	}
	if want := b.IDs("ñandú", QwenMaxTok); len(ej.ids[1]) != len(want) {
		t.Fatalf("la descripción lleva la instrucción: %v", ej.ids[1])
	}
	// truncado a 320
	largo := strings.Repeat("palabra distinta número 12345 ñ ", 400)
	if _, err := c.CodificarConsulta(context.Background(), "p", largo); err != nil {
		t.Fatal(err)
	}
	if n := len(ej.ids[2]); n != QwenMaxTok {
		t.Fatalf("truncado a %d ids, quiero %d", n, QwenMaxTok)
	}
	if c.Sesiones() != 1 || c.Close() != nil {
		t.Fatal("sesiones/close")
	}
}

func TestQwenCodificadorErrores(t *testing.T) {
	b := bpeMini(t)
	boom := errors.New("ort roto")
	if _, err := nuevoCodificadorQwen(b, &ejFalso{err: boom}).Codificar(context.Background(), "a"); !errors.Is(err, boom) {
		t.Fatalf("el error del ejecutor se pierde: %v", err)
	}
	mal := &ejFalso{forma: func(t int) ([]float32, int, int) { return make([]float32, 5), t, 3 }}
	if _, err := nuevoCodificadorQwen(b, mal).Codificar(context.Background(), "a"); err == nil || !strings.Contains(err.Error(), "last_hidden_state") {
		t.Fatalf("forma incoherente: %v", err)
	}
	// sin ORT (o sin el fichero) los errores dicen qué falta
	if _, err := NuevoCodificadorQwenORT("/no/existe.onnx", b, 1); err == nil || !(strings.Contains(err.Error(), "ort") || strings.Contains(err.Error(), "existe")) {
		t.Fatalf("falta el ONNX: %v", err)
	}
}

// L2KNN llama a CodificarConsulta con la PREGUNTA de la tarea y el texto (Estado) si el codificador la quiere, y a Codificar con el texto si no.
type codFalsoConsulta struct{ pregunta, texto string }

func (c *codFalsoConsulta) Codificar(context.Context, string) ([]float32, error) {
	return nil, errors.New("no debía llamarse")
}
func (c *codFalsoConsulta) CodificarConsulta(_ context.Context, p, t string) ([]float32, error) {
	c.pregunta, c.texto = p, t
	return make([]float32, 8), nil
}

type codFalsoPlano struct{ texto string }

func (c *codFalsoPlano) Codificar(_ context.Context, t string) ([]float32, error) {
	c.texto = t
	return make([]float32, 8), nil
}

func TestL2KNNUsaCodificarConsultaSiExiste(t *testing.T) {
	cfg := ConfigKNN{Tareas: map[string]TareaKNN{"a": {Banco: "testdata/knn/banco-a", Preguntas: []string{"¿Qué emoción?"}}}}
	pg := Pregunta{ID: "q", Estado: "ESTADO ORIGINAL", Texto: "¿Qué emoción?", Kind: KindChoice, Opciones: []Opcion{{ID: "x"}, {ID: "y"}}}
	cc := &codFalsoConsulta{}
	k, err := NuevoL2KNN(cfg, cc)
	if err != nil {
		t.Fatal(err)
	}
	k.Decidir(context.Background(), pg)
	if cc.pregunta != "¿Qué emoción?" || cc.texto != "ESTADO ORIGINAL" {
		t.Fatalf("CodificarConsulta(%q, %q)", cc.pregunta, cc.texto)
	}
	cp := &codFalsoPlano{}
	k, _ = NuevoL2KNN(cfg, cp)
	k.Decidir(context.Background(), pg)
	if cp.texto != "ESTADO ORIGINAL" {
		t.Fatalf("Codificar(%q)", cp.texto)
	}
}

// Una config anterior a qwen3-emb (sin `codificador`) es e5-small con los mismos campos de siempre; el ejemplo del producto sigue siendo e5-small.
func TestConfigVieja_E5SinCambios(t *testing.T) {
	viejo := `{"knn": {"umbral": 0.5, "e5_onnx": "/m/e5.onnx", "e5_sesiones": 3, "vecinos": 32, "tareas": {"t": {"banco": "/b/banco-t", "preguntas": ["¿t?"]}}}}`
	dir := t.TempDir()
	if err := os.WriteFile(dir+"/c.json", []byte(viejo), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := CargarConfig(dir + "/c.json")
	if err != nil {
		t.Fatal(err)
	}
	k := c.KNN
	if k.NombreCodificador() != CodificadorE5Small || k.Codificador != "" || k.E5ONNX != "/m/e5.onnx" || k.E5Sesiones != 3 || k.QwenONNX != "" || k.QwenTok != "" || k.QwenSesiones != 0 {
		t.Fatalf("la config vieja cambió: %+v", k)
	}
	// explícito es lo mismo
	k2 := *k
	k2.Codificador = CodificadorE5Small
	if err := k2.validar(); err != nil {
		t.Fatal(err)
	}
	ej, err := CargarConfig("../deploy/config.ejemplo.json")
	if err != nil {
		t.Fatal(err)
	}
	if ej.KNN.NombreCodificador() != CodificadorE5Small || ej.KNN.E5ONNX == "" {
		t.Fatalf("el ejemplo del producto debe seguir en e5-small: %+v", ej.KNN)
	}
	q, err := CargarConfig("../deploy/config.ejemplo-qwen3emb.json")
	if err != nil {
		t.Fatal(err)
	}
	if q.KNN.NombreCodificador() != CodificadorQwen3Emb || q.KNN.QwenONNX == "" || q.KNN.QwenTok == "" || q.KNN.E5ONNX != "" || q.KNN.Vecinos != 32 {
		t.Fatalf("ejemplo qwen3-emb: %+v", q.KNN)
	}
}

func TestConfigCodificadorValida(t *testing.T) {
	tareas := map[string]TareaKNN{"t": {Banco: "b"}}
	casos := []struct {
		nombre string
		k      ConfigKNN
		error  string // "" = válida
	}{
		{"e5 por defecto", ConfigKNN{Tareas: tareas, E5ONNX: "x"}, ""},
		{"qwen completa", ConfigKNN{Tareas: tareas, Codificador: "qwen3-emb", QwenONNX: "a", QwenTok: "b", Vecinos: 8}, ""},
		{"qwen sin onnx", ConfigKNN{Tareas: tareas, Codificador: "qwen3-emb", QwenTok: "b"}, "qwen_onnx"},
		{"qwen sin tok", ConfigKNN{Tareas: tareas, Codificador: "qwen3-emb", QwenONNX: "a"}, "qwen_tok"},
		{"qwen con e5_onnx", ConfigKNN{Tareas: tareas, Codificador: "qwen3-emb", QwenONNX: "a", QwenTok: "b", E5ONNX: "x"}, "e5_onnx"},
		{"e5 con campos de qwen", ConfigKNN{Tareas: tareas, QwenONNX: "a"}, "qwen3-emb"},
		{"desconocido", ConfigKNN{Tareas: tareas, Codificador: "bge"}, "desconocido"},
		{"sesiones negativas", ConfigKNN{Tareas: tareas, Codificador: "qwen3-emb", QwenONNX: "a", QwenTok: "b", QwenSesiones: -1}, "qwen_sesiones"},
	}
	for _, c := range casos {
		err := c.k.validar()
		if c.error == "" && err != nil || c.error != "" && (err == nil || !strings.Contains(err.Error(), c.error)) {
			t.Errorf("%s: %v (se esperaba %q)", c.nombre, err, c.error)
		}
	}
}

type codDim struct{ d int }

func (c codDim) Codificar(context.Context, string) ([]float32, error) { return make([]float32, c.d), nil }

func TestL2KNNComprobarDim(t *testing.T) {
	cfg := ConfigKNN{Tareas: map[string]TareaKNN{"a": {Banco: "testdata/knn/banco-a"}}} // banco de 8 dimensiones
	k, _ := NuevoL2KNN(cfg, codDim{8})
	if err := k.ComprobarDim(context.Background()); err != nil {
		t.Fatal(err)
	}
	k, _ = NuevoL2KNN(cfg, codDim{1024})
	if err := k.ComprobarDim(context.Background()); err == nil || !strings.Contains(err.Error(), "dimensión 8") || !strings.Contains(err.Error(), "1024") {
		t.Fatalf("dimensión distinta: %v", err)
	}
	k, _ = NuevoL2KNN(cfg, nil)
	if err := k.ComprobarDim(context.Background()); err != nil {
		t.Fatal("sin codificador no hay nada que comprobar")
	}
}
