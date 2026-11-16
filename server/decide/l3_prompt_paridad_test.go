package decide

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

// Paridad del PROMPT de L3 en cascada con el de `model/alt/llm_logprobs.py` (T5c). Cada fila de testdata/l3_prompt_paridad.jsonl (la genera
// `python3 -m model.alt.exportar_prompts_l3`) trae la consulta tal como la manda un cliente por /v1/decide y los mensajes y etiquetas que da
// `llm_logprobs.preparar` para la misma fila de la batería. La cascada es la del servidor con `knn`: una L2 que SIEMPRE escala (con la guía de
// sus probabilidades, como el kNN) y L3 por logprobs montada con `Config.DecisorL3`; se compara byte a byte el primer cuerpo que llega al vLLM falso.
// CHISPA_PROMPTS_L3=fichero.jsonl usa otro fichero (p. ej. las ~190 filas del fixture completo).

type filaPromptL3 struct {
	ID          string            `json:"id"`
	Tarea       string            `json:"tarea"`
	Tipo        string            `json:"tipo"`
	Pregunta    string            `json:"pregunta"`
	Texto       string            `json:"texto"`
	Opciones    []string          `json:"opciones"`
	Descripcion map[string]string `json:"descripciones"`
	Mensajes    []msgL3           `json:"mensajes"`
	Etiquetas   []string          `json:"etiquetas"`
}

func leerPromptsL3(t *testing.T) []filaPromptL3 {
	t.Helper()
	ruta := os.Getenv("CHISPA_PROMPTS_L3")
	if ruta == "" {
		ruta = "testdata/l3_prompt_paridad.jsonl"
	}
	f, err := os.Open(ruta)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []filaPromptL3
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r filaPromptL3
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		out = append(out, r)
	}
	if len(out) == 0 {
		t.Fatal("sin filas")
	}
	return out
}

// escaladorKNN imita a L2KNN cuando escala: devuelve las probabilidades del kNN (aquí uniformes, con una pista distinta de la primera opción)
// y un τ que nunca se alcanza, de modo que la cascada sigue con L3, que recibe sus probabilidades como «previas».
type escaladorKNN struct{}

func (escaladorKNN) Nombre() string { return "L2" }
func (escaladorKNN) Decidir(_ context.Context, p Pregunta) (Respuesta, error) {
	probs := map[string]float64{}
	for i, o := range p.Opciones {
		probs[o.ID] = 1 / float64(len(p.Opciones)) * (1 + 0.001*float64(i))
	}
	tau := math.Inf(1)
	return Respuesta{Answer: p.Opciones[len(p.Opciones)-1].ID, Probs: probs, Conf: 0.2, TauPropio: &tau, Tarea: "x", MotivoKNN: MotivoKNNMargen}, nil
}

// primerosCuerposPorFila sirve cada fila (en serie) por /v1/decide y devuelve el primer cuerpo que recibió el vLLM falso por fila.
func primerosCuerposPorFila(t *testing.T, filas []filaPromptL3, cfg Config, conDescripciones, forzado bool) []struct{ Messages []msgL3 } {
	t.Helper()
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}}
	s := f.arranca(t)
	cfg.L3 = &ConfigL3{URL: s.URL, Modelo: "m", Modo: L3ModoLogprobs, Tau: new(float64)}
	l3, err := cfg.DecisorL3()
	if err != nil {
		t.Fatal(err)
	}
	m := NuevasMetricas()
	casc, err := NuevaCascada([]Capa{{"L2", 0.5}, {"L3", 0.5}}, []Decisor{escaladorKNN{}, l3}, m)
	if err != nil {
		t.Fatal(err)
	}
	if forzado { // forzar_l3: la cascada salta L2 y solo corre L3, sin previas (la vía de R9-e2e, 99,84 % de acuerdo con Python)
		casc.SoloL3()
	}
	srv := httptest.NewServer(NuevoServidor(casc, m, 0).Handler())
	defer srv.Close()
	var out []struct{ Messages []msgL3 }
	for _, r := range filas {
		antes := len(f.cuerpos)
		ops := make([]map[string]any, len(r.Opciones))
		for i, o := range r.Opciones {
			ops[i] = map[string]any{"id": o}
			if d := r.Descripcion[o]; conDescripciones && d != "" {
				ops[i]["description"] = d
			}
		}
		cuerpo, _ := json.Marshal(map[string]any{"state": r.Texto, "questions": []any{map[string]any{"id": "q", "question": r.Pregunta, "kind": r.Tipo, "options": ops}}})
		resp, err := http.Post(srv.URL+"/v1/decide", "application/json", bytes.NewReader(cuerpo))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		f.mu.Lock()
		nuevos := f.cuerpos[antes:]
		f.mu.Unlock()
		var e struct{ Messages []msgL3 }
		if len(nuevos) > 0 {
			if err := json.Unmarshal([]byte(nuevos[0]), &e); err != nil {
				t.Fatal(err)
			}
		}
		out = append(out, e)
	}
	return out
}

func igualesMsgs(a, b []msgL3) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Con knn activo, L3 recibe el mismo prompt que llm_logprobs.py: mismas opciones en el mismo orden (las mismas etiquetas), mismas descripciones, sin recorte.
func TestL3PromptCascadaIdenticoAPython(t *testing.T) {
	filas := leerPromptsL3(t)
	cfg := Config{KNN: &ConfigKNN{}} // con knn el recorte a top_k y la guía de capas previas se apagan
	got := primerosCuerposPorFila(t, filas, cfg, true, false)
	malas := 0
	for i, r := range filas {
		if !igualesMsgs(got[i].Messages, r.Mensajes) {
			malas++
			if malas <= 3 {
				t.Errorf("%s (%s, %d opciones): el prompt de la cascada difiere del de Python\n go: %.300q\n py: %.300q", r.ID, r.Tarea, len(r.Opciones), got[i].Messages, r.Mensajes)
			}
		}
	}
	if malas > 0 {
		t.Fatalf("%d de %d filas con prompt distinto", malas, len(filas))
	}
	t.Logf("%d de %d filas con el prompt idéntico byte a byte al de Python", len(filas)-malas, len(filas))
}

// Causa del −3,2 puntos de T5b-e2e: las consultas del fixture iban SIN las descripciones de las opciones (consultas_con_texto.py no las traía),
// y el prompt de Python sí las lleva: sin ellas el prompt de L3 es otro. Este test fija que descripciones y prompt dependen una de otra.
func TestL3PromptSinDescripcionesNoEsElDePython(t *testing.T) {
	filas := leerPromptsL3(t)
	con := 0
	var cd []filaPromptL3
	for _, r := range filas {
		for _, d := range r.Descripcion {
			if d != "" {
				con++
				cd = append(cd, r)
				break
			}
		}
	}
	if con == 0 {
		t.Skip("ninguna fila con descripciones")
	}
	got := primerosCuerposPorFila(t, cd, Config{KNN: &ConfigKNN{}}, false, false)
	distintas := 0
	for i, r := range cd {
		if !igualesMsgs(got[i].Messages, r.Mensajes) {
			distintas++
		}
	}
	if distintas != len(cd) {
		t.Errorf("de %d filas con descripciones, sin ellas solo %d cambian de prompt", len(cd), distintas)
	}
}

// El prompt de L3 en cascada (kNN que escala + L3) es idéntico, byte a byte, al que se manda con L3 forzado (forzar_l3) para la misma fila, y los dos al de
// llm_logprobs.py. Si alguien vuelve a meter un recorte, un reordenado o una pista en la vía de cascada, este test falla.
func TestL3PromptCascadaIdenticoAlForzadoYAPython(t *testing.T) {
	filas := leerPromptsL3(t)
	cascada := primerosCuerposPorFila(t, filas, Config{KNN: &ConfigKNN{}}, true, false)
	forzado := primerosCuerposPorFila(t, filas, Config{}, true, true)
	for i, r := range filas {
		if !igualesMsgs(cascada[i].Messages, forzado[i].Messages) {
			t.Errorf("%s (%s): el prompt en cascada difiere del de L3 forzado\n cascada: %.300q\n forzado: %.300q", r.ID, r.Tarea, cascada[i].Messages, forzado[i].Messages)
		}
		if !igualesMsgs(forzado[i].Messages, r.Mensajes) {
			t.Errorf("%s (%s): el prompt con L3 forzado difiere del de llm_logprobs.py", r.ID, r.Tarea)
		}
	}
}
