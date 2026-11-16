package decide

import (
	"encoding/json"
	"os"
	"testing"
)

// Paridad del prompt few-shot (fs8) de L3 con `llm_logprobs.py --ejemplos --k 8` (producto). testdata/l3_fs_paridad.jsonl lo genera
// `python3 -m model.alt.exportar_prompts_l3 --fs-k 8 --fs-salida …`: por fila, los mensajes de `con_ejemplos(preparar(fila), turnos_ejemplos(ejemplos, 1200))`
// (`mensajes_fs`, null en las tareas sin oro: zero-shot) y, aparte, los ejemplos que eligió `elegir_ejemplos` (testdata/l3_fs_ejemplos.json), que es
// lo que el servidor lee de <banco>.ejemplos.jsonl. Se compara byte a byte el primer cuerpo que llega al vLLM falso, en cascada con knn.
// CHISPA_PROMPTS_FS=fichero.jsonl y CHISPA_EJEMPLOS_FS=fichero.json usan otros datos.

func ejemplosFSTest(t *testing.T) map[string][]EjemploFS {
	t.Helper()
	ruta := os.Getenv("CHISPA_EJEMPLOS_FS")
	if ruta == "" {
		ruta = "testdata/l3_fs_ejemplos.json"
	}
	b, err := os.ReadFile(ruta)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string][]EjemploFS
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestL3FewShotIdenticoAPython(t *testing.T) {
	if r := os.Getenv("CHISPA_PROMPTS_FS"); r != "" {
		t.Setenv("CHISPA_PROMPTS_L3", r)
	} else {
		t.Setenv("CHISPA_PROMPTS_L3", "testdata/l3_fs_paridad.jsonl")
	}
	filas := leerPromptsL3(t)
	ej := ejemplosFSTest(t)
	porPregunta := map[string][]EjemploFS{}
	for _, r := range filas {
		porPregunta[r.Pregunta] = ej[r.Tarea]
	}
	got := primerosCuerposPorFila(t, filas, Config{KNN: &ConfigKNN{}}, true, false, func(p string) []EjemploFS { return porPregunta[p] })
	con, sin := 0, 0
	for i, r := range filas {
		want := r.MensajesFS
		if want == nil { // tarea sin oro: zero-shot, el prompt de siempre
			want = r.Mensajes
			sin++
		} else {
			con++
		}
		if !igualesMsgs(got[i].Messages, want) {
			t.Errorf("%s (%s): el prompt fs8 difiere del de llm_logprobs.py --ejemplos\n go: %.400q\n py: %.400q", r.ID, r.Tarea, got[i].Messages, want)
		}
	}
	if con == 0 {
		t.Fatal("ninguna fila con few-shot")
	}
	t.Logf("%d filas con fs8 y %d zero-shot (sin oro), todas idénticas byte a byte a las de Python", con, sin)
}

// Con l3.ejemplos = 0 (o sin ejemplos) el prompt es el zero-shot de siempre.
func TestL3FewShotApagadoEsZeroShot(t *testing.T) {
	t.Setenv("CHISPA_PROMPTS_L3", "testdata/l3_fs_paridad.jsonl")
	filas := leerPromptsL3(t)
	got := primerosCuerposPorFila(t, filas, Config{KNN: &ConfigKNN{}}, true, false, nil)
	for i, r := range filas {
		if !igualesMsgs(got[i].Messages, r.Mensajes) {
			t.Errorf("%s: sin ejemplos el prompt debe ser el zero-shot", r.ID)
		}
	}
}
