package decide

import (
	"strings"
	"testing"
)

// Los ejemplos fs8 se leen de <banco>.ejemplos.jsonl: solo en tareas con oro; sin fichero, aviso y zero-shot; un fichero con más ejemplos de los pedidos no arranca.
func TestL2KNNEjemplosFSDeLosBancos(t *testing.T) {
	cfg := ConfigKNN{Umbral: 0.5, Tareas: map[string]TareaKNN{
		"a": {Banco: "testdata/knn/banco-a", Preguntas: []string{"¿a?"}}, // oro + ejemplos
		"b": {Banco: "testdata/knn/banco-b", Preguntas: []string{"¿b?"}}, // sin oro
		"c": {Banco: "testdata/knn/banco-c", Preguntas: []string{"¿c?"}}, // oro sin fichero de ejemplos
	}}
	k, err := NuevoL2KNN(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(k.EjemplosDe("¿a?")) != 3 || k.EjemplosDe("¿b?") != nil || k.EjemplosDe("¿c?") != nil || k.EjemplosDe("otra") != nil {
		t.Errorf("ejemplos por pregunta: a=%d b=%v c=%v", len(k.EjemplosDe("¿a?")), k.EjemplosDe("¿b?"), k.EjemplosDe("¿c?"))
	}
	if len(k.Avisos) != 1 || !strings.Contains(k.Avisos[0], `"c"`) {
		t.Errorf("avisos: %v", k.Avisos)
	}
	if err := k.ComprobarEjemplos(8); err != nil {
		t.Error(err)
	}
	if err := k.ComprobarEjemplos(2); err == nil || !strings.Contains(err.Error(), "generar_banco.py --k 2") {
		t.Errorf("3 ejemplos con l3.ejemplos=2 debía fallar: %v", err)
	}
}

// Con ejemplos, el prompt de L3 lleva los turnos previos y el sistema con la nota; apagado, es el zero-shot (la paridad byte a byte con Python está en l3_fs_paridad_test.go).
func TestL3FewShotMensajes(t *testing.T) {
	ejs, err := CargarEjemplosFS("testdata/knn/banco-a")
	if err != nil || len(ejs) != 3 {
		t.Fatal(err, len(ejs))
	}
	turnos, err := turnosEjemplosL3(ejs)
	if err != nil || len(turnos) != 6 {
		t.Fatal(err, len(turnos))
	}
	if turnos[1].Role != "assistant" || turnos[1].Content != "A" || turnos[3].Content != "B" || turnos[5].Content != "C" {
		t.Errorf("la respuesta del asistente es la etiqueta de la opción correcta: %v %v %v", turnos[1], turnos[3], turnos[5])
	}
	if !strings.Contains(turnos[0].Content, `"opciones": [{"id": "x", "etiqueta": "A"}, {"id": "y", "etiqueta": "B"}, {"id": "z", "etiqueta": "C"}]`) ||
		strings.Contains(turnos[0].Content, "descripcion") || !strings.Contains(turnos[0].Content, "ejemplo 0 (b)x(/b)") {
		t.Errorf("turno de usuario: %s", turnos[0].Content)
	}
	msgs, _, _ := mensajesLogprobsL3(Pregunta{Texto: "¿a?", Kind: KindChoice, Estado: "caso", Opciones: []Opcion{{ID: "x"}, {ID: "y"}}}, []Opcion{{ID: "x"}, {ID: "y"}})
	fs := conEjemplosL3(msgs, turnos)
	if len(fs) != 8 || !strings.HasSuffix(fs[0].Content, "trae sus opciones con etiqueta completas.") || fs[7] != msgs[1] {
		t.Errorf("mensajes fs: %d, sistema %q", len(fs), fs[0].Content[len(fs[0].Content)-60:])
	}
	// recorte a 1 200 caracteres (puntos de código)
	largo := EjemploFS{ID: "l", Pregunta: "¿a?", Tipo: "choice", Texto: strings.Repeat("ñ", 2000), Opciones: []string{"x", "y"}, Etiqueta: "x"}
	tt, _ := turnosEjemplosL3([]EjemploFS{largo})
	if n := strings.Count(tt[0].Content, "ñ"); n != 1200 {
		t.Errorf("recorte: %d caracteres", n)
	}
}
