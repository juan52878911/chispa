package decide

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Few-shot (fs8) de L3: cuando L3 contesta una pregunta de una tarea con banco de oro, el prompt lleva como turnos previos usuario/asistente k ejemplos
// de oro de la tarea (estratificados por clase y con semilla fija), como `llm_logprobs.py --ejemplos --k 8`, la referencia exacta: mismo sistema
// (+ NOTA_EJEMPLOS), un turno de usuario por ejemplo con las opciones solo como id → etiqueta (sin descripciones, para que quepan 8 con 60 opciones) y el
// texto recortado a 1 200 caracteres, un turno de asistente con la etiqueta correcta y, al final, el caso nuevo tal cual lo prepara `preparar`.
//
// La ELECCIÓN de los ejemplos (random.Random(semilla) de Python: barajado de clases, de ejemplos de cada clase y del orden final) la hace el
// generador del banco (model/alt/generar_banco.py, con `llm_logprobs.elegir_ejemplos`) y se guarda en `<banco>.ejemplos.jsonl`; el servidor solo
// los lee, así que la paridad con Python es exacta sin reimplementar el Mersenne Twister. Una tarea sin oro (o sin fichero de ejemplos) contesta zero-shot.

const (
	// L3EjemplosPorDefecto: nº de ejemplos de oro de L3 (l3.ejemplos ausente); 0 apaga el few-shot.
	L3EjemplosPorDefecto = 8
	// ejemploMaxChars: tope de caracteres (puntos de código) del texto de cada ejemplo (`--ejemplo-max-chars` de Python).
	ejemploMaxChars = 1200
	// notaEjemplosL3 se añade al mensaje de sistema cuando hay ejemplos (NOTA_EJEMPLOS de llm_logprobs.py).
	notaEjemplosL3 = ` Los turnos previos de usuario/asistente son ejemplos resueltos de la misma tarea: el asistente respondió con la etiqueta correcta de cada texto; ` +
		`el último mensaje de usuario es el caso nuevo y trae sus opciones con etiqueta completas.`
)

// EjemploFS es un ejemplo de oro de una tarea (una línea de `<banco>.ejemplos.jsonl`): la fila tal cual la lee `llm_logprobs.preparar`.
type EjemploFS struct {
	ID       string   `json:"id"`
	Pregunta string   `json:"pregunta"`
	Tipo     string   `json:"tipo"`
	Texto    string   `json:"texto"`
	Opciones []string `json:"opciones"`
	Etiqueta string   `json:"etiqueta"`
}

// CargarEjemplosFS lee `<base>.ejemplos.jsonl`; (nil, nil) si no existe.
func CargarEjemplosFS(base string) ([]EjemploFS, error) {
	f, err := os.Open(base + ".ejemplos.jsonl")
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []EjemploFS
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var e EjemploFS
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("%s.ejemplos.jsonl línea %d: %w", base, len(out)+1, err)
		}
		if e.Etiqueta == "" || len(e.Opciones) < 2 {
			return nil, fmt.Errorf("%s.ejemplos.jsonl línea %d: faltan la etiqueta o las opciones", base, len(out)+1)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

// opcionesEjemplo es el orden de las opciones de un ejemplo como en `jevbench.mensajes`: el dado, salvo en score, que va de menor a mayor nivel (sorted(key=int)).
func opcionesEjemplo(e EjemploFS) []string {
	ops := append([]string(nil), e.Opciones...)
	if e.Tipo == KindScore {
		sort.SliceStable(ops, func(i, j int) bool {
			a, ea := strconv.Atoi(ops[i])
			b, eb := strconv.Atoi(ops[j])
			return ea == nil && eb == nil && a < b
		})
	}
	return ops
}

// turnosEjemplosL3 es `turnos_ejemplos`: [usuario, asistente] por ejemplo.
func turnosEjemplosL3(ejs []EjemploFS) ([]msgL3, error) {
	var out []msgL3
	for _, e := range ejs {
		ops := opcionesEjemplo(e)
		p := Pregunta{Kind: e.Tipo}
		for _, o := range ops {
			p.Opciones = append(p.Opciones, Opcion{ID: o})
		}
		cods, err := codigosEtiquetaL3(p, p.Opciones)
		if err != nil {
			return nil, fmt.Errorf("ejemplo %s: %w", e.ID, err)
		}
		cod := ""
		for i, o := range ops {
			if o == e.Etiqueta {
				cod = cods[i]
			}
		}
		if cod == "" {
			return nil, fmt.Errorf("ejemplo %s: la etiqueta %q no es una de sus opciones", e.ID, e.Etiqueta)
		}
		var b strings.Builder
		b.WriteString(`{"pregunta": ` + jsonPyL3(e.Pregunta) + `, "tipo": ` + jsonPyL3(e.Tipo) + `, "opciones": [`)
		for i, o := range ops {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(`{"id": ` + jsonPyL3(o) + `, "etiqueta": ` + jsonPyL3(cods[i]) + "}")
		}
		txt := e.Texto
		if r := []rune(txt); len(r) > ejemploMaxChars {
			txt = string(r[:ejemploMaxChars])
		}
		b.WriteString(`], "texto": ` + jsonPyL3(delimitaTextoL3(txt)) + "}")
		out = append(out, msgL3{"user", b.String()}, msgL3{"assistant", cod})
	}
	return out, nil
}

// conEjemplosL3 es `con_ejemplos`: el sistema con la nota, los turnos y el caso nuevo.
func conEjemplosL3(msgs, turnos []msgL3) []msgL3 {
	out := make([]msgL3, 0, len(turnos)+2)
	out = append(out, msgL3{"system", msgs[0].Content + notaEjemplosL3})
	out = append(out, turnos...)
	return append(out, msgs[1])
}

// ConEjemplos activa el few-shot de L3: f da, para el texto de una pregunta, los ejemplos de oro de su tarea (nil = zero-shot). Los turnos se
// calculan una vez por pregunta y se cachean. Solo afecta al modo logprobs y a las preguntas que no son multi.
func (l *L3LLM) ConEjemplos(f func(pregunta string) []EjemploFS) *L3LLM {
	l.ejemplosDe = f
	return l
}

// turnosParaL3 devuelve los turnos fs de la pregunta (nil si no hay).
func (l *L3LLM) turnosParaL3(pregunta string) []msgL3 {
	if l.ejemplosDe == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if t, ok := l.turnosCache[pregunta]; ok {
		return t
	}
	var t []msgL3
	if ejs := l.ejemplosDe(pregunta); len(ejs) > 0 {
		var err error
		if t, err = turnosEjemplosL3(ejs); err != nil {
			t = nil // un ejemplo roto no tumba la fila: zero-shot (el arranque ya lo comprobó)
		}
	}
	if l.turnosCache == nil {
		l.turnosCache = map[string][]msgL3{}
	}
	l.turnosCache[pregunta] = t
	return t
}
