package tok

import (
	"container/heap"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// BPE es el tokenizador BPE byte-level de Qwen2/Qwen3 (Qwen3-Embedding-0.6B) en Go puro: NFC → separación de los tokens añadidos → el
// pretokenizador regex de Qwen2 (a mano: RE2 no tiene mirada adelante) → bytes → fusiones por rango → plantilla de especiales y truncado,
// como `AutoTokenizer.from_pretrained(dir, padding_side="left")(texto, truncation=True, max_length=N)` de transformers 5.x.
// Lo carga el fichero que escribe exportar_bpe.py desde tokenizer.json (formato `chispa-bpe-1`, JSON determinista).
// Es seguro para uso concurrente.
type BPE struct {
	tokens         [][]byte // id → bytes (nil si el id no existe)
	porBytes       map[string]int32
	fusion         map[[2]int32]fusionBPE
	anadidos       []anadidoBPE // por longitud descendente
	plantilla      []itemPlantilla
	pad            int32
	nEsp           int // especiales que añade la plantilla
	byteID         [256]int32
	inicialAnadido [256]bool
}

type fusionBPE struct{ rango, id int32 }
type anadidoBPE struct {
	contenido string
	id        int32
}
type itemPlantilla struct {
	secuencia bool
	id        int32
}

// FormatoBPE es el valor de "formato" de los ficheros .chispa-bpe.
const FormatoBPE = "chispa-bpe-1"

// regexQwen2 es el patrón de pretokenización que implementa PretokenizarQwen2 (se comprueba contra tokenizer.json al exportar).
const RegexQwen2 = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type ficheroBPE struct {
	Formato   string   `json:"formato"`
	Normaliza string   `json:"normalizador"`
	Regex     string   `json:"regex"`
	Vocab     []string `json:"vocab"`    // hex de los bytes de cada token por id; "" = id sin token
	Fusiones  []int32  `json:"fusiones"` // pares (a, b) de ids, en orden de rango
	Anadidos  []struct {
		ID        int32  `json:"id"`
		Contenido string `json:"contenido"`
		Especial  bool   `json:"especial"`
	} `json:"anadidos"`
	Plantilla []struct {
		Secuencia bool  `json:"secuencia,omitempty"`
		ID        int32 `json:"id,omitempty"`
	} `json:"plantilla"`
	Pad int32 `json:"pad"`
}

// AbrirBPE lee un fichero .chispa-bpe.
func AbrirBPE(ruta string) (*BPE, error) {
	b, err := os.ReadFile(ruta)
	if err != nil {
		return nil, fmt.Errorf("bpe: %w", err)
	}
	t, err := BPEDesde(b)
	if err != nil {
		return nil, fmt.Errorf("bpe: %s: %w", ruta, err)
	}
	return t, nil
}

// BPEDesde interpreta el contenido de un .chispa-bpe.
func BPEDesde(data []byte) (*BPE, error) {
	var f ficheroBPE
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("no es un .chispa-bpe: %w", err)
	}
	if f.Formato != FormatoBPE {
		return nil, fmt.Errorf("formato %q (se esperaba %q)", f.Formato, FormatoBPE)
	}
	if f.Normaliza != "NFC" || f.Regex != "qwen2" {
		return nil, fmt.Errorf("normalizador %q y regex %q no soportados (solo NFC y qwen2)", f.Normaliza, f.Regex)
	}
	if len(f.Fusiones)%2 != 0 {
		return nil, errors.New("fusiones de longitud impar")
	}
	t := &BPE{tokens: make([][]byte, len(f.Vocab)), porBytes: make(map[string]int32, len(f.Vocab)), fusion: make(map[[2]int32]fusionBPE, len(f.Fusiones)/2), pad: f.Pad}
	for i := range t.byteID {
		t.byteID[i] = -1
	}
	for id, h := range f.Vocab {
		if h == "" {
			continue
		}
		b, err := hex.DecodeString(h)
		if err != nil {
			return nil, fmt.Errorf("vocab[%d]: %w", id, err)
		}
		t.tokens[id] = b
		if _, dup := t.porBytes[string(b)]; dup {
			return nil, fmt.Errorf("vocab[%d]: bytes repetidos", id)
		}
		t.porBytes[string(b)] = int32(id)
		if len(b) == 1 {
			t.byteID[b[0]] = int32(id)
		}
	}
	for i, v := range t.byteID {
		if v < 0 {
			return nil, fmt.Errorf("falta el token del byte %#x", i)
		}
	}
	for i := 0; i < len(f.Fusiones); i += 2 {
		a, b := f.Fusiones[i], f.Fusiones[i+1]
		if a < 0 || b < 0 || int(a) >= len(t.tokens) || int(b) >= len(t.tokens) || t.tokens[a] == nil || t.tokens[b] == nil {
			return nil, fmt.Errorf("fusión %d: ids fuera del vocabulario", i/2)
		}
		id, ok := t.porBytes[string(t.tokens[a])+string(t.tokens[b])]
		if !ok {
			return nil, fmt.Errorf("fusión %d: el resultado no está en el vocabulario", i/2)
		}
		k := [2]int32{a, b}
		if _, dup := t.fusion[k]; !dup {
			t.fusion[k] = fusionBPE{int32(i / 2), id}
		}
	}
	for _, a := range f.Anadidos {
		if a.Contenido == "" {
			return nil, errors.New("token añadido vacío")
		}
		t.anadidos = append(t.anadidos, anadidoBPE{a.Contenido, a.ID})
		t.inicialAnadido[a.Contenido[0]] = true
	}
	sort.SliceStable(t.anadidos, func(i, j int) bool { return len(t.anadidos[i].contenido) > len(t.anadidos[j].contenido) })
	for _, p := range f.Plantilla {
		t.plantilla = append(t.plantilla, itemPlantilla{p.Secuencia, p.ID})
		if !p.Secuencia {
			t.nEsp++
		}
	}
	if len(t.plantilla) == 0 {
		t.plantilla = []itemPlantilla{{secuencia: true}}
	}
	return t, nil
}

// PadID es el id de relleno (solo importa con lotes con relleno a la izquierda).
func (t *BPE) PadID() int32 { return t.pad }

// IDs tokeniza texto como transformers: con la plantilla de especiales y truncado a maxLen (≤ 0: sin truncar). El truncado recorta el cuerpo por la
// derecha para que cuerpo + especiales ≤ maxLen, como el de tokenizers con TemplateProcessing.
func (t *BPE) IDs(texto string, maxLen int) []int32 {
	hueco := -1
	if maxLen > 0 {
		hueco = max(maxLen-t.nEsp, 0)
	}
	cuerpo := t.cuerpo(strings.ToValidUTF8(texto, "�"), hueco)
	out := make([]int32, 0, len(cuerpo)+t.nEsp)
	for _, p := range t.plantilla {
		if p.secuencia {
			out = append(out, cuerpo...)
		} else {
			out = append(out, p.id)
		}
	}
	return out
}

// cuerpo son los ids sin especiales de plantilla, como mucho limite (< 0 = todos).
func (t *BPE) cuerpo(texto string, limite int) []int32 {
	var out []int32
	lleno := func() bool { return limite >= 0 && len(out) >= limite }
	texto1 := func(s string) {
		for _, p := range pretokenizarQwen2(NFC(s)) {
			if lleno() {
				return
			}
			out = t.bpe(out, []byte(p))
		}
	}
	ini := 0
	for i := 0; i < len(texto) && !lleno(); {
		var a *anadidoBPE
		if t.inicialAnadido[texto[i]] {
			for k := range t.anadidos {
				if strings.HasPrefix(texto[i:], t.anadidos[k].contenido) {
					a = &t.anadidos[k]
					break
				}
			}
		}
		if a == nil {
			i++
			continue
		}
		texto1(texto[ini:i])
		if !lleno() {
			out = append(out, a.id)
		}
		i += len(a.contenido)
		ini = i
	}
	if ini < len(texto) {
		texto1(texto[ini:])
	}
	if limite >= 0 && len(out) > limite {
		out = out[:limite]
	}
	return out
}

type parBPE struct {
	rango int32
	izq   int   // índice del símbolo izquierdo
	vIzq  int32 // versiones, para descartar entradas obsoletas
	vDer  int32
}
type colaBPE []parBPE

func (q colaBPE) Len() int { return len(q) }
func (q colaBPE) Less(i, j int) bool {
	if q[i].rango != q[j].rango {
		return q[i].rango < q[j].rango
	}
	return q[i].izq < q[j].izq
}
func (q colaBPE) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *colaBPE) Push(x any)   { *q = append(*q, x.(parBPE)) }
func (q *colaBPE) Pop() any {
	o := *q
	x := o[len(o)-1]
	*q = o[:len(o)-1]
	return x
}

// bpe añade a dst los ids de un trozo (bytes) tras aplicar las fusiones por rango (la menor primero; a igual rango, la más a la izquierda).
func (t *BPE) bpe(dst []int32, b []byte) []int32 {
	n := len(b)
	if n == 0 {
		return dst
	}
	if n == 1 {
		return append(dst, t.byteID[b[0]])
	}
	ids := make([]int32, n)
	ver := make([]int32, n)
	prev := make([]int, n)
	sig := make([]int, n)
	for i := range b {
		ids[i], prev[i], sig[i] = t.byteID[b[i]], i-1, i+1
	}
	sig[n-1] = -1
	var q colaBPE
	meter := func(i int) {
		j := sig[i]
		if j < 0 {
			return
		}
		if f, ok := t.fusion[[2]int32{ids[i], ids[j]}]; ok {
			heap.Push(&q, parBPE{f.rango, i, ver[i], ver[j]})
		}
	}
	for i := 0; i < n-1; i++ {
		meter(i)
	}
	for q.Len() > 0 {
		p := heap.Pop(&q).(parBPE)
		i := p.izq
		j := sig[i]
		if ids[i] < 0 || j < 0 || ver[i] != p.vIzq || ver[j] != p.vDer {
			continue
		}
		f := t.fusion[[2]int32{ids[i], ids[j]}]
		ids[i] = f.id
		ver[i]++
		ids[j] = -1
		sig[i] = sig[j]
		if sig[j] >= 0 {
			prev[sig[j]] = i
		}
		meter(i)
		if prev[i] >= 0 {
			meter(prev[i])
		}
	}
	for i := 0; i >= 0 && i < n; i = sig[i] {
		dst = append(dst, ids[i])
	}
	return dst
}

func esEspacio(r rune) bool { return unicode.IsSpace(r) } // White_Space de Unicode, como \s de Oniguruma

func esLetra(r rune) bool { return unicode.IsLetter(r) }
func esNum(r rune) bool   { return unicode.IsNumber(r) }

// pretokenizarQwen2 parte s como `re.findall(RegexQwen2, s)` (las piezas cubren todo el texto): las alternativas se prueban en orden y gana la primera.
// La mirada adelante de `\s+(?!\S)` se resuelve a mano: de una racha de n blancos, si no acaba el texto, la alternativa se queda con n−1 (y si n = 1, cae en `\s+`).
func pretokenizarQwen2(s string) []string {
	var out []string
	for i := 0; i < len(s); {
		out = append(out, s[i:i+largoQwen2(s[i:])])
		i += len(out[len(out)-1])
	}
	return out
}

func largoQwen2(s string) int {
	r0, w0 := utf8.DecodeRuneInString(s)
	// 1. contracciones
	if r0 == '\'' {
		if n := contraccion(s[1:]); n > 0 {
			return 1 + n
		}
	}
	// 2. [^\r\n\p{L}\p{N}]?\p{L}+
	{
		i := 0
		if r0 != '\r' && r0 != '\n' && !esLetra(r0) && !esNum(r0) {
			i = w0
		}
		j := i
		for j < len(s) {
			r, w := utf8.DecodeRuneInString(s[j:])
			if !esLetra(r) {
				break
			}
			j += w
		}
		if j > i {
			return j
		}
	}
	// 3. \p{N}
	if esNum(r0) {
		return w0
	}
	// 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
	{
		i := 0
		if r0 == ' ' {
			i = 1
		}
		j := i
		for j < len(s) {
			r, w := utf8.DecodeRuneInString(s[j:])
			if esEspacio(r) || esLetra(r) || esNum(r) {
				break
			}
			j += w
		}
		if j > i {
			for j < len(s) && (s[j] == '\r' || s[j] == '\n') {
				j++
			}
			return j
		}
	}
	// 5-7. rachas de blancos
	if esEspacio(r0) {
		fin, ultimoSalto, n := 0, -1, 0 // fin de la racha; fin del último \r o \n; número de blancos
		finPenultimo := 0
		for fin < len(s) {
			r, w := utf8.DecodeRuneInString(s[fin:])
			if !esEspacio(r) {
				break
			}
			finPenultimo = fin
			fin += w
			n++
			if r == '\r' || r == '\n' {
				ultimoSalto = fin
			}
		}
		if ultimoSalto > 0 { // \s*[\r\n]+
			return ultimoSalto
		}
		if fin == len(s) { // \s+(?!\S) con la racha hasta el final
			return fin
		}
		if n > 1 { // \s+(?!\S) cede el último blanco
			return finPenultimo
		}
		return fin // \s+
	}
	return w0 // no ocurre (todo carácter cae en alguna alternativa); evita bucles
}

// contraccion da el largo de «s|t|re|ve|m|ll|d» (sin distinguir mayúsculas) al comienzo de s, o 0.
func contraccion(s string) int {
	if len(s) == 0 {
		return 0
	}
	l := func(i int) byte {
		if i < len(s) {
			return s[i] | 0x20
		}
		return 0
	}
	switch l(0) {
	case 's', 't', 'm', 'd':
		return 1
	case 'r':
		if l(1) == 'e' {
			return 2
		}
	case 'v':
		if l(1) == 'e' {
			return 2
		}
	case 'l':
		if l(1) == 'l' {
			return 2
		}
	}
	return 0
}
