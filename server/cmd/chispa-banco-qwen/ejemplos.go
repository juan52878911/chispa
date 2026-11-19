package main

// <banco>.ejemplos.jsonl: los k ejemplos de oro del few-shot (fs8) de L3, con el mismo formato y la MISMA elección que `model/alt/generar_banco.py` (que usa
// `llm_logprobs.elegir_ejemplos`: estratificados por clase con random.Random(semilla)). Para que el servidor lea el mismo fichero que con e5 sin torch, aquí está el
// Mersenne Twister (MT19937) de CPython con su `seed(int)`, `getrandbits` y `shuffle`; el test compara con ficheros que escribió Python.

import (
	"sort"
	"strings"
)

// pyRandom es random.Random(semilla) de CPython (solo lo que usa elegir_ejemplos).
type pyRandom struct {
	mt  [624]uint32
	pos int
}

func nuevoPyRandom(semilla uint64) *pyRandom {
	r := &pyRandom{}
	r.initGenrand(19650218)
	// random.seed(int): la clave son los trozos de 32 bits de abs(semilla), el menos significativo primero (al menos uno).
	key := []uint32{uint32(semilla)}
	if semilla>>32 != 0 {
		key = append(key, uint32(semilla>>32))
	}
	i, j := 1, 0
	k := 624
	if len(key) > k {
		k = len(key)
	}
	for ; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1] >> 30)) * 1664525)) + key[j] + uint32(j)
		i++
		j++
		if i >= 624 {
			r.mt[0] = r.mt[623]
			i = 1
		}
		if j >= len(key) {
			j = 0
		}
	}
	for k = 623; k > 0; k-- {
		r.mt[i] = (r.mt[i] ^ ((r.mt[i-1] ^ (r.mt[i-1] >> 30)) * 1566083941)) - uint32(i)
		i++
		if i >= 624 {
			r.mt[0] = r.mt[623]
			i = 1
		}
	}
	r.mt[0] = 0x80000000
	r.pos = 624
	return r
}

func (r *pyRandom) initGenrand(s uint32) {
	r.mt[0] = s
	for i := 1; i < 624; i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + uint32(i)
	}
	r.pos = 624
}

func (r *pyRandom) uint32() uint32 {
	if r.pos >= 624 {
		for i := 0; i < 624; i++ {
			y := (r.mt[i] & 0x80000000) | (r.mt[(i+1)%624] & 0x7fffffff)
			v := r.mt[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				v ^= 0x9908b0df
			}
			r.mt[i] = v
		}
		r.pos = 0
	}
	y := r.mt[r.pos]
	r.pos++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// randbelow es `Random._randbelow_with_getrandbits(n)` para n ≥ 1 (n < 2³²).
func (r *pyRandom) randbelow(n int) int {
	k := 0
	for v := n; v > 0; v >>= 1 {
		k++
	}
	for {
		x := int(r.uint32() >> (32 - uint(k)))
		if x < n {
			return x
		}
	}
}

// shuffle es `Random.shuffle`: para i de len-1 a 1, j = randbelow(i+1) e intercambio.
func pyShuffle[T any](r *pyRandom, x []T) {
	for i := len(x) - 1; i >= 1; i-- {
		j := r.randbelow(i + 1)
		x[i], x[j] = x[j], x[i]
	}
}

// elegirEjemplos es `llm_logprobs.elegir_ejemplos(soporte, k, semilla)`: solo filas de oro (split train o ausente), las clases barajadas, una por clase a la vez
// (al azar dentro de la clase, con los ids ordenados antes de barajar) hasta k, y el orden final barajado. Devuelve índices de `filas`.
func elegirEjemplos(filas []fila, k int, semilla uint64) []int {
	rng := nuevoPyRandom(semilla)
	por := map[string][]int{}
	for i, f := range filas {
		if f.Split != "" && f.Split != "train" {
			continue
		}
		por[f.Etiqueta] = append(por[f.Etiqueta], i)
	}
	cl := make([]string, 0, len(por))
	for c := range por {
		cl = append(cl, c)
	}
	sort.Strings(cl)
	pyShuffle(rng, cl)
	for _, c := range cl {
		xs := por[c]
		sort.SliceStable(xs, func(a, b int) bool { return strings.Compare(filas[xs[a]].ID, filas[xs[b]].ID) < 0 })
		pyShuffle(rng, xs)
	}
	var out []int
	for i := 0; len(out) < k && quedan(por); {
		c := cl[i%len(cl)]
		i++
		if xs := por[c]; len(xs) > 0 {
			out = append(out, xs[len(xs)-1])
			por[c] = xs[:len(xs)-1]
		}
	}
	pyShuffle(rng, out)
	return out
}

func quedan(por map[string][]int) bool {
	for _, v := range por {
		if len(v) > 0 {
			return true
		}
	}
	return false
}

// jsonPy es json.dumps(s, ensure_ascii=False) de Python para una cadena.
func jsonPy(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				b.WriteString(`\u00`)
				b.WriteByte("0123456789abcdef"[r>>4])
				b.WriteByte("0123456789abcdef"[r&15])
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

// lineaEjemplo es una línea de <banco>.ejemplos.jsonl como la escribe generar_banco.py: json.dumps({id, pregunta, tipo, texto, opciones, etiqueta}, ensure_ascii=False).
func lineaEjemplo(f fila) string {
	var b strings.Builder
	b.WriteString(`{"id": ` + jsonPy(f.ID) + `, "pregunta": ` + jsonPy(f.Pregunta) + `, "tipo": ` + jsonPy(f.Tipo) + `, "texto": ` + jsonPy(f.Texto) + `, "opciones": [`)
	for i, o := range f.Opciones {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(jsonPy(o))
	}
	b.WriteString(`], "etiqueta": ` + jsonPy(f.Etiqueta) + "}\n")
	return b.String()
}
