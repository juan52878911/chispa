package decide

import (
	"context"
	"fmt"
	"sort"
)

// Demostraciones fs8 por consulta (modo gpu): las k del soporte más parecidas a ESTA consulta, como `lineas_base_v2_pred.demos_fila` (la elección de
// `lineas_base_v2_pred demos` / `lineas_base_v2d.py` y de la apertura sellada, `apertura_v2.py`, LEEME §17.24 D y §17.26). El soporte es el pool de
// `<banco>.ejemplos.jsonl` (el oro del banco, con el índice `fila` de su vector) o, si la pregunta trae `examples[]`, los examples de la petición.
//
// Regla (idéntica a Python, con un test de paridad contra un fixture que genera `model/alt/w1_fixture_demos.py`):
//   - similitud = coseno (producto escalar de vectores normalizados) de la consulta con cada ejemplo del soporte;
//   - se ordena por similitud descendente y, a igualdad, por posición en el soporte (orden estable);
//   - si el soporte tiene k clases o menos: el más cercano de cada clase (las clases en orden alfabético) y se completa hasta k por cercanía;
//   - si no: los k más cercanos;
//   - orden en el prompt: de menor a mayor similitud (el más parecido queda pegado a la pregunta; a igualdad, el orden de elección).
// Con un soporte de menos de k ejemplos salen todos.
// Los productos se acumulan en float64 y se redondean a float32 (la precisión de las similitudes de Python): dos vectores iguales empatan siempre.

// SeleccionarDemos devuelve los índices del soporte elegidos, ya en el orden del prompt. ys son las etiquetas del soporte y sim las similitudes.
func SeleccionarDemos(sim []float32, ys []string, k int) []int {
	if k <= 0 || len(sim) == 0 || len(sim) != len(ys) {
		return nil
	}
	orden := make([]int, len(sim))
	for i := range orden {
		orden[i] = i
	}
	sort.SliceStable(orden, func(a, b int) bool { return sim[orden[a]] > sim[orden[b]] })
	set := map[string]bool{}
	var clases []string
	for _, y := range ys {
		if !set[y] {
			set[y] = true
			clases = append(clases, y)
		}
	}
	var elegidos []int
	if len(clases) <= k {
		sort.Strings(clases)
		usado := map[int]bool{}
		for _, c := range clases {
			for _, i := range orden {
				if ys[i] == c {
					elegidos = append(elegidos, i)
					usado[i] = true
					break
				}
			}
		}
		for _, i := range orden {
			if len(elegidos) >= k {
				break
			}
			if !usado[i] {
				elegidos = append(elegidos, i)
			}
		}
	} else {
		elegidos = append(elegidos, orden[:k]...)
	}
	sort.SliceStable(elegidos, func(a, b int) bool { return sim[elegidos[a]] < sim[elegidos[b]] })
	return elegidos
}

// simF32 es el producto escalar de dos vectores (float64, redondeado a float32).
func simF32(a, b []float32) float32 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return float32(s)
}

// poolDeBanco comprueba un fichero de ejemplos en formato pool y lo da como tal; (nil, nil) si es el formato viejo (ninguna fila trae `fila`).
func poolDeBanco(ej []EjemploFS, b *BancoKNN) ([]EjemploFS, error) {
	n := 0
	for _, e := range ej {
		if e.Fila != nil {
			n++
		}
	}
	if n == 0 {
		return nil, nil
	}
	if n != len(ej) {
		return nil, fmt.Errorf("ejemplos.jsonl mezcla filas con y sin `fila`")
	}
	vistas := map[int]bool{}
	for i, e := range ej {
		f := *e.Fila
		if f < 0 || f >= b.NOro {
			return nil, fmt.Errorf("ejemplos.jsonl línea %d: fila %d fuera del oro del banco (%d filas)", i+1, f, b.NOro)
		}
		if vistas[f] {
			return nil, fmt.Errorf("ejemplos.jsonl línea %d: fila %d repetida", i+1, f)
		}
		vistas[f] = true
		if b.Etiq[f] != e.Etiqueta {
			return nil, fmt.Errorf("ejemplos.jsonl línea %d: la etiqueta %q no es la de la fila %d del banco (%q)", i+1, e.Etiqueta, f, b.Etiq[f])
		}
	}
	return ej, nil
}

// fijasDePool son los ejemplos del modo fijo de un fichero pool: los marcados `fs` (la elección del generador), en el orden que dice `fs`.
func fijasDePool(pool []EjemploFS) []EjemploFS {
	var out []EjemploFS
	for _, e := range pool {
		if e.Fs > 0 {
			out = append(out, e)
		}
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].Fs < out[b].Fs })
	return out
}

// ConDemosKNN activa las demostraciones por consulta (k > 0; modo gpu): L2 codifica la consulta aunque todo escale y deja en Respuesta.Demos las k más cercanas.
func (k *L2KNN) ConDemosKNN(n int) *L2KNN {
	k.demosK = n
	return k
}

// demosConsulta elige las demostraciones de una consulta con vector q. soporte y vecs son los ejemplos y sus vectores (una fila de la matriz por ejemplo).
func (k *L2KNN) demosConsulta(q []float32, soporte []EjemploFS, vec func(i int) []float32) []EjemploFS {
	sim := make([]float32, len(soporte))
	ys := make([]string, len(soporte))
	for i, e := range soporte {
		v := vec(i)
		if len(v) != len(q) {
			return nil
		}
		sim[i], ys[i] = simF32(q, v), e.Etiqueta
	}
	var out []EjemploFS
	for _, i := range SeleccionarDemos(sim, ys, k.demosK) {
		out = append(out, soporte[i])
	}
	return out
}

// demosDeTarea: el soporte es el pool del banco de la tarea.
func (k *L2KNN) demosDeTarea(t *tareaKNNViva, q []float32) []EjemploFS {
	d := t.banco.Dim
	return k.demosConsulta(q, t.pool, func(i int) []float32 { f := *t.pool[i].Fila; return t.banco.X[f*d : (f+1)*d] })
}

// demosDePeticionKNN: el soporte son los examples de la petición, con los vectores del banco que se montó con ellos.
func (k *L2KNN) demosDePeticionKNN(p Pregunta, b *BancoKNN, q []float32) []EjemploFS {
	return k.demosConsulta(q, ejemplosDePeticion(p), func(i int) []float32 { return b.X[i*b.Dim : (i+1)*b.Dim] })
}

type claveDemosKNN struct{}

// conDemosDeL2 pasa a L3 las demostraciones por consulta que dejó L2 (si las hay) por el contexto: así las ven también las llamadas con el top-k de opciones.
func conDemosDeL2(ctx context.Context, previas Previas) context.Context {
	if r, ok := previas["L2"]; ok && len(r.Demos) > 0 {
		return context.WithValue(ctx, claveDemosKNN{}, r.Demos)
	}
	return ctx
}
