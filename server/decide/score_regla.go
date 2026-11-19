package decide

import (
	"sort"
	"strconv"
)

// Regla de respuesta del camino kNN (L2), la de `model/alt/lineas_base_v2.top1(probs, r)` con esp=False, que es como se mide el kNN en Python:
//   - score: la mediana ordinal (CORAL) de las probabilidades del kNN, el primer nivel cuya cola acumulada (niveles por encima) no pasa de 0,5.
//     Los niveles se ordenan por su valor entero, no por el orden en que llegan en la petición.
//   - choice, noul y el resto: el argmax (en un empate, la primera opción).
//
// En dev (L2-qwen3emb, 6 tareas, 8 vecinos) esta regla da macro 0,4700 sobre las mismas `probs` con las que el argmax da 0,4609 (v8) y 0,4603 (v32);
// la diferencia solo está en las tareas score. `probs` siguen siendo las del kNN, sin tocar: solo cambia `answer`.

// RespuestaKNN devuelve la opción que contesta el kNN según la regla de arriba. probs va en el orden de ops. ok es false si Python daría None (las
// probabilidades no suman más de 0) o, en score, si algún id no es un entero (Python falla): quien llama se queda entonces con el argmax.
func RespuestaKNN(kind string, ops []string, probs []float64) (string, bool) {
	if len(ops) == 0 || len(ops) != len(probs) {
		return "", false
	}
	var suma float64
	for _, v := range probs { // sum() de Python: de izquierda a derecha
		suma += v
	}
	if !(suma > 0) {
		return "", false
	}
	if kind != KindScore {
		return ops[argmaxF(probs)], true
	}
	niv := make([]int, len(ops))
	for i, o := range ops {
		n, err := strconv.Atoi(o)
		if err != nil {
			return "", false
		}
		niv[i] = n
	}
	idx := make([]int, len(ops))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return niv[idx[a]] < niv[idx[b]] })
	// cola[j] = Σ_{i>j} q[i], acumulada de abajo arriba como np.cumsum(q[::-1])[::-1][1:] (mismo orden de sumas: sin diferencias de redondeo).
	n := len(idx)
	nivel := 0
	var cola float64
	for j := n - 1; j >= 1; j-- {
		cola += probs[idx[j]] / suma
		if cola > 0.5 {
			nivel++
		}
	}
	return ops[idx[nivel]], true
}

func argmaxF(p []float64) int {
	best := 0
	for i, v := range p {
		if v > p[best] {
			best = i
		}
	}
	return best
}
