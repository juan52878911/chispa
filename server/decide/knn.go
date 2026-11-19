package decide

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"sync"
)

// kNN por tarea (L2 de la cascada L2→L3 de R8/R10). Es la regla de `model/alt/escalado.py:l2`, la referencia exacta:
//   - banco = ejemplos de oro (primero) ∪ fondo etiquetado por el LLM; vectores float32 YA normalizados (coseno = producto escalar);
//   - 32 vecinos por similitud (a igualdad, el de menor índice), pesos exp((s − smax)/0,05) con smax la similitud del mejor vecino;
//   - regla de umbral: si el oro más cercano tiene similitud ≥ 0,90, solo vota el oro (candidatos = el oro);
//   - votos solo entre las opciones de la fila; sin votos, uniforme; probs = votos normalizados;
//   - margen = p1 − p2.

const (
	KNNVecinosPorDefecto = 32
	// KNNVecinosQwen3Emb: vecinos por defecto con el codificador qwen3-emb (ver ConfigKNN.VecinosEfectivos).
	KNNVecinosQwen3Emb = 8
	KNNTempPorDefecto  = 0.05
	KNNSOroPorDefecto  = 0.90
)

// BancoKNN es el banco de una tarea. Formato en disco (el del exportador `model/alt/exportar_banco.py`): `<base>.npy` con la matriz
// float32 (n × dim, C-order, .npy v1/v2) y `<base>.jsonl` con una línea por fila, {"etiqueta": "...", "es_oro": true|false}, el oro primero.
type BancoKNN struct {
	Dim   int
	X     []float32 // n·Dim
	Etiq  []string
	NOro  int // los NOro primeros son oro
	norms int // vectores que hubo que normalizar al cargar

	centOnce sync.Once
	cent     map[string][]float64 // etiqueta → centroide normalizado de sus filas de oro (ver SimCentroides)
}

// N es el número de filas del banco.
func (b *BancoKNN) N() int { return len(b.Etiq) }

// CargarBancoKNN lee `<base>.npy` y `<base>.jsonl`.
func CargarBancoKNN(base string) (*BancoKNN, error) {
	x, n, dim, err := leerNpyF32(base + ".npy")
	if err != nil {
		return nil, err
	}
	f, err := os.Open(base + ".jsonl")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b := &BancoKNN{Dim: dim, X: x}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var r struct {
			Etiqueta string `json:"etiqueta"`
			EsOro    bool   `json:"es_oro"`
		}
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("%s.jsonl línea %d: %w", base, len(b.Etiq)+1, err)
		}
		if r.EsOro {
			if b.NOro != len(b.Etiq) {
				return nil, fmt.Errorf("%s.jsonl: el oro debe ir antes del fondo (línea %d)", base, len(b.Etiq)+1)
			}
			b.NOro++
		}
		b.Etiq = append(b.Etiq, r.Etiqueta)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(b.Etiq) != n {
		return nil, fmt.Errorf("%s: %d vectores y %d etiquetas", base, n, len(b.Etiq))
	}
	// «ya normalizados»: si algún vector no lo está, se normaliza al cargar (el coseno exige norma 1).
	for i := 0; i < n; i++ {
		v := x[i*dim : (i+1)*dim]
		var s float64
		for _, e := range v {
			s += float64(e) * float64(e)
		}
		if math.Abs(s-1) > 1e-3 && s > 0 {
			inv := float32(1 / math.Sqrt(s))
			for j := range v {
				v[j] *= inv
			}
			b.norms++
		}
	}
	return b, nil
}

var reNpy = regexp.MustCompile(`'(\w+)':\s*(\(.*?\)|'[^']*'|\w+)`)

// leerNpyF32 lee un .npy v1/v2/v3 float32 little-endian, C-order, 2-D (n × dim); un vector suelto cuenta como 1 × dim.
func leerNpyF32(ruta string) (x []float32, n, dim int, err error) {
	b, err := os.ReadFile(ruta)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(b) < 10 || string(b[:6]) != "\x93NUMPY" {
		return nil, 0, 0, fmt.Errorf("%s: no es un .npy", ruta)
	}
	var hl, off int
	switch b[6] {
	case 1:
		hl, off = int(binary.LittleEndian.Uint16(b[8:10])), 10
	case 2, 3:
		if len(b) < 12 {
			return nil, 0, 0, fmt.Errorf("%s: .npy truncado", ruta)
		}
		hl, off = int(binary.LittleEndian.Uint32(b[8:12])), 12
	default:
		return nil, 0, 0, fmt.Errorf("%s: versión de .npy %d no admitida", ruta, b[6])
	}
	if off+hl > len(b) {
		return nil, 0, 0, fmt.Errorf("%s: cabecera de .npy truncada", ruta)
	}
	campos := map[string]string{}
	for _, m := range reNpy.FindAllStringSubmatch(string(b[off:off+hl]), -1) {
		campos[m[1]] = m[2]
	}
	if campos["descr"] != "'<f4'" || campos["fortran_order"] != "False" {
		return nil, 0, 0, fmt.Errorf("%s: se espera float32 little-endian en orden C (descr %s, fortran_order %s)", ruta, campos["descr"], campos["fortran_order"])
	}
	var dims []int
	for _, t := range regexp.MustCompile(`\d+`).FindAllString(campos["shape"], -1) {
		d, _ := strconv.Atoi(t)
		dims = append(dims, d)
	}
	switch len(dims) {
	case 2:
		n, dim = dims[0], dims[1]
	case 1:
		n, dim = 1, dims[0]
	default:
		return nil, 0, 0, fmt.Errorf("%s: forma %s no admitida (se espera n × dim)", ruta, campos["shape"])
	}
	datos := b[off+hl:]
	if n < 0 || dim <= 0 || len(datos) != n*dim*4 {
		return nil, 0, 0, fmt.Errorf("%s: %d bytes de datos para %d×%d float32", ruta, len(datos), n, dim)
	}
	x = make([]float32, n*dim)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(datos[i*4:]))
	}
	return x, n, dim, nil
}

// ParamsKNN son los hiperparámetros del kNN (los de escalado.py por defecto).
type ParamsKNN struct {
	Vecinos int
	Temp    float64
	SOro    float64
}

func (p ParamsKNN) conDefectos() ParamsKNN {
	if p.Vecinos <= 0 {
		p.Vecinos = KNNVecinosPorDefecto
	}
	if p.Temp <= 0 {
		p.Temp = KNNTempPorDefecto
	}
	if p.SOro <= 0 {
		p.SOro = KNNSOroPorDefecto
	}
	return p
}

// SenalKNN son las señales de una consulta.
type SenalKNN struct {
	Margen    float64 // p1 − p2
	SimOro    float64 // similitud al oro más cercano (−1 sin oro)
	SimVecino float64 // similitud al vecino más cercano
	SoloOro   bool    // la regla de umbral dejó votar solo al oro
}

var errDimKNN = errors.New("kNN: la dimensión del vector de consulta no es la del banco")

// Votar es `escalado.l2` para UNA consulta q (normalizada) y las opciones ops de la fila: devuelve las probabilidades en el orden de ops.
func (b *BancoKNN) Votar(q []float32, ops []string, p ParamsKNN) ([]float64, SenalKNN, error) {
	p = p.conDefectos()
	if len(q) != b.Dim {
		return nil, SenalKNN{}, errDimKNN
	}
	n := b.N()
	s := make([]float32, n)
	for i := 0; i < n; i++ {
		row := b.X[i*b.Dim : (i+1)*b.Dim]
		var acc float32
		for j, e := range row {
			acc += e * q[j]
		}
		s[i] = acc
	}
	sen := SenalKNN{SimOro: -1, SimVecino: -1}
	if b.NOro > 0 {
		so := s[0]
		for _, v := range s[:b.NOro] {
			if v > so {
				so = v
			}
		}
		sen.SimOro = float64(so)
	}
	if n > 0 {
		m := s[0]
		for _, v := range s {
			if v > m {
				m = v
			}
		}
		sen.SimVecino = float64(m)
	}
	cand := n
	if b.NOro > 0 && sen.SimOro >= p.SOro {
		cand, sen.SoloOro = b.NOro, true
	}
	// top-K por similitud entre los candidatos [0, cand); a igualdad gana el de menor índice (argsort estable)
	k := p.Vecinos
	if k > cand {
		k = cand
	}
	top := make([]int, 0, k)
	for i := 0; k > 0 && i < cand; i++ {
		if len(top) == k && !(s[i] > s[top[k-1]]) {
			continue
		}
		pos := len(top)
		for pos > 0 && s[i] > s[top[pos-1]] {
			pos--
		}
		if len(top) < k {
			top = append(top, 0)
		}
		copy(top[pos+1:], top[pos:len(top)-1])
		top[pos] = i
	}
	votos := make(map[string]float64, len(ops))
	for _, o := range ops {
		votos[o] = 0
	}
	if len(top) > 0 {
		smax := s[top[0]]
		for _, j := range top {
			if _, ok := votos[b.Etiq[j]]; ok {
				votos[b.Etiq[j]] += math.Exp(float64(s[j]-smax) / p.Temp)
			}
		}
	}
	var tot float64
	for _, o := range ops {
		tot += votos[o]
	}
	probs := make([]float64, len(ops))
	for i, o := range ops {
		if tot > 0 {
			probs[i] = votos[o] / tot
		} else {
			probs[i] = 1 / float64(len(ops))
		}
	}
	p1, p2 := 0.0, 0.0
	for _, v := range probs {
		if v > p1 {
			p1, p2 = v, p1
		} else if v > p2 {
			p2 = v
		}
	}
	if len(probs) == 1 {
		p2 = 0
	}
	sen.Margen = p1 - p2
	return probs, sen, nil
}

// SimCentroides es la similitud (coseno) de q al centroide de oro de cada opción de ops, 0 si la clase no tiene oro: `C` de `fusion_knn_l2.knn`
// (centroide = media de los vectores de oro de la clase, normalizada con piso 1e-12). Sirve de desempate del ranking top-k.
func (b *BancoKNN) SimCentroides(q []float32, ops []string) ([]float64, error) {
	if len(q) != b.Dim {
		return nil, errDimKNN
	}
	b.centOnce.Do(func() {
		suma := map[string][]float64{}
		for i := 0; i < b.NOro; i++ {
			v := suma[b.Etiq[i]]
			if v == nil {
				v = make([]float64, b.Dim)
				suma[b.Etiq[i]] = v
			}
			for j, e := range b.X[i*b.Dim : (i+1)*b.Dim] {
				v[j] += float64(e)
			}
		}
		b.cent = make(map[string][]float64, len(suma))
		for et, v := range suma {
			var n float64
			for _, e := range v {
				n += e * e
			}
			inv := 1 / math.Max(math.Sqrt(n), 1e-12) // la media y su norma: el factor 1/n de la media se cancela salvo contra el piso, que no se alcanza con vectores unitarios
			for j := range v {
				v[j] *= inv
			}
			b.cent[et] = v
		}
	})
	out := make([]float64, len(ops))
	for i, o := range ops {
		if c, ok := b.cent[o]; ok {
			var acc float64
			for j, e := range c {
				acc += e * float64(q[j])
			}
			out[i] = acc
		}
	}
	return out, nil
}

// RankTopK devuelve los k índices de ops con mejor ranking según `probs` del kNN y, a igualdad, la similitud al centroide (descendente) y el orden de la fila
// (`np.lexsort((-C, -P))` de `lineas_base_v2_pred.cmd_topk`), devueltos en el orden de la fila. Con k ≥ len(ops) son todos.
func RankTopK(probs, cent []float64, k int) []int {
	n := len(probs)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		if probs[idx[a]] != probs[idx[b]] {
			return probs[idx[a]] > probs[idx[b]]
		}
		return cent[idx[a]] > cent[idx[b]]
	})
	if k > n {
		k = n
	}
	sel := append([]int(nil), idx[:k]...)
	sort.Ints(sel)
	return sel
}
