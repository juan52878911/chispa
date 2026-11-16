package decide

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
)

// Score calibrado (R10-T3b, variante rub+cal). La referencia exacta es Python: `lineas_base_v2.esperanza`/`orden`/`top1` y `lineas_base_v2b.pav`/`umbrales`.
//   - esperanza ordinal e = Σ nivel·p(nivel) / Σ p(nivel), con los niveles en orden numérico (1..K);
//   - umbrales: con las predicciones de L3 sobre el soporte de oro, la mediana de e de cada nivel, forzada a no decrecer con PAV; los K−1 umbrales son
//     los puntos medios entre niveles consecutivos, redondeados a 6 decimales;
//   - nivel predicho = 1 + nº de umbrales ESTRICTAMENTE por debajo de e (e > u).
// Los umbrales viven en el banco de la tarea, en `<banco>.umbrales.json`; L2KNN los carga junto al banco y L3 los usa en lugar de la mediana de las probs.
// Las filas que sirvieron de ejemplo de la rúbrica se excluyen del ajuste (`Excluir`). Rúbrica = las descripciones por nivel que el cliente manda en las opciones
// (con su ejemplo corto); el servidor ya las pasa al prompt de L3.

// MinOroPorNivel: el ajuste exige al menos este nº de filas de oro de cada nivel en el soporte (antes de excluir ejemplos de la rúbrica).
const MinOroPorNivel = 8

// UmbralesScore es el contenido de `<banco>.umbrales.json`.
type UmbralesScore struct {
	Version     int            `json:"version"`
	Niveles     []string       `json:"niveles"`                    // los niveles en orden numérico: len(umbrales) = len(niveles) − 1
	Umbrales    []float64      `json:"umbrales"`                   // K−1, no decrecientes
	OroPorNivel map[string]int `json:"oro_por_nivel,omitempty"`    // filas de oro del soporte por nivel
	Usadas      map[string]int `json:"usadas_por_nivel,omitempty"` // predicciones válidas que entraron al ajuste, por nivel
	Excluidas   []string       `json:"excluidas,omitempty"`        // ids excluidos (ejemplos de la rúbrica)
}

// PrediccionSoporte es la probabilidad por nivel que L3 dio a una fila de oro del soporte.
type PrediccionSoporte struct {
	ID       string
	Etiqueta string             // el nivel de oro
	Probs    map[string]float64 // por nivel; nil o valida=false = sin predicción
	Valida   bool
}

// nivelesOrdenados devuelve los niveles en orden numérico (`sorted(key=int)`); ok=false si alguno no es un entero.
func nivelesOrdenados(niv []string) ([]string, bool) {
	out := append([]string(nil), niv...)
	num := make(map[string]int, len(out))
	for _, n := range out {
		v, err := strconv.Atoi(n)
		if err != nil {
			return nil, false
		}
		num[n] = v
	}
	sort.SliceStable(out, func(i, j int) bool { return num[out[i]] < num[out[j]] })
	return out, true
}

// EsperanzaOrdinal es `lineas_base_v2.esperanza`: Σ nivel·p / Σ p, con p en el orden de niveles; ok=false si Σ p ≤ 0 o un nivel no es entero.
func EsperanzaOrdinal(niveles []string, p []float64) (float64, bool) {
	if len(niveles) != len(p) || len(p) == 0 {
		return 0, false
	}
	var num, den float64
	for i, o := range niveles {
		v, err := strconv.Atoi(o)
		if err != nil {
			return 0, false
		}
		num += float64(v) * p[i]
		den += p[i]
	}
	if !(den > 0) {
		return 0, false
	}
	return num / den, true
}

// PAV es `lineas_base_v2b.pav`: regresión isotónica no decreciente de mínimos cuadrados con pesos iguales.
func PAV(y []float64) []float64 {
	type bloque struct {
		m float64
		n int
	}
	var bs []bloque
	for _, v := range y {
		bs = append(bs, bloque{v, 1})
		for len(bs) > 1 && bs[len(bs)-2].m > bs[len(bs)-1].m {
			a, b := bs[len(bs)-2], bs[len(bs)-1]
			bs = append(bs[:len(bs)-2], bloque{(a.m*float64(a.n) + b.m*float64(b.n)) / float64(a.n+b.n), a.n + b.n})
		}
	}
	out := make([]float64, 0, len(y))
	for _, b := range bs {
		for i := 0; i < b.n; i++ {
			out = append(out, b.m)
		}
	}
	return out
}

func medianaF(xs []float64) float64 { // numpy.median
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func redondea6(x float64) float64 { // round(x, 6) de Python en los casos que importan; la paridad exigida es 1e-6
	r, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 6, 64), 64)
	return r
}

// AjustarUmbrales es `lineas_base_v2b.umbrales` con la regla del servidor: cada nivel necesita ≥ MinOroPorNivel filas de oro en el soporte, y tras excluir los ids
// de `excluir` y las predicciones inválidas todavía una predicción por nivel. `niveles` son los niveles de la tarea (en cualquier orden).
func AjustarUmbrales(niveles []string, sop []PrediccionSoporte, excluir map[string]bool) (UmbralesScore, error) {
	ord, ok := nivelesOrdenados(niveles)
	if !ok || len(ord) < 2 {
		return UmbralesScore{}, errors.New("umbrales: los niveles deben ser al menos dos enteros")
	}
	idx := make(map[string]int, len(ord))
	for i, o := range ord {
		idx[o] = i
	}
	oro := map[string]int{}
	por := make([][]float64, len(ord))
	for _, r := range sop {
		i, ok := idx[r.Etiqueta]
		if !ok {
			return UmbralesScore{}, fmt.Errorf("umbrales: la fila %q tiene la etiqueta %q, que no es un nivel", r.ID, r.Etiqueta)
		}
		oro[r.Etiqueta]++
		if excluir[r.ID] || !r.Valida || r.Probs == nil {
			continue
		}
		pr := make([]float64, len(ord))
		var suma float64
		for j, o := range ord {
			pr[j] = r.Probs[o]
			suma += pr[j]
		}
		if suma > 0 {
			e, _ := EsperanzaOrdinal(ord, pr)
			por[i] = append(por[i], e)
		}
	}
	for _, o := range ord {
		if oro[o] < MinOroPorNivel {
			return UmbralesScore{}, fmt.Errorf("umbrales: el nivel %s tiene %d filas de oro en el soporte y se necesitan al menos %d por nivel", o, oro[o], MinOroPorNivel)
		}
	}
	usadas := map[string]int{}
	med := make([]float64, len(ord))
	for i, o := range ord {
		if len(por[i]) == 0 {
			return UmbralesScore{}, fmt.Errorf("umbrales: ninguna predicción válida del soporte para el nivel %s (tras excluir los ejemplos de la rúbrica)", o)
		}
		usadas[o] = len(por[i])
		med[i] = medianaF(por[i])
	}
	med = PAV(med)
	u := make([]float64, len(ord)-1)
	for i := range u {
		u[i] = redondea6((med[i] + med[i+1]) / 2)
	}
	var ex []string
	for _, r := range sop {
		if excluir[r.ID] {
			ex = append(ex, r.ID)
		}
	}
	sort.Strings(ex)
	return UmbralesScore{Version: 1, Niveles: ord, Umbrales: u, OroPorNivel: oro, Usadas: usadas, Excluidas: ex}, nil
}

// NivelPorUmbrales es el nivel predicho: 1 + nº de umbrales por debajo de e (comparación estricta, e > u); devuelve el índice 0-based en el orden de niveles.
func NivelPorUmbrales(e float64, umbrales []float64) int {
	n := 0
	for _, u := range umbrales {
		if e > u {
			n++
		}
	}
	return n
}

func (u UmbralesScore) validar(niveles []string) error {
	if u.Version != 1 {
		return fmt.Errorf("versión %d no soportada", u.Version)
	}
	if len(u.Niveles) < 2 || len(u.Umbrales) != len(u.Niveles)-1 {
		return fmt.Errorf("%d niveles y %d umbrales (deben ser K y K−1)", len(u.Niveles), len(u.Umbrales))
	}
	for i, v := range u.Umbrales {
		if math.IsNaN(v) || math.IsInf(v, 0) || (i > 0 && v < u.Umbrales[i-1]) {
			return errors.New("los umbrales deben ser finitos y no decrecientes")
		}
	}
	if o, ok := nivelesOrdenados(u.Niveles); !ok || len(o) != len(u.Niveles) {
		return errors.New("los niveles deben ser enteros")
	}
	return nil
}

// RutaUmbrales es el fichero de umbrales de un banco.
func RutaUmbrales(base string) string { return base + ".umbrales.json" }

// CargarUmbralesScore lee `<base>.umbrales.json`; (nil, nil) si no existe.
func CargarUmbralesScore(base string) (*UmbralesScore, error) {
	b, err := os.ReadFile(RutaUmbrales(base))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var u UmbralesScore
	if err := json.Unmarshal(b, &u); err != nil {
		return nil, fmt.Errorf("%s: %w", RutaUmbrales(base), err)
	}
	if err := u.validar(nil); err != nil {
		return nil, fmt.Errorf("%s: %w", RutaUmbrales(base), err)
	}
	return &u, nil
}

// EscribirUmbralesScore escribe `<base>.umbrales.json`.
func EscribirUmbralesScore(base string, u UmbralesScore) error {
	if err := u.validar(nil); err != nil {
		return err
	}
	b, err := json.MarshalIndent(u, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(RutaUmbrales(base), append(b, '\n'), 0o644)
}

// nivelCalibrado devuelve el nivel (índice en `ordenadas`) por esperanza + umbrales, o ok=false si no se puede (sin umbrales, o no casan con los niveles de la pregunta).
func nivelCalibrado(ordenadas []Opcion, probs []float64, u *UmbralesScore) (int, bool) {
	if u == nil || len(u.Umbrales) != len(ordenadas)-1 || len(u.Niveles) != len(ordenadas) {
		return 0, false
	}
	ids := make([]string, len(ordenadas))
	for i, o := range ordenadas {
		ids[i] = o.ID
		if u.Niveles[i] != o.ID {
			return 0, false
		}
	}
	e, ok := EsperanzaOrdinal(ids, probs)
	if !ok {
		return 0, false
	}
	return NivelPorUmbrales(e, u.Umbrales), true
}
