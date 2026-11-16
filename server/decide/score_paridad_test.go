package decide

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Paridad fila a fila con Python (R10-T3b, rub+cal). CHISPA_PARIDAD_SCORE = directorio que escribe `python -m model.alt.exportar_score` (score-<tarea>.json y
// topk-<tarea>.json); sin la variable se salta. Recalcula en Go los umbrales (AjustarUmbrales con los ejemplos de la rúbrica excluidos) y la esperanza y el nivel de
// cada fila del test, y exige: umbrales a 1e-6, esperanzas a 1e-9 y el 100 % de los niveles iguales.
type fixtureScore struct {
	Tarea     string    `json:"tarea"`
	Niveles   []string  `json:"niveles"`
	Umbrales  []float64 `json:"umbrales"`
	Excluidos []string  `json:"excluidos"`
	Soporte   []struct {
		ID       string             `json:"id"`
		Etiqueta string             `json:"etiqueta"`
		Valida   bool               `json:"valida"`
		Probs    map[string]float64 `json:"probs"`
	} `json:"soporte"`
	Filas []struct {
		ID        string             `json:"id"`
		Etiqueta  string             `json:"etiqueta"`
		Probs     map[string]float64 `json:"probs"`
		Esperanza *float64           `json:"esperanza"`
		Nivel     *string            `json:"nivel"`
	} `json:"filas"`
	NTest     int     `json:"n_test"`
	Exactitud float64 `json:"exactitud"`
}

func TestParidadScoreCalibradoConPython(t *testing.T) {
	dir := os.Getenv("CHISPA_PARIDAD_SCORE")
	if dir == "" {
		t.Skip("CHISPA_PARIDAD_SCORE sin fijar (directorio de model/alt/exportar_score.py)")
	}
	ficheros, _ := filepath.Glob(filepath.Join(dir, "score-*.json"))
	if len(ficheros) == 0 {
		t.Fatalf("no hay score-*.json en %s", dir)
	}
	for _, f := range ficheros {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var fx fixtureScore
		if err := json.Unmarshal(b, &fx); err != nil {
			t.Fatal(err)
		}
		t.Run(fx.Tarea, func(t *testing.T) {
			var sop []PrediccionSoporte
			for _, r := range fx.Soporte {
				sop = append(sop, PrediccionSoporte{ID: r.ID, Etiqueta: r.Etiqueta, Probs: r.Probs, Valida: r.Valida})
			}
			excl := map[string]bool{}
			for _, id := range fx.Excluidos {
				excl[id] = true
			}
			u, err := AjustarUmbrales(fx.Niveles, sop, excl)
			if err != nil {
				t.Fatal(err)
			}
			if len(u.Umbrales) != len(fx.Umbrales) {
				t.Fatalf("umbrales: %d en Go, %d en Python", len(u.Umbrales), len(fx.Umbrales))
			}
			var dU float64
			for i := range u.Umbrales {
				dU = math.Max(dU, math.Abs(u.Umbrales[i]-fx.Umbrales[i]))
			}
			if dU > 1e-6 {
				t.Errorf("umbrales Go %v, Python %v (dif máx %g)", u.Umbrales, fx.Umbrales, dU)
			}
			// Las filas se puntúan con los umbrales de Python (aislan el nivel de los umbrales) y con los de Go (de punta a punta, por el camino de L3).
			ord := make([]Opcion, len(fx.Niveles))
			for i, n := range fx.Niveles {
				ord[i] = Opcion{ID: n}
			}
			var dE float64
			var mal, aciertos, validas int
			for _, r := range fx.Filas {
				if r.Probs == nil {
					if r.Nivel != nil || r.Esperanza != nil {
						t.Fatalf("%s: sin probs y con nivel", r.ID)
					}
					continue
				}
				validas++
				p := make([]float64, len(fx.Niveles))
				for i, n := range fx.Niveles {
					p[i] = r.Probs[n]
				}
				e, ok := EsperanzaOrdinal(fx.Niveles, p)
				if r.Esperanza == nil {
					if ok {
						mal++
					}
					continue
				}
				if !ok {
					t.Fatalf("%s: Go sin esperanza y Python %v", r.ID, *r.Esperanza)
				}
				dE = math.Max(dE, math.Abs(e-*r.Esperanza))
				i1 := NivelPorUmbrales(e, fx.Umbrales)
				i2, ok2 := nivelCalibrado(ord, p, &u)
				if r.Nivel == nil || fx.Niveles[i1] != *r.Nivel || !ok2 || fx.Niveles[i2] != *r.Nivel {
					mal++
					if mal <= 5 {
						t.Logf("%s: e=%v Go(umbrales Py)=%s Go(umbrales Go)=%s Python=%v", r.ID, e, fx.Niveles[i1], fx.Niveles[i2], r.Nivel)
					}
					continue
				}
				if *r.Nivel == r.Etiqueta {
					aciertos++
				}
			}
			if dE > 1e-9 {
				t.Errorf("esperanza: dif máx %g", dE)
			}
			if mal != 0 {
				t.Errorf("%d filas con nivel distinto de Python", mal)
			}
			exact := float64(aciertos) / float64(fx.NTest)
			if math.Abs(exact-fx.Exactitud) > 5e-5 {
				t.Errorf("exactitud Go %.4f, Python %.4f", exact, fx.Exactitud)
			}
			t.Logf("PARIDAD %s: filas=%d válidas=%d desacuerdos=%d umbrales dif=%.2g esperanza dif=%.2g exactitud Go=%.4f Python=%.4f", fx.Tarea, len(fx.Filas), validas, mal, dU, dE, exact, fx.Exactitud)
		})
	}
}

// La preselección de Python (zs-topk) llevó k candidatas a L3 en cada fila. Sus vectores e5 no están en el fixture, así que el ranking en sí se prueba con datos sintéticos
// (TestRankTopKDesempate, TestL3TopKPorKNN); aquí solo se comprueba que el fixture trae k = L3TopKKNNPorDefecto candidatas y se informa el recall.
func TestParidadTopKFixture(t *testing.T) {
	dir := os.Getenv("CHISPA_PARIDAD_SCORE")
	if dir == "" {
		t.Skip("CHISPA_PARIDAD_SCORE sin fijar")
	}
	ficheros, _ := filepath.Glob(filepath.Join(dir, "topk-*.json"))
	for _, f := range ficheros {
		b, _ := os.ReadFile(f)
		var fx struct {
			Tarea string `json:"tarea"`
			K     int    `json:"k"`
			Filas []struct {
				Etiqueta   string   `json:"etiqueta"`
				Candidatas []string `json:"candidatas"`
			} `json:"filas"`
		}
		if err := json.Unmarshal(b, &fx); err != nil {
			t.Fatal(err)
		}
		if fx.K != L3TopKKNNPorDefecto {
			t.Errorf("%s: k=%d en Python y %d por defecto en Go", fx.Tarea, fx.K, L3TopKKNNPorDefecto)
		}
		var con, rec int
		for _, r := range fx.Filas {
			if r.Candidatas == nil {
				continue
			}
			con++
			if len(r.Candidatas) != fx.K {
				t.Fatalf("%s: una fila con %d candidatas", fx.Tarea, len(r.Candidatas))
			}
			for _, c := range r.Candidatas {
				if c == r.Etiqueta {
					rec++
					break
				}
			}
		}
		t.Logf("TOPK %s: k=%d filas=%d con candidatas=%d recall@k=%.4f (el ranking no se puede rehacer sin los vectores e5)", fx.Tarea, fx.K, len(fx.Filas), con, float64(rec)/float64(len(fx.Filas)))
	}
}
