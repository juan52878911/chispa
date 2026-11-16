package decide

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestPAV(t *testing.T) {
	for _, c := range []struct{ in, want []float64 }{
		{[]float64{1, 2, 3}, []float64{1, 2, 3}},
		{[]float64{1, 3, 2, 4}, []float64{1, 2.5, 2.5, 4}},
		{[]float64{3, 2, 1}, []float64{2, 2, 2}},
		{[]float64{2, 1, 3, 2}, []float64{1.5, 1.5, 2.5, 2.5}},
		{[]float64{5}, []float64{5}},
	} {
		if got := PAV(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("PAV(%v) = %v, quería %v", c.in, got, c.want)
		}
	}
}

// predNivel da una predicción que concentra la masa en `nivel` (y algo en el vecino de arriba, para que la esperanza no sea entera).
func predNivel(id, etiqueta string, nivel int, extra float64) PrediccionSoporte {
	p := map[string]float64{"1": 0.001, "2": 0.001, "3": 0.001}
	p[fmt.Sprint(nivel)] = 1 - extra
	if nivel < 3 {
		p[fmt.Sprint(nivel+1)] = extra
	}
	return PrediccionSoporte{ID: id, Etiqueta: etiqueta, Probs: p, Valida: true}
}

func soporte3(n int) []PrediccionSoporte {
	var sop []PrediccionSoporte
	for nivel := 1; nivel <= 3; nivel++ {
		for i := 0; i < n; i++ {
			sop = append(sop, predNivel(fmt.Sprintf("n%d-%d", nivel, i), fmt.Sprint(nivel), nivel, 0.1))
		}
	}
	return sop
}

func TestAjustarUmbralesMedianaPAVyPuntosMedios(t *testing.T) {
	niv := []string{"3", "1", "2"} // el orden de entrada no importa
	u, err := AjustarUmbrales(niv, soporte3(8), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(u.Niveles, []string{"1", "2", "3"}) || len(u.Umbrales) != 2 {
		t.Fatalf("%+v", u)
	}
	// esperanza de cada fila del nivel k: (k·0,9 + (k+1)·0,1 + resto) / suma; con 0,001 en los demás niveles
	e := func(nivel int) float64 {
		v, _ := EsperanzaOrdinal([]string{"1", "2", "3"}, predNivel("", "", nivel, 0.1).probsEnOrden())
		return v
	}
	if want := redondea6((e(1) + e(2)) / 2); math.Abs(u.Umbrales[0]-want) > 1e-9 {
		t.Errorf("u0 = %v, quería %v", u.Umbrales[0], want)
	}
	if want := redondea6((e(2) + e(3)) / 2); math.Abs(u.Umbrales[1]-want) > 1e-9 {
		t.Errorf("u1 = %v, quería %v", u.Umbrales[1], want)
	}
	if u.Usadas["1"] != 8 || u.OroPorNivel["3"] != 8 {
		t.Errorf("recuentos: %+v", u)
	}
}

func (p PrediccionSoporte) probsEnOrden() []float64 {
	return []float64{p.Probs["1"], p.Probs["2"], p.Probs["3"]}
}

func TestAjustarUmbralesFuerzaMonotonia(t *testing.T) {
	// El nivel 2 «parece» más bajo que el 1: la mediana por nivel decrece y PAV las junta, así que el umbral 1-2 es la mitad de la masa común.
	var sop []PrediccionSoporte
	for i := 0; i < 8; i++ {
		sop = append(sop, predNivel(fmt.Sprintf("a%d", i), "1", 2, 0.0)) // etiqueta 1, pero L3 dijo 2
		sop = append(sop, predNivel(fmt.Sprintf("b%d", i), "2", 1, 0.0)) // etiqueta 2, pero L3 dijo 1
		sop = append(sop, predNivel(fmt.Sprintf("c%d", i), "3", 3, 0.0))
	}
	u, err := AjustarUmbrales([]string{"1", "2", "3"}, sop, nil)
	if err != nil {
		t.Fatal(err)
	}
	if u.Umbrales[0] < 1.4 || u.Umbrales[0] > 1.6 || u.Umbrales[1] < u.Umbrales[0] {
		t.Errorf("umbrales no monótonos o fuera de lo esperado: %v", u.Umbrales)
	}
	for i := 1; i < len(u.Umbrales); i++ {
		if u.Umbrales[i] < u.Umbrales[i-1] {
			t.Errorf("decrecen: %v", u.Umbrales)
		}
	}
}

func TestAjustarUmbralesExigeOchoPorNivel(t *testing.T) {
	_, err := AjustarUmbrales([]string{"1", "2", "3"}, soporte3(7), nil)
	if err == nil || !strings.Contains(err.Error(), "al menos 8") || !strings.Contains(err.Error(), "nivel 1") {
		t.Fatalf("con 7 por nivel debía fallar con un mensaje claro: %v", err)
	}
	sop := soporte3(8)
	sop = sop[:len(sop)-1] // al nivel 3 le falta una
	if _, err := AjustarUmbrales([]string{"1", "2", "3"}, sop, nil); err == nil || !strings.Contains(err.Error(), "nivel 3 tiene 7") {
		t.Fatalf("al nivel 3 le faltaba una: %v", err)
	}
	if _, err := AjustarUmbrales([]string{"1", "2", "3"}, soporte3(8), nil); err != nil {
		t.Fatalf("con 8 por nivel debía ajustar: %v", err)
	}
	if _, err := AjustarUmbrales([]string{"a", "b"}, nil, nil); err == nil {
		t.Error("niveles no enteros")
	}
}

func TestAjustarUmbralesExcluyeEjemplosDeLaRubrica(t *testing.T) {
	sop := soporte3(8)
	// una fila del nivel 1 con una predicción disparatada: excluida no mueve la mediana, y la regla de 8 cuenta el oro ANTES de excluir.
	sop[0] = predNivel("n1-0", "1", 3, 0)
	base, _ := AjustarUmbrales([]string{"1", "2", "3"}, soporte3(8), nil)
	conRuido, _ := AjustarUmbrales([]string{"1", "2", "3"}, sop, nil)
	excl, err := AjustarUmbrales([]string{"1", "2", "3"}, sop, map[string]bool{"n1-0": true})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(excl.Umbrales, base.Umbrales) || !reflect.DeepEqual(excl.Excluidas, []string{"n1-0"}) || excl.Usadas["1"] != 7 || excl.OroPorNivel["1"] != 8 {
		t.Errorf("excluida: %+v, base %+v", excl, base)
	}
	if reflect.DeepEqual(conRuido.Umbrales, base.Umbrales) && conRuido.Usadas["1"] == 8 {
		// la mediana de 8 filas con una atípica puede no moverse; lo que importa es que `excl` use 7
		t.Log("la fila atípica no movió la mediana")
	}
	// Si tras excluir un nivel se queda sin predicciones, falla.
	todos := map[string]bool{}
	for _, r := range soporte3(8)[:8] {
		todos[r.ID] = true
	}
	if _, err := AjustarUmbrales([]string{"1", "2", "3"}, soporte3(8), todos); err == nil || !strings.Contains(err.Error(), "ninguna predicción válida") {
		t.Errorf("nivel vacío tras excluir: %v", err)
	}
	// Predicciones inválidas no cuentan.
	inv := soporte3(8)
	for i := 8; i < 16; i++ {
		inv[i].Valida = false
	}
	if _, err := AjustarUmbrales([]string{"1", "2", "3"}, inv, nil); err == nil {
		t.Error("el nivel 2 solo tiene predicciones inválidas")
	}
}

func TestNivelPorUmbralesEsEstricto(t *testing.T) {
	u := []float64{1.5, 2.5}
	for _, c := range []struct {
		e    float64
		want int
	}{{1, 0}, {1.5, 0}, {1.5000001, 1}, {2.5, 1}, {2.6, 2}, {9, 2}} {
		if got := NivelPorUmbrales(c.e, u); got != c.want {
			t.Errorf("e=%v: %d, quería %d", c.e, got, c.want)
		}
	}
}

func TestUmbralesEnElBanco(t *testing.T) {
	base := filepath.Join(t.TempDir(), "banco-t")
	if u, err := CargarUmbralesScore(base); u != nil || err != nil {
		t.Fatalf("sin fichero: %v %v", u, err)
	}
	u, _ := AjustarUmbrales([]string{"1", "2", "3"}, soporte3(8), nil)
	if err := EscribirUmbralesScore(base, u); err != nil {
		t.Fatal(err)
	}
	got, err := CargarUmbralesScore(base)
	if err != nil || !reflect.DeepEqual(got.Umbrales, u.Umbrales) || !reflect.DeepEqual(got.Niveles, u.Niveles) {
		t.Fatalf("ida y vuelta: %+v %v", got, err)
	}
	mal := UmbralesScore{Version: 1, Niveles: []string{"1", "2", "3"}, Umbrales: []float64{2, 1}}
	if err := EscribirUmbralesScore(base, mal); err == nil {
		t.Error("umbrales decrecientes")
	}
	if err := EscribirUmbralesScore(base, UmbralesScore{Version: 1, Niveles: []string{"1", "2", "3"}, Umbrales: []float64{1}}); err == nil {
		t.Error("faltan umbrales")
	}
}

func TestNivelCalibrado(t *testing.T) {
	ord := []Opcion{{ID: "1"}, {ID: "2"}, {ID: "3"}}
	u := &UmbralesScore{Version: 1, Niveles: []string{"1", "2", "3"}, Umbrales: []float64{1, 2.6}}
	if i, ok := nivelCalibrado(ord, []float64{0.0486, 0.3592, 0.5922}, u); !ok || i != 1 {
		t.Errorf("e≈2,54 con umbrales [1, 2,6] debía dar el nivel 2: %d %v", i, ok)
	}
	if _, ok := nivelCalibrado(ord, []float64{0.2, 0.3, 0.5}, nil); ok {
		t.Error("sin umbrales")
	}
	if _, ok := nivelCalibrado(ord[:2], []float64{0.5, 0.5}, u); ok {
		t.Error("umbrales de otra escala")
	}
	if _, ok := nivelCalibrado(ord, []float64{0, 0, 0}, u); ok {
		t.Error("suma 0")
	}
}

// L3 en modo logprobs: con umbrales, la respuesta sale de la esperanza y no de la mediana; sin ellos, igual que siempre.
func TestL3ScoreCalibradoConUmbrales(t *testing.T) {
	f := &lpFalso{top: map[string]float64{"1": -3, "2": -1, "3": -0.5}} // p ≈ 0,049 / 0,359 / 0,592: mediana = nivel 3; esperanza ≈ 2,54
	l := nuevoLP(t, f, nil)
	p := Pregunta{ID: "q", Texto: "nivel", Kind: KindScore, Estado: "x", Opciones: []Opcion{{ID: "3"}, {ID: "1"}, {ID: "2"}}}
	r, err := l.Decidir(context.Background(), p)
	if err != nil || r.Answer != "3" {
		t.Fatalf("sin umbrales (mediana): %+v %v", r, err)
	}
	u := &UmbralesScore{Version: 1, Niveles: []string{"1", "2", "3"}, Umbrales: []float64{1, 2.6}}
	l.ConUmbrales(func(pregunta string) *UmbralesScore {
		if pregunta == "nivel" {
			return u
		}
		return nil
	})
	r, err = l.Decidir(context.Background(), p)
	if err != nil || r.Answer != "2" || math.Abs(r.Conf-r.Probs["2"]) > 1e-12 {
		t.Fatalf("con umbrales [1, 2,6]: %+v %v", r, err)
	}
	p.Texto = "otra"
	if r, _ = l.Decidir(context.Background(), p); r.Answer != "3" {
		t.Errorf("pregunta sin umbrales: %q", r.Answer)
	}
}

// ---------------------------------------------------------------- top-k por kNN

func TestRankTopKDesempate(t *testing.T) {
	probs := []float64{0.1, 0.4, 0.1, 0.1, 0.4, 0.0}
	cent := []float64{0.2, 0.1, 0.9, 0.5, 0.3, 0.99}
	// P desc: {1,4} (0,4; el 4 tiene centroide mayor), luego {0,2,3} (0,1; por centroide 2, 3, 0), luego 5.
	if got := RankTopK(probs, cent, 3); !reflect.DeepEqual(got, []int{1, 2, 4}) {
		t.Errorf("k=3: %v", got)
	}
	if got := RankTopK(probs, cent, 4); !reflect.DeepEqual(got, []int{1, 2, 3, 4}) {
		t.Errorf("k=4 (en el orden de la fila): %v", got)
	}
	// empate total: gana el orden de la fila (lexsort estable)
	if got := RankTopK([]float64{0.5, 0.5, 0.5}, []float64{0, 0, 0}, 2); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Errorf("empate total: %v", got)
	}
	if got := RankTopK(probs, cent, 99); len(got) != 6 {
		t.Errorf("k > n: %v", got)
	}
}

func TestSimCentroides(t *testing.T) {
	b := &BancoKNN{Dim: 2, X: []float32{1, 0, 0, 1, 0, 1, 0.6, 0.8}, Etiq: []string{"a", "b", "b", "c"}, NOro: 3} // la fila 3 es fondo: no cuenta
	c, err := b.SimCentroides([]float32{1, 0}, []string{"a", "b", "c", "z"})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(c[0]-1) > 1e-12 || math.Abs(c[1]) > 1e-12 || c[2] != 0 || c[3] != 0 {
		t.Errorf("centroides: %v", c)
	}
	if _, err := b.SimCentroides([]float32{1}, nil); err == nil {
		t.Error("dimensión")
	}
}

func opcionesTK(n int) []string {
	ids := make([]string, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("o%02d", i)
	}
	return ids
}

func respuestaKNNTK(ids []string, p map[string]float64, c map[string]float64) Previas {
	probs := make(map[string]float64, len(ids))
	var rank map[string]float64
	if c != nil {
		rank = make(map[string]float64, len(ids)) // L2KNN da una similitud por opción (0 sin oro)
	}
	for _, id := range ids {
		probs[id] = p[id]
		if c != nil {
			rank[id] = c[id]
		}
	}
	return Previas{"L2": {Answer: ids[0], Probs: probs, Conf: 0.1, RankKNN: rank}}
}

func TestL3TopKPorKNN(t *testing.T) {
	k := 4
	f := &llmFalso{contenido: `{"probs":{"o05":0.1,"o17":0.2,"o22":0.6,"o30":0.1}}`}
	l, _ := nuevoL3Test(t, f.arranca(t), func(c *ConfigL3) { c.TopK, c.TopKKNN = L3TopKSinRecorte, &k })
	ids := opcionesTK(40)
	p := pregL3(KindChoice, ids...)
	pr := map[string]float64{"o05": 0.3, "o17": 0.3, "o30": 0.2}
	for _, id := range ids {
		if _, ok := pr[id]; !ok {
			pr[id] = 0.2 / 37 // 37 opciones empatadas por probabilidad: manda el centroide
		}
	}
	cent := map[string]float64{"o22": 0.9, "o03": 0.8}
	r, err := l.DecidirCon(context.Background(), p, respuestaKNNTK(ids, pr, cent))
	if err != nil || r.Answer != "o22" || len(r.Probs) != 40 || r.Probs["o00"] != 0 || math.Abs(r.Probs["o22"]-0.6) > 1e-9 {
		t.Fatalf("r=%+v err=%v", r, err)
	}
	var e entradaL3
	pe := f.ultima.Load().(peticionL3)
	if err := json.Unmarshal([]byte(pe.Messages[1].Content), &e); err != nil {
		t.Fatal(err)
	}
	var vistas []string
	for _, o := range e.Opciones {
		vistas = append(vistas, o.ID)
	}
	if !reflect.DeepEqual(vistas, []string{"o05", "o17", "o22", "o30"}) { // por P: o05, o17, o30; el cuarto, el del centroide más alto (o22); orden de la pregunta
		t.Fatalf("L3 vio %v", vistas)
	}
	// La etiqueta es siempre una de las preseleccionadas: si el LLM nombra otra, la respuesta no es válida.
	f.contenido = `{"probs":{"o05":0.1,"o17":0.2,"o22":0.6,"o01":0.1}}`
	if r, _ := l.DecidirCon(context.Background(), p, respuestaKNNTK(ids, pr, cent)); r.Answer != "" && r.Answer == "o01" {
		t.Errorf("la respuesta salió de fuera del top-k: %+v", r)
	}
}

func TestL3TopKPorKNNNoAplica(t *testing.T) {
	k := 10
	f := &llmFalso{contenido: `{"probs":{"o00":1}}`}
	l, _ := nuevoL3Test(t, f.arranca(t), func(c *ConfigL3) { c.TopK, c.TopKKNN = L3TopKSinRecorte, &k })
	// 20 opciones: no son «más de 20»
	ids := opcionesTK(20)
	pr := map[string]float64{}
	if _, err := l.DecidirCon(context.Background(), pregL3(KindChoice, ids...), respuestaKNNTK(ids, pr, map[string]float64{})); err != nil {
		t.Fatal(err)
	}
	if pe := f.ultima.Load().(peticionL3); pe.MaxTokens != 16+24*20 {
		t.Errorf("con 20 opciones L3 debía ver las 20: max_tokens %d", pe.MaxTokens)
	}
	// 30 opciones sin RankKNN (la capa previa no es el kNN): todas
	ids = opcionesTK(30)
	prev := respuestaKNNTK(ids, pr, nil)
	if _, err := l.DecidirCon(context.Background(), pregL3(KindChoice, ids...), prev); err != nil {
		t.Fatal(err)
	}
	if pe := f.ultima.Load().(peticionL3); pe.MaxTokens != 16+24*30 {
		t.Errorf("sin RankKNN debía ver las 30: max_tokens %d", pe.MaxTokens)
	}
	// apagado con 0
	cero := 0
	l0, _ := nuevoL3Test(t, f.arranca(t), func(c *ConfigL3) { c.TopK, c.TopKKNN = L3TopKSinRecorte, &cero })
	prev = respuestaKNNTK(ids, pr, map[string]float64{})
	if _, err := l0.DecidirCon(context.Background(), pregL3(KindChoice, ids...), prev); err != nil {
		t.Fatal(err)
	}
	if pe := f.ultima.Load().(peticionL3); pe.MaxTokens != 16+24*30 {
		t.Errorf("top_k_knn=0 debía ver las 30: max_tokens %d", pe.MaxTokens)
	}
	for _, v := range []int{-1, 1} {
		if err := (ConfigL3{URL: "http://127.0.0.1:1/v1", Modelo: "m", TopKKNN: &v}).validar(); err == nil {
			t.Errorf("top_k_knn=%d debía ser inválido", v)
		}
	}
}

func TestL3TopKKNNPorDefectoConKNN(t *testing.T) {
	get := func(knn bool, topK int, kk *int) *int {
		c := Config{L3: &ConfigL3{URL: "http://127.0.0.1:1/v1", Modelo: "m", TopK: topK, TopKKNN: kk}}
		if knn {
			c.KNN = &ConfigKNN{}
		}
		d, err := c.DecisorL3()
		if err != nil {
			t.Fatal(err)
		}
		return d.(*L3LLM).cfg.TopKKNN
	}
	if k := get(true, 0, nil); k == nil || *k != 10 {
		t.Errorf("con knn: %v", k)
	}
	if k := get(false, 0, nil); k != nil {
		t.Errorf("sin knn: %v", *k)
	}
	if k := get(true, 5, nil); k != nil {
		t.Errorf("con l3.top_k explícito manda este: %v", *k)
	}
	siete := 7
	if k := get(true, 0, &siete); k == nil || *k != 7 {
		t.Errorf("explícito: %v", k)
	}
}

// L2KNN carga los umbrales junto al banco; score_umbrales=false los ignora y =true los exige.
func TestL2KNNCargaUmbrales(t *testing.T) {
	d := t.TempDir()
	base := filepath.Join(d, "banco-s")
	if err := os.WriteFile(base+".jsonl", []byte("{\"etiqueta\":\"1\",\"es_oro\":true}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(base+".npy", npyF32Test(1, 2, []float32{1, 0}), 0o644); err != nil {
		t.Fatal(err)
	}
	u, _ := AjustarUmbrales([]string{"1", "2", "3"}, soporte3(8), nil)
	mk := func(su *bool) (*L2KNN, error) {
		return NuevoL2KNN(ConfigKNN{Tareas: map[string]TareaKNN{"s": {Banco: base, Preguntas: []string{"¿nivel?"}, ScoreUmbrales: su}}}, nil)
	}
	if _, err := mk(boolP(true)); err == nil || !strings.Contains(err.Error(), "score_umbrales=true") {
		t.Fatalf("exigidos y ausentes: %v", err)
	}
	k, err := mk(nil)
	if err != nil || k.UmbralesDe("¿nivel?") != nil {
		t.Fatalf("sin fichero: %v", err)
	}
	if err := EscribirUmbralesScore(base, u); err != nil {
		t.Fatal(err)
	}
	if k, err = mk(nil); err != nil || k.UmbralesDe("¿nivel?") == nil || k.UmbralesDe("otra") != nil {
		t.Fatalf("con fichero: %v", err)
	}
	if k, err = mk(boolP(false)); err != nil || k.UmbralesDe("¿nivel?") != nil {
		t.Fatalf("ignorados: %v", err)
	}
}

func boolP(b bool) *bool { return &b }

// npyF32Test escribe un .npy v1 float32 (n × dim, C-order).
func npyF32Test(n, dim int, x []float32) []byte {
	h := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", n, dim)
	for (10+len(h)+1)%64 != 0 {
		h += " "
	}
	h += "\n"
	b := append([]byte("\x93NUMPY\x01\x00"), byte(len(h)), byte(len(h)>>8))
	b = append(b, h...)
	for _, v := range x {
		u := math.Float32bits(v)
		b = append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	return b
}
