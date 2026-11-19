package decide

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Paridad de la elección: SeleccionarDemos da lo mismo que `lineas_base_v2_pred.demos_fila` (fixture de model/alt/w1_fixture_demos.py).
func TestSeleccionarDemosParidadPython(t *testing.T) {
	b, err := os.ReadFile("../testdata/w1_demos_knn.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx struct {
		Casos []struct {
			Nombre   string
			K        int
			Ys       []string
			X, Q     [][]float32
			Esperado [][]int
		}
	}
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	if len(fx.Casos) < 8 {
		t.Fatalf("fixture con %d casos", len(fx.Casos))
	}
	n := 0
	for _, c := range fx.Casos {
		for j, q := range c.Q {
			sim := make([]float32, len(c.X))
			for i, x := range c.X {
				sim[i] = simF32(q, x)
			}
			got := SeleccionarDemos(sim, c.Ys, c.K)
			if len(got) != len(c.Esperado[j]) {
				t.Fatalf("%s consulta %d: %v, Python %v", c.Nombre, j, got, c.Esperado[j])
			}
			for i := range got {
				if got[i] != c.Esperado[j][i] {
					t.Fatalf("%s consulta %d: %v, Python %v", c.Nombre, j, got, c.Esperado[j])
				}
			}
			n++
		}
	}
	t.Logf("%d selecciones iguales a Python", n)
}

func TestSeleccionarDemosBordes(t *testing.T) {
	if SeleccionarDemos(nil, nil, 8) != nil || SeleccionarDemos([]float32{1}, []string{"a"}, 0) != nil {
		t.Fatal("vacío")
	}
	// k = 2 con 3 clases: los 2 más cercanos, del más lejano al más cercano
	got := SeleccionarDemos([]float32{0.1, 0.9, 0.5, 0.7}, []string{"a", "b", "c", "a"}, 2)
	if len(got) != 2 || got[0] != 3 || got[1] != 1 {
		t.Fatalf("%v", got)
	}
	// con k clases o menos: una por clase aunque la clase b esté lejos
	got = SeleccionarDemos([]float32{0.9, 0.8, 0.7, 0.1}, []string{"a", "a", "a", "b"}, 3)
	if len(got) != 3 || got[0] != 3 || got[1] != 1 || got[2] != 0 {
		t.Fatalf("%v", got)
	}
}

// banco de prueba con ejemplos en formato pool (copia de banco-a: 8 de oro, x y z x y z x y).
func bancoPoolTmp(t *testing.T, conFs bool) string {
	t.Helper()
	d := t.TempDir()
	for _, ext := range []string{".npy", ".jsonl"} {
		b, err := os.ReadFile("testdata/knn/banco-a" + ext)
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(d, "banco-p"+ext), b, 0o644)
	}
	et := []string{"x", "y", "z", "x", "y", "z", "x", "y"}
	var sb strings.Builder
	for i, e := range et {
		l, _ := json.Marshal(map[string]any{"id": "e" + string(rune('0'+i)), "pregunta": "¿a?", "tipo": "choice", "texto": "ejemplo " + string(rune('0'+i)),
			"opciones": []string{"x", "y", "z"}, "etiqueta": e, "fila": i, "fs": map[bool]int{true: 3 - i, false: 0}[conFs && i < 3]})
		sb.Write(l)
		sb.WriteByte('\n')
	}
	os.WriteFile(filepath.Join(d, "banco-p.ejemplos.jsonl"), []byte(sb.String()), 0o644)
	return filepath.Join(d, "banco-p")
}

func TestL2KNNPoolFormatoViejoYNuevo(t *testing.T) {
	k, err := NuevoL2KNN(ConfigKNN{Tareas: map[string]TareaKNN{"p": {Banco: bancoPoolTmp(t, true), Preguntas: []string{"¿a?"}}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tv := k.tareas["p"]
	if len(tv.pool) != 8 || len(k.EjemplosDe("¿a?")) != 3 || k.EjemplosDe("¿a?")[0].ID != "e2" || len(k.Avisos) != 0 { // las fijas, en el orden de `fs`
		t.Fatalf("pool %d, fijas %d, avisos %v", len(tv.pool), len(k.EjemplosDe("¿a?")), k.Avisos)
	}
	// formato viejo (banco-a): sin pool, las 3 de siempre
	k2, err := NuevoL2KNN(ConfigKNN{Tareas: map[string]TareaKNN{"a": {Banco: "testdata/knn/banco-a", Preguntas: []string{"¿a?"}}}}, nil)
	if err != nil || k2.tareas["a"].pool != nil || len(k2.EjemplosDe("¿a?")) != 3 {
		t.Fatalf("formato viejo: %v", err)
	}
	// una etiqueta que no es la de su fila del banco no arranca
	d := bancoPoolTmp(t, true)
	b, _ := os.ReadFile(d + ".ejemplos.jsonl")
	os.WriteFile(d+".ejemplos.jsonl", []byte(strings.Replace(string(b), `"etiqueta":"y","fila":1`, `"etiqueta":"z","fila":1`, 1)), 0o644)
	if _, err := NuevoL2KNN(ConfigKNN{Tareas: map[string]TareaKNN{"p": {Banco: d}}}, nil); err == nil || !strings.Contains(err.Error(), "fila 1") {
		t.Fatalf("etiqueta cambiada: %v", err)
	}
}

// De punta a punta en modo gpu: L2 codifica la consulta aunque todo escale, elige las k más cercanas del pool y L3 las pone en el prompt, de la más lejana a la más cercana.
func TestW1DemosPorConsultaDelBancoLleganAlPrompt(t *testing.T) {
	cfg := ConfigKNN{Modo: ModoKNNGPU, Tareas: map[string]TareaKNN{"p": {Banco: bancoPoolTmp(t, true), Preguntas: []string{"¿a?"}}}}
	cod := &codHash{dim: dimBancoA(t)}
	k, err := NuevoL2KNN(cfg, cod)
	if err != nil {
		t.Fatal(err)
	}
	k.ConDemosKNN(3)
	b := k.tareas["p"].banco
	q := append([]float32(nil), b.X[4*b.Dim:5*b.Dim]...) // idéntica a la fila 4 (y)
	cod2 := codMapa{"caso": q}
	k.cod = cod2
	p := Pregunta{ID: "q", Estado: "caso", Texto: "¿a?", Kind: KindChoice, Opciones: []Opcion{{ID: "x"}, {ID: "y"}, {ID: "z"}}}
	r, err := k.Decidir(context.Background(), p)
	if err != nil || len(r.Demos) != 3 || r.Demos[2].ID != "e4" {
		t.Fatalf("demos %+v %v", r.Demos, err)
	}
	f := &lpFalso{top: map[string]float64{"A": -0.1, "B": -2}}
	l := nuevoLP(t, f, nil)
	if _, err := l.DecidirCon(context.Background(), p, Previas{"L2": r}); err != nil {
		t.Fatal(err)
	}
	ms := f.cuerpo(0)["messages"].([]any)
	if len(ms) != 2+6 || !strings.Contains(ms[len(ms)-3].(map[string]any)["content"].(string), "ejemplo 4") {
		t.Fatalf("%d mensajes", len(ms))
	}
	// sin ConDemosKNN el modo gpu sigue sin codificar y L3 usa las fijas
	k.ConDemosKNN(0)
	if r, _ := k.Decidir(context.Background(), p); r.Demos != nil || r.MotivoKNN != MotivoKNNSiempre {
		t.Fatalf("sin demos kNN: %+v", r)
	}
}

// Con examples en la petición, las demostraciones son las k más cercanas de ellos (una por clase si caben, y el resto por cercanía).
func TestW1DemosPorConsultaDeLosExamples(t *testing.T) {
	cod := codMapa{"caso": {1, 0}, "uno": {0, 1}, "dos": {0.6, 0.8}, "tres": {0.8, 0.6}, "cuatro": {-1, 0}}
	k, err := NuevoL2KNN(ConfigKNN{Modo: ModoKNNGPU, Tareas: map[string]TareaKNN{"a": {Banco: "testdata/knn/banco-a", Preguntas: []string{"¿a?"}}}}, cod)
	if err != nil {
		t.Fatal(err)
	}
	k.ConDemosKNN(2)
	p := Pregunta{ID: "q", Estado: "caso", Texto: "¿q?", Kind: KindChoice, Opciones: []Opcion{{ID: "x"}, {ID: "y"}, {ID: "z"}, {ID: "w"}},
		Ejemplos: []EjemploPeticion{{"uno", "x"}, {"dos", "x"}, {"tres", "x"}, {"cuatro", "y"}}}
	r, err := k.Decidir(context.Background(), p)
	if err != nil || len(r.Demos) != 2 || r.Demos[0].Texto != "cuatro" || r.Demos[1].Texto != "tres" {
		t.Fatalf("%+v %v", r.Demos, err)
	}
	if r.Demos[0].Pregunta != "¿q?" || len(r.Demos[0].Opciones) != 4 {
		t.Fatalf("%+v", r.Demos[0])
	}
}
