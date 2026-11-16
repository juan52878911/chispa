package decide

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Tests del endurecimiento de L3 (circuito, reintentos, cola acotada, cupo por lote, multi por logprobs, métricas).

// srvLP es un vLLM falso con control del estado HTTP: estado(n) devuelve el código de la llamada número n (1-based; 200 = contesta con
// top). Para multi, topPorOpcion decide el top según el id de la opción que viene en el prompt.
type srvLP struct {
	top      map[string]float64
	estado   func(n int64) (int, map[string]string)
	espera   time.Duration
	llamadas atomic.Int64
	enVuelo  atomic.Int64
	maxVuelo atomic.Int64
	mu       sync.Mutex
	cuerpos  []string
	porTexto func(user string) map[string]float64
}

func (f *srvLP) arranca(t *testing.T) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := f.llamadas.Add(1)
		v := f.enVuelo.Add(1)
		defer f.enVuelo.Add(-1)
		for {
			m := f.maxVuelo.Load()
			if v <= m || f.maxVuelo.CompareAndSwap(m, v) {
				break
			}
		}
		b, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.cuerpos = append(f.cuerpos, string(b))
		f.mu.Unlock()
		time.Sleep(f.espera)
		if f.estado != nil {
			if cod, cab := f.estado(n); cod != 200 {
				for k, v := range cab {
					w.Header().Set(k, v)
				}
				w.WriteHeader(cod)
				return
			}
		}
		top := f.top
		if f.porTexto != nil {
			var c struct {
				Messages []msgL3 `json:"messages"`
			}
			json.Unmarshal(b, &c)
			top = f.porTexto(c.Messages[len(c.Messages)-1].Content)
		}
		var tops []map[string]any
		for tok, lp := range top {
			tops = append(tops, map[string]any{"token": tok, "logprob": lp})
		}
		out, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"logprobs": map[string]any{"content": []any{map[string]any{"top_logprobs": tops}}}}}})
		w.Write(out)
	}))
	t.Cleanup(s.Close)
	return s
}

// l3Duro arma L3 en modo logprobs contra f con una espera registrada (sin dormir de verdad) y azar fijo.
func l3Duro(t *testing.T, f *srvLP, mod func(*ConfigL3)) (*L3LLM, *Metricas, *[]time.Duration) {
	t.Helper()
	s := f.arranca(t)
	cfg := ConfigL3{URL: s.URL, Modelo: "m", Modo: L3ModoLogprobs}
	if mod != nil {
		mod(&cfg)
	}
	l, err := NuevoL3LLM(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := NuevasMetricas()
	l.UsaMetricas(m)
	var esperas []time.Duration
	var mu sync.Mutex
	l.dormir = func(ctx context.Context, d time.Duration) error {
		mu.Lock()
		esperas = append(esperas, d)
		mu.Unlock()
		return ctx.Err()
	}
	l.azar = func() float64 { return 1 } // jitter máximo: la espera completa
	return l, m, &esperas
}

func metricaTiene(t *testing.T, m *Metricas, ss ...string) {
	t.Helper()
	txt := metricasTexto(m)
	for _, s := range ss {
		if !strings.Contains(txt, s) {
			t.Errorf("falta %q en /metrics", s)
		}
	}
}

func opcs2() Pregunta { return preguntaLP(2) }

// ---------- 1. cortacircuitos ----------

func TestL3CircuitoSeAbreRechazaSinLlamarYSemiabre(t *testing.T) {
	var sano atomic.Bool
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}, estado: func(int64) (int, map[string]string) {
		if sano.Load() {
			return 200, nil
		}
		return 500, nil
	}}
	l, m, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxIntentos = 1; c.CircuitoFallos = 3; c.CircuitoEspera = Duracion(time.Minute) })
	ahora := time.Now()
	l.circ.ahora = func() time.Time { return ahora }
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		if _, err := l.Decidir(ctx, opcs2()); !errors.Is(err, ErrSinL3) || errors.Is(err, ErrCircuitoAbierto) {
			t.Fatalf("fila %d: debía ser un fallo normal: %v", i, err)
		}
	}
	metricaTiene(t, m, "chispa_l3_circuito_estado 1", "chispa_l3_circuito_aperturas_total 1")
	antes := f.llamadas.Load()
	for i := 0; i < 5; i++ {
		if _, err := l.Decidir(ctx, opcs2()); !errors.Is(err, ErrCircuitoAbierto) || !errors.Is(err, ErrSinL3) {
			t.Fatalf("abierto: %v", err)
		}
	}
	if f.llamadas.Load() != antes {
		t.Fatalf("abierto no debe llamar al LLM: %d llamadas nuevas", f.llamadas.Load()-antes)
	}
	metricaTiene(t, m, `chispa_l3_descartes_total{causa="circuito_abierto"} 5`)
	// pasado el plazo: semiabierto deja pasar UNA fila de prueba; si falla, vuelve a abrir
	ahora = ahora.Add(2 * time.Minute)
	if _, err := l.Decidir(ctx, opcs2()); errors.Is(err, ErrCircuitoAbierto) || err == nil {
		t.Fatalf("la prueba debía llamar y fallar: %v", err)
	}
	if _, err := l.Decidir(ctx, opcs2()); !errors.Is(err, ErrCircuitoAbierto) {
		t.Fatalf("tras la prueba fallida vuelve a abrir: %v", err)
	}
	metricaTiene(t, m, "chispa_l3_circuito_aperturas_total 2")
	// el LLM se recupera: la prueba ok cierra
	sano.Store(true)
	ahora = ahora.Add(2 * time.Minute)
	if r, err := l.Decidir(ctx, opcs2()); err != nil || r.Probs == nil {
		t.Fatalf("la prueba ok: %v %+v", err, r)
	}
	metricaTiene(t, m, "chispa_l3_circuito_estado 0")
	for i := 0; i < 3; i++ {
		if _, err := l.Decidir(ctx, opcs2()); err != nil {
			t.Fatalf("cerrado de nuevo: %v", err)
		}
	}
}

func TestL3CircuitoSemiabiertoSoloUnaPrueba(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}, espera: 150 * time.Millisecond}
	l, _, _ := l3Duro(t, f, func(c *ConfigL3) { c.CircuitoFallos = 1 })
	ahora := time.Now()
	l.circ.ahora = func() time.Time { return ahora }
	l.circ.registra(false, false, false) // un fallo seguido: abre
	ahora = ahora.Add(time.Hour)
	var wg sync.WaitGroup
	var rechazadas, ok atomic.Int64
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := l.Decidir(context.Background(), opcs2()); errors.Is(err, ErrCircuitoAbierto) {
				rechazadas.Add(1)
			} else if err == nil {
				ok.Add(1)
			}
		}()
	}
	wg.Wait()
	if ok.Load() != 1 || rechazadas.Load() != 5 {
		t.Fatalf("una prueba y cinco rechazos: ok=%d rechazadas=%d", ok.Load(), rechazadas.Load())
	}
}

func TestL3CircuitoPorTasaEnVentana(t *testing.T) {
	var n atomic.Int64
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}, estado: func(int64) (int, map[string]string) {
		if n.Add(1)%2 == 1 { // fallo, éxito, fallo, éxito…: nunca 2 seguidos
			return 500, nil
		}
		return 200, nil
	}}
	l, m, _ := l3Duro(t, f, func(c *ConfigL3) {
		c.MaxIntentos = 1
		c.CircuitoFallos = 100
		c.CircuitoTasa = 0.5
		c.CircuitoVentana = 4
	})
	for i := 0; i < 4; i++ {
		l.Decidir(context.Background(), opcs2())
	}
	metricaTiene(t, m, "chispa_l3_circuito_estado 1")
	if _, err := l.Decidir(context.Background(), opcs2()); !errors.Is(err, ErrCircuitoAbierto) {
		t.Fatalf("2 de 4 fallos (50 %%) abren por tasa: %v", err)
	}
}

func TestL3CircuitoApagadoNoAbre(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1}, estado: func(int64) (int, map[string]string) { return 500, nil }}
	l, _, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxIntentos = 1; c.CircuitoFallos = 1; c.CircuitoApagado = true })
	for i := 0; i < 5; i++ {
		if _, err := l.Decidir(context.Background(), opcs2()); errors.Is(err, ErrCircuitoAbierto) {
			t.Fatal("apagado no abre")
		}
	}
	if f.llamadas.Load() != 5 {
		t.Fatal(f.llamadas.Load())
	}
}

func TestL3CircuitoNoCuentaInvalidaNiCancelada(t *testing.T) {
	f := &srvLP{top: map[string]float64{"zz": -1}} // sin etiquetas: inválida, pero el LLM contestó
	l, _, _ := l3Duro(t, f, func(c *ConfigL3) { c.CircuitoFallos = 2; c.MaxExtra = new(int) })
	for i := 0; i < 6; i++ {
		if r, err := l.Decidir(context.Background(), opcs2()); err != nil || !r.Incompleto {
			t.Fatalf("inválida sin error: %+v %v", r, err)
		}
	}
}

// ---------- 2. reintentos con espera exponencial, jitter y Retry-After ----------

func TestL3ReintentoEsperaExponencialConJitter(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}, estado: func(n int64) (int, map[string]string) {
		if n <= 3 {
			return 503, nil
		}
		return 200, nil
	}}
	l, m, esperas := l3Duro(t, f, func(c *ConfigL3) {
		c.MaxIntentos = 5
		c.EsperaBase = Duracion(100 * time.Millisecond)
		c.EsperaMax = Duracion(350 * time.Millisecond)
	})
	if _, err := l.Decidir(context.Background(), opcs2()); err != nil {
		t.Fatal(err)
	}
	want := []time.Duration{100 * time.Millisecond, 200 * time.Millisecond, 350 * time.Millisecond} // 100, 200, tope
	if fmt.Sprint(*esperas) != fmt.Sprint(want) || f.llamadas.Load() != 4 {
		t.Fatalf("esperas %v (querido %v) llamadas %d", *esperas, want, f.llamadas.Load())
	}
	metricaTiene(t, m, `chispa_l3_reintentos_total{causa="5xx"} 3`)
	// jitter: con azar=0 la espera es la mitad
	f2 := &srvLP{top: map[string]float64{"A": -1}, estado: func(n int64) (int, map[string]string) {
		if n == 1 {
			return 500, nil
		}
		return 200, nil
	}}
	l2, _, e2 := l3Duro(t, f2, func(c *ConfigL3) { c.EsperaBase = Duracion(100 * time.Millisecond) })
	l2.azar = func() float64 { return 0 }
	l2.Decidir(context.Background(), opcs2())
	if len(*e2) != 1 || (*e2)[0] != 50*time.Millisecond {
		t.Fatalf("jitter mínimo = la mitad: %v", *e2)
	}
}

func TestL3ReintentoTopeDeIntentos(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1}, estado: func(int64) (int, map[string]string) { return 502, nil }}
	l, _, esperas := l3Duro(t, f, func(c *ConfigL3) { c.MaxIntentos = 3; c.CircuitoApagado = true })
	if _, err := l.Decidir(context.Background(), opcs2()); !errors.Is(err, ErrSinL3) {
		t.Fatal(err)
	}
	if f.llamadas.Load() != 3 || len(*esperas) != 2 {
		t.Fatalf("3 intentos y 2 esperas: llamadas %d esperas %v", f.llamadas.Load(), *esperas)
	}
}

func TestL3ReintentoNoReintentaUn4xx(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1}, estado: func(int64) (int, map[string]string) { return 400, nil }}
	l, _, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxIntentos = 4 })
	l.Decidir(context.Background(), opcs2())
	if f.llamadas.Load() != 1 {
		t.Fatalf("400 no se reintenta: %d", f.llamadas.Load())
	}
}

func TestL3Reintento429RespetaRetryAfterSegundos(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}, estado: func(n int64) (int, map[string]string) {
		if n == 1 {
			return 429, map[string]string{"Retry-After": "2"}
		}
		return 200, nil
	}}
	l, m, esperas := l3Duro(t, f, func(c *ConfigL3) { c.TimeoutFila = Duracion(30 * time.Second) })
	if _, err := l.Decidir(context.Background(), opcs2()); err != nil {
		t.Fatal(err)
	}
	if len(*esperas) != 1 || (*esperas)[0] != 2*time.Second {
		t.Fatalf("espera = Retry-After: %v", *esperas)
	}
	metricaTiene(t, m, `chispa_l3_reintentos_total{causa="429"} 1`)
}

func TestL3Reintento429RetryAfterFechaHTTP(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1}, estado: func(n int64) (int, map[string]string) {
		if n == 1 {
			return 429, map[string]string{"Retry-After": time.Now().Add(5 * time.Second).UTC().Format(http.TimeFormat)}
		}
		return 200, nil
	}}
	l, _, esperas := l3Duro(t, f, func(c *ConfigL3) { c.TimeoutFila = Duracion(30 * time.Second) })
	if _, err := l.Decidir(context.Background(), opcs2()); err != nil {
		t.Fatal(err)
	}
	if len(*esperas) != 1 || (*esperas)[0] < 3*time.Second || (*esperas)[0] > 5*time.Second {
		t.Fatalf("fecha HTTP a +5 s: %v", *esperas)
	}
}

func TestL3Reintento429NoSuperaElTopeDeLaFila(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1}, estado: func(int64) (int, map[string]string) {
		return 429, map[string]string{"Retry-After": "60"}
	}}
	l, _, esperas := l3Duro(t, f, func(c *ConfigL3) { c.TimeoutFila = Duracion(2 * time.Second); c.MaxIntentos = 5 })
	t0 := time.Now()
	_, err := l.Decidir(context.Background(), opcs2())
	if !errors.Is(err, ErrSinL3) || len(*esperas) != 0 || f.llamadas.Load() != 1 || time.Since(t0) > time.Second {
		t.Fatalf("Retry-After 60 s con fila de 2 s: no se espera ni se reintenta: err=%v esperas=%v llamadas=%d", err, *esperas, f.llamadas.Load())
	}
	// y una espera exponencial que no cabe en lo que queda tampoco se hace
	f2 := &srvLP{top: map[string]float64{"A": -1}, estado: func(int64) (int, map[string]string) { return 500, nil }}
	l2, _, e2 := l3Duro(t, f2, func(c *ConfigL3) {
		c.TimeoutFila = Duracion(100 * time.Millisecond)
		c.EsperaBase = Duracion(time.Second)
		c.MaxIntentos = 5
	})
	l2.Decidir(context.Background(), opcs2())
	if len(*e2) != 0 || f2.llamadas.Load() != 1 {
		t.Fatalf("esperas que no caben: %v llamadas %d", *e2, f2.llamadas.Load())
	}
}

func TestParseRetryAfter(t *testing.T) {
	ahora := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for v, want := range map[string]time.Duration{"3": 3 * time.Second, "1.5": 1500 * time.Millisecond, "": 0, "-4": 0, "abc": 0, "999999": 0,
		ahora.Add(7 * time.Second).Format(http.TimeFormat): 7 * time.Second, ahora.Add(-time.Hour).Format(http.TimeFormat): 0} {
		if got := parseRetryAfter(v, ahora); got != want {
			t.Errorf("%q: %v, querido %v", v, got, want)
		}
	}
}

// ---------- 3. cola acotada con descarte ----------

func TestL3ColaAcotadaDescartaSinEsperar(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}, espera: 300 * time.Millisecond}
	uno := 1
	l, m, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxConcurrente = 1; c.MaxCola = &uno; c.TimeoutFila = Duracion(5 * time.Second) })
	var wg sync.WaitGroup
	res := make([]error, 3)
	ts := make([]time.Duration, 3)
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			t0 := time.Now()
			_, res[i] = l.Decidir(context.Background(), opcs2())
			ts[i] = time.Since(t0)
		}()
		time.Sleep(60 * time.Millisecond) // orden de llegada: 0 ocupa el hueco, 1 espera, 2 no cabe
	}
	wg.Wait()
	if res[0] != nil || res[1] != nil || !errors.Is(res[2], ErrColaLlena) || !errors.Is(res[2], ErrSinL3) {
		t.Fatalf("resultados %v", res)
	}
	if ts[2] > 100*time.Millisecond {
		t.Errorf("el descarte debía ser inmediato: %v", ts[2])
	}
	if f.llamadas.Load() != 2 {
		t.Errorf("la fila descartada no llama al LLM: %d", f.llamadas.Load())
	}
	metricaTiene(t, m, `chispa_l3_descartes_total{causa="cola_llena"} 1`, "chispa_l3_cola_longitud 0", "chispa_l3_semaforo_en_uso 0", "chispa_l3_semaforo_capacidad 1")
}

func TestL3SinColaDescartaEnCuantoNoHayHueco(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1}, espera: 200 * time.Millisecond}
	cero := 0
	l, _, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxConcurrente = 1; c.MaxCola = &cero })
	go l.Decidir(context.Background(), opcs2())
	time.Sleep(50 * time.Millisecond)
	if _, err := l.Decidir(context.Background(), opcs2()); !errors.Is(err, ErrColaLlena) {
		t.Fatal(err)
	}
}

func TestL3MetricasDeOcupacionYCola(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -1, "B": -2}, espera: 300 * time.Millisecond}
	l, m, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxConcurrente = 2 })
	for i := 0; i < 4; i++ {
		go l.Decidir(context.Background(), opcs2())
	}
	time.Sleep(120 * time.Millisecond)
	metricaTiene(t, m, "chispa_l3_semaforo_en_uso 2", "chispa_l3_semaforo_capacidad 2", "chispa_l3_cola_longitud 2")
	time.Sleep(900 * time.Millisecond)
	metricaTiene(t, m, "chispa_l3_semaforo_en_uso 0", "chispa_l3_cola_longitud 0")
}

// ---------- 4. cupo por lote ----------

// Causa del problema de la novena fila: max_por_peticion era un tope de 8 llamadas para TODA la petición HTTP (el presupuesto lo comparten
// sus preguntas, conPresupuesto) y reservar() devolvía false a partir de la 9.ª pregunta: «presupuesto» → Incompleto, sin L3.
func TestL3CupoProporcionalAlLoteDe32(t *testing.T) {
	post := func(l *L3LLM, m *Metricas, n int) []Resultado {
		casc, err := NuevaCascada([]Capa{{"L3", 0.5}}, []Decisor{l}, m)
		if err != nil {
			t.Fatal(err)
		}
		var qs []map[string]any
		for i := 0; i < n; i++ {
			qs = append(qs, map[string]any{"id": fmt.Sprintf("q%d", i), "question": "¿?", "kind": "choice", "options": []any{map[string]any{"id": "a"}, map[string]any{"id": "b"}}})
		}
		cuerpo, _ := json.Marshal(map[string]any{"state": "hola", "questions": qs})
		srv := httptest.NewServer(NuevoServidor(casc, m, 0).Handler())
		defer srv.Close()
		r, err := http.Post(srv.URL+"/v1/decide", "application/json", bytes.NewReader(cuerpo))
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		var out struct{ Results []Resultado }
		if err := json.NewDecoder(r.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out.Results
	}
	contestadas := func(rs []Resultado) (n int) {
		for _, r := range rs {
			if r.Layer == "L3" {
				n++
			}
		}
		return
	}
	// Con el cupo antiguo (tope fijo de 8) 24 de 32 filas se quedan sin L3.
	f := &srvLP{top: map[string]float64{"A": -0.1, "B": -3}}
	l, m, _ := l3Duro(t, f, func(c *ConfigL3) { c.MinPorPeticion = 8; c.MaxPorPeticion = 8 })
	if n := contestadas(post(l, m, 32)); n != 8 {
		t.Fatalf("con máximo 8 solo 8 de 32 filas tienen L3: %d", n)
	}
	// Con los valores por defecto el cupo es el lote (32 ≤ 256): las 32 contestan.
	f2 := &srvLP{top: map[string]float64{"A": -0.1, "B": -3}}
	l2, m2, _ := l3Duro(t, f2, nil)
	if n := contestadas(post(l2, m2, 32)); n != 32 || f2.llamadas.Load() < 32 {
		t.Fatalf("lote de 32: %d con L3, %d llamadas", n, f2.llamadas.Load())
	}
	if c := cuentaL3(m2, "presupuesto"); c != 0 {
		t.Fatalf("sin filas por presupuesto: %d", c)
	}
}

func TestL3CupoMinimoYMaximo(t *testing.T) {
	l, _, _ := l3Duro(t, &srvLP{}, func(c *ConfigL3) { c.MinPorPeticion = 4; c.MaxPorPeticion = 10 })
	for n, want := range map[int64]int{0: 4, 1: 4, 4: 4, 7: 7, 10: 10, 32: 10} {
		if got := l.cupo(n); got != want {
			t.Errorf("lote %d: cupo %d, querido %d", n, got, want)
		}
	}
	l2, _, _ := l3Duro(t, &srvLP{}, nil)
	if l2.cupo(0) != 8 || l2.cupo(100) != 100 || l2.cupo(1000) != 256 {
		t.Errorf("valores por defecto 8/256: %d %d %d", l2.cupo(0), l2.cupo(100), l2.cupo(1000))
	}
	l3, _, _ := l3Duro(t, &srvLP{}, func(c *ConfigL3) { c.MaxPorPeticion = 2 }) // máximo < mínimo por defecto: manda el máximo
	if l3.cupo(0) != 2 || l3.cupo(50) != 2 {
		t.Errorf("máximo 2: %d %d", l3.cupo(0), l3.cupo(50))
	}
}

// ---------- 5. multi por logprobs ----------

func TestL3MultiPorLogprobsSiNoIndependientes(t *testing.T) {
	// opción "uno": S muy probable; "dos": N muy probable; "tres": S y " S" suman (variante con espacio) y N falta → piso
	f := &srvLP{porTexto: func(u string) map[string]float64 {
		switch {
		case strings.Contains(u, `"id": "uno"`):
			return map[string]float64{"S": -0.1, "N": -2.5, "x": -9}
		case strings.Contains(u, `"id": "dos"`):
			return map[string]float64{"S": -3, "N": -0.05}
		}
		return map[string]float64{"S": -1, " S": -1, "y": -8}
	}}
	l, m, _ := l3Duro(t, f, nil)
	p := pregL3(KindMulti, "uno", "dos", "tres")
	r, err := l.Decidir(context.Background(), p)
	if err != nil || r.Incompleto {
		t.Fatal(err, r)
	}
	if f.llamadas.Load() != 3 {
		t.Errorf("una llamada por opción: %d", f.llamadas.Load())
	}
	sig := func(a, b float64) float64 { return 1 / (1 + math.Exp(b-a)) }
	cerca(t, r.Probs["uno"], sig(-0.1, -2.5), "p(uno)")
	cerca(t, r.Probs["dos"], sig(-3, -0.05), "p(dos)")
	sumaS := lseL3(-1, -1)
	cerca(t, r.Probs["tres"], sig(sumaS, -9), "p(tres): S+ S frente al piso (−8 − 1)")
	if fmt.Sprint(r.Answers) != "[uno tres]" || r.Answer != "tres" && r.Answer != "uno" {
		t.Errorf("answers %v answer %q", r.Answers, r.Answer)
	}
	var sum float64
	for _, v := range r.Probs {
		sum += v
	}
	if sum < 1.05 || sum > 2.9 { // independientes: no suman 1
		t.Errorf("multi no suma 1: %v", r.Probs)
	}
	metricaTiene(t, m, "chispa_l3_filas_parciales_total 1") // «tres» usó el piso
	// el prompt es el de multi (una opción por llamada, etiquetas S/N), no el de choice
	if !strings.Contains(f.cuerpos[0], "UNA sola opción") || !strings.Contains(f.cuerpos[0], `\"tipo\": \"multi\"`) || strings.Contains(f.cuerpos[0], `\"etiqueta\"`) {
		t.Errorf("prompt multi: %s", f.cuerpos[0])
	}
}

func TestL3MultiSinEtiquetasEsInvalidaYDemasiadasOpciones(t *testing.T) {
	f := &srvLP{top: map[string]float64{"zz": -1}}
	l, _, _ := l3Duro(t, f, nil)
	if r, err := l.Decidir(context.Background(), pregL3(KindMulti, "a", "b")); err != nil || !r.Incompleto {
		t.Fatalf("sin S ni N: inválida: %+v %v", r, err)
	}
	f2 := &srvLP{top: map[string]float64{"S": -1, "N": -2}}
	l2, _, _ := l3Duro(t, f2, func(c *ConfigL3) { c.MultiMaxOpciones = 3 })
	if r, err := l2.Decidir(context.Background(), pregL3(KindMulti, "a", "b", "c", "d")); err != nil || !r.Incompleto || f2.llamadas.Load() != 0 {
		t.Fatalf("4 opciones con máximo 3: sin llamar: %+v %v %d", r, err, f2.llamadas.Load())
	}
}

func TestL3MultiErrorDeRedEsSinL3(t *testing.T) {
	f := &srvLP{top: map[string]float64{"S": -1, "N": -2}, estado: func(int64) (int, map[string]string) { return 500, nil }}
	l, _, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxIntentos = 1 })
	if _, err := l.Decidir(context.Background(), pregL3(KindMulti, "a", "b")); !errors.Is(err, ErrSinL3) {
		t.Fatal(err)
	}
}

// ---------- 7. métricas: parciales con piso en choice ----------

func TestL3MetricaFilasParcialesYEnumeracion(t *testing.T) {
	f := &srvLP{top: map[string]float64{"A": -0.5, "B": -1}}
	l, m, _ := l3Duro(t, f, func(c *ConfigL3) { c.MaxExtra = new(int) })
	l.Decidir(context.Background(), preguntaLP(2)) // A y B en el top: completa
	l.Decidir(context.Background(), preguntaLP(3)) // C falta y max_extra = 0: parcial con piso
	metricaTiene(t, m, "chispa_l3_filas_parciales_total 1", `chispa_l3_reintentos_total{causa="red"} 0`, `chispa_l3_descartes_total{causa="cola_llena"} 0`,
		"chispa_l3_circuito_estado 0")
}
