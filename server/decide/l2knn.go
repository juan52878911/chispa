package decide

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

// L2KNN es la capa L2 de la cascada L2→L3 de R8/R10: un kNN por tarea (knn.go) sobre un banco de oro ∪ fondo etiquetado por el LLM.
// Sustituye al L2 de ONNX cuando la config trae la sección `knn` (ver ConfigKNN). La regla de escalado es la de Python
// (`exportar_banco.py`): escala a L3 si el margen p1 − p2 < u (global o por tarea), SIEMPRE si la tarea no tiene oro («sin_oro») o si
// la config la marca «siempre»; si no, contesta L2.
//
// El kNN necesita el vector e5 de la consulta (multilingual-e5-small, media de tokens normalizada, prefijo «query: », 256 tokens).
// El servidor NO tiene ese codificador: el ONNX del .chispa es el L2 ajustado (ranuras S), no e5 puro. Por eso el vector se inyecta: con
// ConVectorConsulta en el contexto de la petición o con un Codificador. Sin vector, L2KNN no opina (y la cascada sigue con L3).

// ConfigKNN es la sección `knn` de la config.
type ConfigKNN struct {
	// Tareas: nombre → banco y preguntas que atiende. Una pregunta se asigna a la tarea cuyo texto de `question` coincide exactamente.
	Tareas map[string]TareaKNN `json:"tareas"`
	// Umbral u global: se escala si margen < u. Por tarea, ver TareaKNN.Umbral. En Python, u = cuantil 0,25 del margen en calibración.
	Umbral float64 `json:"umbral"`
	// Siempre: tareas que escalan siempre a L3 (en Python, las de AUROC del margen < 0,55 en calibración). Las que no tienen oro escalan
	// siempre aunque no estén aquí (motivo «sin_oro»).
	Siempre     []string `json:"siempre,omitempty"`
	Vecinos     int      `json:"vecinos,omitempty"`     // 0 = 32
	Temperatura float64  `json:"temperatura,omitempty"` // 0 = 0,05
	SOro        float64  `json:"s_oro,omitempty"`       // 0 = 0,90: con el oro más cercano ≥ s_oro, vota solo el oro
	// ConservarL0L1: por defecto, con `knn` la cascada es L2(kNN)→L3 como en Python (L0 y L1 se saltan); con true se conservan.
	ConservarL0L1 bool `json:"conservar_l0_l1,omitempty"`
	// E5ONNX: ruta del ONNX de multilingual-e5-small sin ajustar (model/alt/exportar_e5_onnx.py, fp32 o int8). Con ella el servidor monta el
	// codificador (CodificadorE5, binario con -tags ort y flag -tok) y el kNN contesta por HTTP; sin ella, el vector se inyecta (ConVectorConsulta).
	E5ONNX string `json:"e5_onnx,omitempty"`
	// E5Sesiones: sesiones ORT del codificador (cada una carga su copia de los pesos, ≈ 470 MB en fp32); 0 = min(4, núcleos).
	E5Sesiones int `json:"e5_sesiones,omitempty"`
}

// TareaKNN describe una tarea.
type TareaKNN struct {
	Banco     string   `json:"banco"`               // ruta base: se leen <banco>.npy y <banco>.jsonl (ver BancoKNN)
	Preguntas []string `json:"preguntas,omitempty"` // textos de `question` que atiende; vacío = solo se usa con Evaluar (pruebas)
	Umbral    *float64 `json:"umbral,omitempty"`    // u de la tarea; ausente = el global
	Siempre   bool     `json:"siempre,omitempty"`
}

func (c ConfigKNN) validar() error {
	if len(c.Tareas) == 0 {
		return errors.New("config: knn.tareas no puede estar vacío")
	}
	if !(c.Umbral >= 0 && c.Umbral <= 1) {
		return fmt.Errorf("config: knn.umbral fuera de [0,1]: %v", c.Umbral)
	}
	if c.E5Sesiones < 0 {
		return errors.New("config: knn.e5_sesiones no puede ser negativo")
	}
	if c.Vecinos < 0 || c.Temperatura < 0 || c.SOro < 0 || c.SOro > 1 {
		return errors.New("config: knn.vecinos, temperatura y s_oro deben ser ≥ 0 (s_oro ≤ 1)")
	}
	vistas := map[string]string{}
	for n, t := range c.Tareas {
		if t.Banco == "" {
			return fmt.Errorf("config: knn.tareas[%q].banco vacío", n)
		}
		if t.Umbral != nil && !(*t.Umbral >= 0 && *t.Umbral <= 1) {
			return fmt.Errorf("config: knn.tareas[%q].umbral fuera de [0,1]: %v", n, *t.Umbral)
		}
		for _, q := range t.Preguntas {
			if o, ok := vistas[q]; ok {
				return fmt.Errorf("config: la pregunta %q está en las tareas %q y %q", q, o, n)
			}
			vistas[q] = n
		}
	}
	for _, s := range c.Siempre {
		if _, ok := c.Tareas[s]; !ok {
			return fmt.Errorf("config: knn.siempre nombra una tarea que no existe: %q", s)
		}
	}
	return nil
}

// Codificador da el vector e5 normalizado de un texto (ver el comentario del paquete).
type Codificador interface {
	Codificar(ctx context.Context, texto string) ([]float32, error)
}

type claveVectorKNN struct{}

// ConVectorConsulta inyecta el vector (normalizado) de la consulta de la petición: L2KNN lo usa en vez de llamar a un Codificador.
func ConVectorConsulta(ctx context.Context, v []float32) context.Context {
	return context.WithValue(ctx, claveVectorKNN{}, v)
}

// Motivos de escalado de L2KNN (también etiquetas de chispa_knn_escaladas_total).
const (
	MotivoKNNMargen     = "margen"     // margen < u
	MotivoKNNSinOro     = "sin_oro"    // la tarea no tiene ejemplos de oro
	MotivoKNNSiempre    = "siempre"    // la config marca la tarea «siempre»
	MotivoKNNIncompleta = "incompleta" // escalada a L3 y L3 falló (subconjunto de las anteriores)
)

type tareaKNNViva struct {
	ejemplos []EjemploFS // few-shot de L3 (<banco>.ejemplos.jsonl); nil = la tarea contesta zero-shot
	banco    *BancoKNN
	umbral   float64
	siempre  bool
}

// L2KNN: ver el comentario del fichero.
type L2KNN struct {
	p       ParamsKNN
	tareas  map[string]*tareaKNNViva
	porPreg map[string]string
	cod     Codificador
	met     *Metricas
	cfg     ConfigKNN
	Avisos  []string // cosas dudosas al cargar (tareas con oro y sin ejemplos fs)
}

// EjemplosDe da los ejemplos de oro del few-shot de L3 para una pregunta (la tarea cuyo texto de `question` coincide); nil si no hay.
func (k *L2KNN) EjemplosDe(pregunta string) []EjemploFS {
	if t, ok := k.porPreg[pregunta]; ok {
		return k.tareas[t].ejemplos
	}
	return nil
}

// ComprobarEjemplos falla si algún fichero de ejemplos trae más de n (el servidor no recorta: la elección es la del generador, que da la paridad con Python).
func (k *L2KNN) ComprobarEjemplos(n int) error {
	for _, nombre := range k.Tareas() {
		if e := k.tareas[nombre].ejemplos; len(e) > n {
			return fmt.Errorf("knn: tarea %q: el fichero de ejemplos trae %d y l3.ejemplos pide %d; regenere el banco con generar_banco.py --k %d", nombre, len(e), n, n)
		}
	}
	return nil
}

// NuevoL2KNN carga los bancos de la config. cod puede ser nil (solo vectores inyectados).
func NuevoL2KNN(cfg ConfigKNN, cod Codificador) (*L2KNN, error) {
	if err := cfg.validar(); err != nil {
		return nil, err
	}
	k := &L2KNN{p: ParamsKNN{Vecinos: cfg.Vecinos, Temp: cfg.Temperatura, SOro: cfg.SOro}.conDefectos(), tareas: map[string]*tareaKNNViva{}, porPreg: map[string]string{}, cod: cod, cfg: cfg}
	siempre := map[string]bool{}
	for _, s := range cfg.Siempre {
		siempre[s] = true
	}
	for n, t := range cfg.Tareas {
		b, err := CargarBancoKNN(t.Banco)
		if err != nil {
			return nil, fmt.Errorf("knn: tarea %q: %w", n, err)
		}
		ej, err := CargarEjemplosFS(t.Banco)
		if err != nil {
			return nil, fmt.Errorf("knn: tarea %q: %w", n, err)
		}
		if ej != nil && b.NOro == 0 {
			ej = nil // sin oro no hay few-shot: L3 contesta zero-shot
		}
		if b.NOro > 0 && ej == nil {
			k.Avisos = append(k.Avisos, fmt.Sprintf("tarea %q: tiene oro pero no %s.ejemplos.jsonl: L3 contestará zero-shot", n, t.Banco))
		}
		u := cfg.Umbral
		if t.Umbral != nil {
			u = *t.Umbral
		}
		k.tareas[n] = &tareaKNNViva{ejemplos: ej, banco: b, umbral: u, siempre: siempre[n] || t.Siempre}
		for _, q := range t.Preguntas {
			k.porPreg[q] = n
		}
	}
	return k, nil
}

func (k *L2KNN) Nombre() string { return "L2" }

// UsaMetricas conecta L2KNN con /metrics (chispa_knn_*); lo hace NuevaCascada.
func (k *L2KNN) UsaMetricas(m *Metricas) { k.met = m }

// Tareas devuelve los nombres de las tareas cargadas (ordenados).
func (k *L2KNN) Tareas() []string {
	var ns []string
	for n := range k.tareas {
		ns = append(ns, n)
	}
	sort.Strings(ns)
	return ns
}

// DecisionKNN es la decisión de L2KNN para una consulta.
type DecisionKNN struct {
	Probs  []float64 // en el orden de las opciones
	Answer string
	Sen    SenalKNN
	Escala bool
	Motivo string // margen, sin_oro, siempre o "" si no escala
	Umbral float64
}

// Evaluar decide una consulta de la tarea con el vector q (normalizado) y las opciones de la fila. No toca métricas.
func (k *L2KNN) Evaluar(tarea string, q []float32, ops []string) (DecisionKNN, error) {
	t := k.tareas[tarea]
	if t == nil {
		return DecisionKNN{}, fmt.Errorf("knn: tarea %q desconocida", tarea)
	}
	probs, sen, err := t.banco.Votar(q, ops, k.p)
	if err != nil {
		return DecisionKNN{}, err
	}
	d := DecisionKNN{Probs: probs, Sen: sen, Umbral: t.umbral}
	best := 0
	for i, v := range probs {
		if v > probs[best] {
			best = i
		}
	}
	if len(ops) > 0 {
		d.Answer = ops[best]
	}
	switch {
	case t.banco.NOro == 0:
		d.Escala, d.Motivo = true, MotivoKNNSinOro
	case t.siempre:
		d.Escala, d.Motivo = true, MotivoKNNSiempre
	case sen.Margen < t.umbral:
		d.Escala, d.Motivo = true, MotivoKNNMargen
	}
	return d, nil
}

func (k *L2KNN) Decidir(ctx context.Context, p Pregunta) (Respuesta, error) {
	tarea, ok := k.porPreg[p.Texto]
	if !ok || p.Kind == KindMulti || len(p.Opciones) == 0 {
		return Respuesta{}, nil // la pregunta no es de ninguna tarea con banco: no opina
	}
	q, _ := ctx.Value(claveVectorKNN{}).([]float32)
	pr := presupuestoDe(ctx)
	if q == nil && k.cod != nil {
		var err error
		t0 := time.Now()
		q, err = k.cod.Codificar(ctx, p.Estado)
		d := time.Since(t0)
		k.met.knnCodificador(d)
		pr.knnCodNs.Add(int64(d))
		if err != nil {
			return Respuesta{}, fmt.Errorf("knn: codificador: %w", err)
		}
	}
	if q == nil {
		k.met.knnSinVector()
		return Respuesta{}, nil // sin vector no hay kNN: sigue L3
	}
	ops := make([]string, len(p.Opciones))
	for i, o := range p.Opciones {
		ops[i] = o.ID
	}
	t1 := time.Now()
	d, err := k.Evaluar(tarea, q, ops)
	dv := time.Since(t1)
	k.met.knnVotacion(dv)
	pr.knnNs.Add(int64(dv))
	if err != nil {
		return Respuesta{}, err
	}
	k.met.knnFila(tarea, d.Motivo)
	probs := make(map[string]float64, len(ops))
	for i, o := range ops {
		probs[o] = d.Probs[i]
	}
	tau := d.Umbral
	if d.Escala {
		tau = math.Inf(1) // nunca se acepta: la cascada sigue con L3 y conserva esta respuesta como pista
	}
	return Respuesta{Answer: d.Answer, Probs: probs, Conf: math.Max(d.Sen.Margen, 1e-9), TauPropio: &tau, Tarea: tarea, MotivoKNN: d.Motivo}, nil
}
