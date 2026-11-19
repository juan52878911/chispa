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
	Vecinos     int      `json:"vecinos,omitempty"`     // 0 = 32 con e5-small y 8 con qwen3-emb (VecinosEfectivos)
	Temperatura float64  `json:"temperatura,omitempty"` // 0 = 0,05
	SOro        float64  `json:"s_oro,omitempty"`       // 0 = 0,90: con el oro más cercano ≥ s_oro, vota solo el oro
	// ConservarL0L1: por defecto, con `knn` la cascada es L2(kNN)→L3 como en Python (L0 y L1 se saltan); con true se conservan.
	ConservarL0L1 bool `json:"conservar_l0_l1,omitempty"`
	// E5ONNX: ruta del ONNX de multilingual-e5-small sin ajustar (model/alt/exportar_e5_onnx.py, fp32 o int8). Con ella el servidor monta el
	// codificador (CodificadorE5, binario con -tags ort y flag -tok) y el kNN contesta por HTTP; sin ella, el vector se inyecta (ConVectorConsulta).
	E5ONNX string `json:"e5_onnx,omitempty"`
	// E5Sesiones: sesiones ORT del codificador (cada una carga su copia de los pesos, ≈ 470 MB en fp32); 0 = min(4, núcleos).
	E5Sesiones int `json:"e5_sesiones,omitempty"`
	// Codificador elige el codificador del kNN: "e5-small" (por defecto, y lo que hace una config sin este campo) o "qwen3-emb" (Qwen3-Embedding-0.6B, ver qwen.go).
	// Con "qwen3-emb" son obligatorios qwen_onnx y qwen_tok, y los bancos deben ser de Qwen (model/alt/aws/chispa-banco-qwen: mismos formatos .npy/.jsonl, dim 1024).
	Codificador string `json:"codificador,omitempty"`
	// QwenONNX: ruta del ONNX de Qwen3-Embedding (model/alt/exportar_qwen3emb_onnx.py, fp32 o int8; el .data de pesos va al lado).
	QwenONNX string `json:"qwen_onnx,omitempty"`
	// QwenTok: ruta del tokenizador BPE de Qwen (.chispa-bpe, runtime/tok/exportar_bpe.py).
	QwenTok string `json:"qwen_tok,omitempty"`
	// QwenSesiones: sesiones ORT de Qwen (cada una carga su copia de los pesos, ≈ 2,4 GB en fp32); 0 = 1.
	QwenSesiones int `json:"qwen_sesiones,omitempty"`
	// Modo: "" (el de siempre: contesta L2 si el margen ≥ u, si no escala a L3), "gpu" (todo a L3: el kNN solo sirve para el top-k de las preguntas con más de 20 opciones;
	// el escalado por margen y la respuesta de L2 quedan apagados; exige la sección l3) o "cpu" (solo kNN: contesta siempre L2, sin escalado por margen ni «siempre»;
	// las preguntas deben traer examples o estar en tareas, si no la petición se rechaza con examples_required). Ver server/PRODUCTO.md.
	Modo string `json:"modo,omitempty"`
	// CacheVectoresMB: tope de la caché LRU de vectores de los examples de la petición, en MiB; 0 = 64.
	CacheVectoresMB int `json:"cache_vectores_mb,omitempty"`
}

// Modos de knn.modo.
const (
	ModoKNNClasico = ""
	ModoKNNGPU     = "gpu"
	ModoKNNCPU     = "cpu"
)

// NombreCodificador es el codificador efectivo: e5-small si la config no dice nada (las configs anteriores a qwen3-emb arrancan igual que antes).
// VecinosEfectivos son los vecinos del kNN: los de la config o, con 0, el valor por defecto del codificador (32 con e5-small; 8 con
// qwen3-emb, el ganador en dev de L2-qwen3emb: macro e2e 0,4700 con 8 frente a 0,4686 con 32).
func (c ConfigKNN) VecinosEfectivos() int {
	if c.Vecinos > 0 {
		return c.Vecinos
	}
	if c.NombreCodificador() == CodificadorQwen3Emb {
		return KNNVecinosQwen3Emb
	}
	return KNNVecinosPorDefecto
}

func (c ConfigKNN) NombreCodificador() string {
	if c.Codificador == "" {
		return CodificadorE5Small
	}
	return c.Codificador
}

// TareaKNN describe una tarea.
type TareaKNN struct {
	Banco     string   `json:"banco"`               // ruta base: se leen <banco>.npy y <banco>.jsonl (ver BancoKNN)
	Preguntas []string `json:"preguntas,omitempty"` // textos de `question` que atiende; vacío = solo se usa con Evaluar (pruebas)
	Umbral    *float64 `json:"umbral,omitempty"`    // u de la tarea; ausente = el global
	Siempre   bool     `json:"siempre,omitempty"`
	// ScoreUmbrales: score calibrado con los umbrales de `<banco>.umbrales.json` (ver score_umbrales.go). nil = se usan si el fichero existe; false = se ignoran;
	// true = el fichero es obligatorio.
	ScoreUmbrales *bool `json:"score_umbrales,omitempty"`
}

func (c ConfigKNN) validar() error {
	// Sin tareas solo hay kNN con examples en la petición, y eso exige un codificador en el servidor.
	if len(c.Tareas) == 0 && c.E5ONNX == "" && c.NombreCodificador() != CodificadorQwen3Emb {
		return errors.New("config: knn.tareas no puede estar vacío (salvo con un codificador en el servidor, para usar solo examples de la petición)")
	}
	switch c.Modo {
	case ModoKNNClasico, ModoKNNGPU, ModoKNNCPU:
	default:
		return fmt.Errorf("config: knn.modo %q desconocido (valores: \"\", %q, %q)", c.Modo, ModoKNNGPU, ModoKNNCPU)
	}
	if c.CacheVectoresMB < 0 || c.CacheVectoresMB > MaxCacheMB {
		return fmt.Errorf("config: knn.cache_vectores_mb debe estar en [0, %d]", MaxCacheMB)
	}
	if !(c.Umbral >= 0 && c.Umbral <= 1) {
		return fmt.Errorf("config: knn.umbral fuera de [0,1]: %v", c.Umbral)
	}
	if c.E5Sesiones < 0 {
		return errors.New("config: knn.e5_sesiones no puede ser negativo")
	}
	switch c.NombreCodificador() {
	case CodificadorE5Small:
		if c.QwenONNX != "" || c.QwenTok != "" || c.QwenSesiones != 0 {
			return fmt.Errorf("config: knn.qwen_onnx, qwen_tok y qwen_sesiones solo valen con knn.codificador = %q", CodificadorQwen3Emb)
		}
	case CodificadorQwen3Emb:
		if c.QwenONNX == "" || c.QwenTok == "" {
			return fmt.Errorf("config: knn.codificador = %q exige knn.qwen_onnx y knn.qwen_tok", CodificadorQwen3Emb)
		}
		if c.E5ONNX != "" || c.E5Sesiones != 0 {
			return fmt.Errorf("config: knn.e5_onnx y e5_sesiones no valen con knn.codificador = %q (cambia a e5-small o quítalos)", CodificadorQwen3Emb)
		}
		if c.QwenSesiones < 0 {
			return errors.New("config: knn.qwen_sesiones no puede ser negativo")
		}
	default:
		return fmt.Errorf("config: knn.codificador %q desconocido (valores: %q, %q)", c.Codificador, CodificadorE5Small, CodificadorQwen3Emb)
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
	umbrales *UmbralesScore // score calibrado; nil = sin calibrar
	ejemplos []EjemploFS    // few-shot FIJO de L3 (<banco>.ejemplos.jsonl; en formato pool, las marcadas fs); nil = la tarea contesta zero-shot
	pool     []EjemploFS    // formato pool: todo el oro con su `fila` del .npy, soporte de las demostraciones por kNN; nil = formato viejo
	banco    *BancoKNN
	umbral   float64
	siempre  bool
}

// L2KNN: ver el comentario del fichero.
type L2KNN struct {
	p        ParamsKNN
	tareas   map[string]*tareaKNNViva
	porPreg  map[string]string
	cod      Codificador
	met      *Metricas
	cfg      ConfigKNN
	cacheVec *cacheVectores // vectores de los examples de la petición
	demosK   int            // > 0: demostraciones por consulta (kNN sobre el soporte), ver l3_demos_knn.go
	sinTopK  bool           // L3 no usa el top-k del kNN (l3.top_k_knn = 0 o l3.top_k explícito): en modo gpu el kNN no tiene nada que hacer
	Avisos   []string       // cosas dudosas al cargar (tareas con oro y sin ejemplos fs)
}

// EjemplosDe da los ejemplos de oro del few-shot de L3 para una pregunta (la tarea cuyo texto de `question` coincide); nil si no hay.
func (k *L2KNN) EjemplosDe(pregunta string) []EjemploFS {
	if t, ok := k.porPreg[pregunta]; ok {
		return k.tareas[t].ejemplos
	}
	return nil
}

// UmbralesDe da los umbrales del score calibrado para una pregunta (la tarea cuyo texto de `question` coincide); nil si no hay.
func (k *L2KNN) UmbralesDe(pregunta string) *UmbralesScore {
	if t, ok := k.porPreg[pregunta]; ok {
		return k.tareas[t].umbrales
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

// ComprobarDim codifica un texto de prueba y falla si la dimensión del codificador no es la de algún banco (un banco de e5, 384, con Qwen, 1024, o al revés):
// sin esto el error saldría en cada consulta (errDimKNN). Sin codificador no hace nada.
func (k *L2KNN) ComprobarDim(ctx context.Context) error {
	if k.cod == nil {
		return nil
	}
	v, err := k.cod.Codificar(ctx, "prueba")
	if err != nil {
		return fmt.Errorf("knn: codificador: %w", err)
	}
	for _, n := range k.Tareas() {
		if d := k.tareas[n].banco.Dim; d != len(v) {
			return fmt.Errorf("knn: el banco de la tarea %q tiene dimensión %d y el codificador da %d (¿bancos de otro codificador? regenérelos con él)", n, d, len(v))
		}
	}
	return nil
}

// NuevoL2KNN carga los bancos de la config. cod puede ser nil (solo vectores inyectados).
func NuevoL2KNN(cfg ConfigKNN, cod Codificador) (*L2KNN, error) {
	if err := cfg.validar(); err != nil {
		return nil, err
	}
	k := &L2KNN{p: ParamsKNN{Vecinos: cfg.VecinosEfectivos(), Temp: cfg.Temperatura, SOro: cfg.SOro}.conDefectos(), tareas: map[string]*tareaKNNViva{}, porPreg: map[string]string{}, cod: cod, cfg: cfg, cacheVec: nuevaCacheVectores(cfg.CacheVectoresMB)}
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
		pool, err := poolDeBanco(ej, b)
		if err != nil {
			return nil, fmt.Errorf("knn: tarea %q: %w", n, err)
		}
		if pool != nil {
			if ej = fijasDePool(pool); len(ej) == 0 {
				ej = nil
			}
		}
		if b.NOro > 0 && ej == nil && pool == nil {
			k.Avisos = append(k.Avisos, fmt.Sprintf("tarea %q: tiene oro pero no %s.ejemplos.jsonl: L3 contestará zero-shot", n, t.Banco))
		}
		var umb *UmbralesScore
		if t.ScoreUmbrales == nil || *t.ScoreUmbrales {
			if umb, err = CargarUmbralesScore(t.Banco); err != nil {
				return nil, fmt.Errorf("knn: tarea %q: %w", n, err)
			}
			if umb == nil && t.ScoreUmbrales != nil {
				return nil, fmt.Errorf("knn: tarea %q: score_umbrales=true y no existe %s", n, RutaUmbrales(t.Banco))
			}
		}
		u := cfg.Umbral
		if t.Umbral != nil {
			u = *t.Umbral
		}
		k.tareas[n] = &tareaKNNViva{umbrales: umb, ejemplos: ej, pool: pool, banco: b, umbral: u, siempre: siempre[n] || t.Siempre}
		for _, q := range t.Preguntas {
			k.porPreg[q] = n
		}
	}
	return k, nil
}

func (k *L2KNN) Nombre() string { return "L2" }

// ConTopK dice si L3 usará el top-k del kNN; sin él, con knn.modo = gpu el kNN no codifica la consulta (la lo hace main con Config.TopKKNNActivo).
func (k *L2KNN) ConTopK(activo bool) *L2KNN {
	k.sinTopK = !activo
	return k
}

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
	return k.evaluarBanco(t.banco, t.umbral, t.siempre, q, ops)
}

// evaluarBanco es Evaluar sobre un banco cualquiera (el de la tarea o el de los examples de la petición). `Answer` es el argmax; la regla por tipo
// (mediana en score, ver RespuestaKNN) la aplica Decidir. Con knn.modo = cpu no hay escalado por margen ni «siempre» (solo «sin_oro»); con gpu escala todo.
func (k *L2KNN) evaluarBanco(b *BancoKNN, umbral float64, siempre bool, q []float32, ops []string) (DecisionKNN, error) {
	probs, sen, err := b.Votar(q, ops, k.p)
	if err != nil {
		return DecisionKNN{}, err
	}
	d := DecisionKNN{Probs: probs, Sen: sen, Umbral: umbral}
	if len(ops) > 0 {
		d.Answer = ops[argmaxF(probs)]
	}
	switch {
	case b.NOro == 0:
		d.Escala, d.Motivo = true, MotivoKNNSinOro
	case k.cfg.Modo == ModoKNNCPU:
	case k.cfg.Modo == ModoKNNGPU || siempre:
		d.Escala, d.Motivo = true, MotivoKNNSiempre
	case sen.Margen < umbral:
		d.Escala, d.Motivo = true, MotivoKNNMargen
	}
	return d, nil
}

// ExigeEjemplos: con knn.modo = cpu, una pregunta que no es de ninguna tarea con banco y no trae examples no se puede contestar: la petición se rechaza (examples_required).
func (k *L2KNN) ExigeEjemplos(p Pregunta) bool {
	if k.cfg.Modo != ModoKNNCPU || len(p.Ejemplos) > 0 {
		return false
	}
	_, ok := k.porPreg[p.Texto]
	return !ok
}

// nombreTareaPeticion es la tarea de las métricas de una pregunta con examples que no está en la config.
const nombreTareaPeticion = "(peticion)"

func (k *L2KNN) Decidir(ctx context.Context, p Pregunta) (Respuesta, error) {
	tarea, enConfig := k.porPreg[p.Texto]
	conEj := len(p.Ejemplos) > 0
	if (!enConfig && !conEj) || p.Kind == KindMulti || len(p.Opciones) == 0 {
		return Respuesta{}, nil // la pregunta no es de ninguna tarea con banco ni trae examples: no opina
	}
	ops := make([]string, len(p.Opciones))
	for i, o := range p.Opciones {
		ops[i] = o.ID
	}
	nombre := tarea
	if !enConfig {
		nombre = nombreTareaPeticion
	}
	// knn.modo = gpu: todo va a L3; el vector de la consulta solo hace falta para el top-k de L3 (choice con más de L3TopKKNNDesde opciones).
	conDemos := k.demosK > 0 && k.cfg.Modo == ModoKNNGPU && (conEj || (enConfig && len(k.tareas[tarea].pool) > 0))
	if k.cfg.Modo == ModoKNNGPU && !conDemos && (k.sinTopK || !(p.Kind == KindChoice && len(ops) > L3TopKKNNDesde)) {
		k.met.knnFila(nombre, MotivoKNNSiempre)
		return Respuesta{Tarea: nombre, MotivoKNN: MotivoKNNSiempre}, nil
	}
	q, _ := ctx.Value(claveVectorKNN{}).([]float32)
	pr := presupuestoDe(ctx)
	if q == nil && k.cod != nil {
		var err error
		t0 := time.Now()
		if cq, ok := k.cod.(CodificadorConsulta); ok {
			q, err = cq.CodificarConsulta(ctx, p.Texto, p.Estado) // Qwen: la instrucción lleva la pregunta
		} else {
			q, err = k.cod.Codificar(ctx, p.Estado)
		}
		d := time.Since(t0)
		k.met.knnCodificador(d)
		pr.knnCodNs.Add(int64(d))
		if err != nil {
			return Respuesta{}, fmt.Errorf("knn: codificador: %w", err)
		}
	}
	if q == nil || (conEj && k.cod == nil) {
		k.met.knnSinVector()
		return Respuesta{}, nil // sin vector no hay kNN: sigue L3
	}
	var tv *tareaKNNViva
	if enConfig {
		tv = k.tareas[tarea]
	}
	umbral, siempre := k.cfg.Umbral, false
	if tv != nil {
		umbral, siempre = tv.umbral, tv.siempre
	}
	var banco *BancoKNN
	if conEj { // los examples de la petición sustituyen al banco de la pregunta
		t0 := time.Now()
		var err error
		banco, err = k.bancoDePeticion(ctx, p)
		pr.knnCodNs.Add(int64(time.Since(t0)))
		if err != nil {
			return Respuesta{}, fmt.Errorf("knn: %w", err)
		}
	} else {
		banco = tv.banco
	}
	t1 := time.Now()
	d, err := k.evaluarBanco(banco, umbral, siempre, q, ops)
	dv := time.Since(t1)
	k.met.knnVotacion(dv)
	pr.knnNs.Add(int64(dv))
	if err != nil {
		return Respuesta{}, err
	}
	k.met.knnFila(nombre, d.Motivo)
	if r, ok := RespuestaKNN(p.Kind, ops, d.Probs); ok {
		d.Answer = r // mediana CORAL en score, argmax en el resto
	}
	probs := make(map[string]float64, len(ops))
	for i, o := range ops {
		probs[o] = d.Probs[i]
	}
	tau := d.Umbral
	if d.Escala {
		tau = math.Inf(1) // nunca se acepta: la cascada sigue con L3 y conserva esta respuesta como pista
	} else if k.cfg.Modo == ModoKNNCPU {
		tau = 0 // sin escalado: contesta L2 aunque el margen sea mínimo
	}
	r := Respuesta{Answer: d.Answer, Probs: probs, Conf: math.Max(d.Sen.Margen, 1e-9), TauPropio: &tau, Tarea: nombre, MotivoKNN: d.Motivo}
	if conDemos {
		if conEj {
			r.Demos = k.demosDePeticionKNN(p, banco, q)
		} else {
			r.Demos = k.demosDeTarea(tv, q)
		}
	}
	if d.Escala && p.Kind == KindChoice && len(ops) > L3TopKKNNDesde { // desempate de la preselección top-k de L3
		if c, err := banco.SimCentroides(q, ops); err == nil {
			r.RankKNN = make(map[string]float64, len(ops))
			for i, o := range ops {
				r.RankKNN[o] = c[i]
			}
		}
	}
	return r, nil
}
