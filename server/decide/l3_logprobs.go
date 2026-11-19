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
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Modo logprobs de L3 (config l3.modo = "logprobs"): en vez de pedirle al LLM un JSON de probabilidades, se le pide UNA etiqueta
// de un carácter por opción y se lee la distribución de su primer token. Es el método medido en model/alt/llm_logprobs.py (0,729
// agrupada frente a 0,650 con JSON; 33 ms p50 con un hilo): este fichero es su espejo en Go y el prompt y la conversión a probs
// deben seguir siendo idénticos (ver testdata/l3_logprobs_paridad.json y su test).
//
//  1. Cada opción recibe una etiqueta A-Z, a-z, 0-9 (62; más opciones = la fila sale inválida). En score con niveles de una cifra
//     la etiqueta es el propio nivel.
//  2. POST {url}/chat/completions con max_tokens=1, logprobs=true y top_logprobs=20; las etiquetas que salen en el top suman sus
//     variantes ("A" y " A") en probabilidad.
//  3. Las etiquetas que no salen se puntúan con prompt_logprobs (la conversación termina en la etiqueta: add_generation_prompt=false
//     y continue_final_message=true) para las primeras max_extra (4); el resto recibe un piso (mínimo conocido − 1 nat).
//  4. probs = softmax sobre las opciones. Inválida (Incompleto, sin error) solo si no se puntuó ninguna.
//
// Tope por fila (timeout_fila) incluyendo el relleno: si vence con algo puntuado se devuelven las probs parciales con piso; si no,
// el error ErrSinL3. Los errores de red (y los 5xx) se reintentan 1 vez. La concurrencia máxima es max_concurrente (semáforo).

const (
	L3ModoJSON      = "json"
	L3ModoLogprobs  = "logprobs"
	topLogprobsL3   = 20
	maxExtraL3      = 4
	etiquetasL3     = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	instruccionLPL3 = `Cada opción lleva una "etiqueta" de un solo carácter. Responde SOLO con la etiqueta de la opción elegida: un único carácter, ` +
		`sin comillas, sin explicación y sin JSON. tipo "choice": una opción correcta. tipo "noul": la pregunta es de sí/no; las opciones son "true" y "false". ` +
		`tipo "score": las opciones son los niveles 1..K, del menor (1) al mayor (K); elige el nivel que mejor corresponda. ` +
		`tipo "multi": elige la opción más probable.`
)

// ErrSinL3: L3 no pudo puntuar la fila (red caída, tope de tiempo sin nada puntuado). La cascada la trata como «L3 no opina».
var ErrSinL3 = errors.New("sin L3")

var errMasDe62 = errors.New("más opciones que etiquetas de un carácter")

// mensajesLogprobsL3 devuelve los dos mensajes (sistema, usuario) y las etiquetas en el orden de las opciones. Replica
// `preparar` de llm_logprobs.py, incluido el formato de json.dumps(ensure_ascii=False) (separadores ", " y ": ", sin escapar < > &).
func mensajesLogprobsL3(p Pregunta, ordenadas []Opcion) ([]msgL3, []string, error) {
	cods, err := codigosEtiquetaL3(p, ordenadas)
	if err != nil {
		return nil, nil, err
	}
	var b strings.Builder
	b.WriteString(`{"pregunta": `)
	b.WriteString(jsonPyL3(p.Texto))
	b.WriteString(`, "tipo": `)
	b.WriteString(jsonPyL3(p.Kind))
	b.WriteString(`, "opciones": [`)
	for i, o := range ordenadas {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(`{"id": ` + jsonPyL3(o.ID))
		if o.Description != "" {
			b.WriteString(`, "descripcion": ` + jsonPyL3(o.Description))
		}
		b.WriteString(`, "etiqueta": ` + jsonPyL3(cods[i]) + "}")
	}
	b.WriteString(`], "texto": ` + jsonPyL3(delimitaTextoL3(p.Estado)) + "}")
	sistema := promptSistemaL3[:strings.Index(promptSistemaL3, "Responde SOLO")] + instruccionLPL3
	return []msgL3{{"system", sistema}, {"user", b.String()}}, cods, nil
}

// codigosEtiquetaL3 asigna la etiqueta de un carácter de cada opción (en su orden): el propio nivel en score con niveles de una cifra, y si no A-Z, a-z, 0-9.
func codigosEtiquetaL3(p Pregunta, ordenadas []Opcion) ([]string, error) {
	n := len(ordenadas)
	cods := make([]string, n)
	digitos := p.Kind == KindScore
	for _, o := range ordenadas {
		if len(o.ID) != 1 || o.ID[0] < '0' || o.ID[0] > '9' {
			digitos = false
		}
	}
	switch {
	case digitos:
		for i, o := range ordenadas {
			cods[i] = o.ID
		}
	case n <= len(etiquetasL3):
		for i := range cods {
			cods[i] = etiquetasL3[i : i+1]
		}
	default:
		return nil, errMasDe62
	}
	return cods, nil
}

// jsonPyL3 codifica un string como json.dumps(s, ensure_ascii=False).
func jsonPyL3(s string) string {
	var buf bytes.Buffer
	e := json.NewEncoder(&buf)
	e.SetEscapeHTML(false)
	_ = e.Encode(s)
	r := strings.TrimSuffix(buf.String(), "\n")
	return strings.NewReplacer("\\u2028", " ", "\\u2029", " ").Replace(r)
}

type cuerpoLPL3 struct {
	Model              string         `json:"model"`
	Messages           []msgL3        `json:"messages"`
	Temperature        float64        `json:"temperature"`
	MaxTokens          int            `json:"max_tokens"`
	Stream             bool           `json:"stream"`
	Logprobs           bool           `json:"logprobs,omitempty"`
	TopLogprobs        int            `json:"top_logprobs,omitempty"`
	PromptLogprobs     *int           `json:"prompt_logprobs,omitempty"`
	AddGenerationP     *bool          `json:"add_generation_prompt,omitempty"`
	ContinueFinalMsg   bool           `json:"continue_final_message,omitempty"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

func (l *L3LLM) cuerpoBaseLP(msgs []msgL3) cuerpoLPL3 {
	c := cuerpoLPL3{Model: l.cfg.Modelo, Messages: msgs, MaxTokens: 1}
	if l.cfg.SinPensar {
		c.ChatTemplateKwargs = map[string]any{"enable_thinking": false}
	}
	return c
}

// postLP hace la llamada con reintentos: hasta max_intentos en total ante error de red, 5xx o 429, con espera exponencial y jitter
// (espera_base · 2^(n-1) hasta espera_max, multiplicada por un factor en [0,5, 1)). Un 429 con Retry-After espera lo que pide. Si lo que
// queda de plazo (contexto de la fila) no alcanza para la espera, no se reintenta: el error sale ya y la fila no se pasa de su tope.
func (l *L3LLM) postLP(ctx context.Context, c cuerpoLPL3) ([]byte, error) {
	cuerpo, _ := json.Marshal(c)
	for intento := 1; ; intento++ {
		b, causa, retryAfter, err := l.postUnaVez(ctx, cuerpo)
		if err == nil {
			return b, nil
		}
		if causa == "" || intento >= l.cfg.maxIntentos() || ctx.Err() != nil {
			return nil, err
		}
		espera := l.cfg.espera(intento, l.azar())
		if causa == "429" && retryAfter > 0 {
			espera = retryAfter
		}
		if dl, ok := ctx.Deadline(); ok && time.Until(dl) <= espera {
			return nil, err
		}
		l.met.l3Reintento(causa)
		if l.dormir(ctx, espera) != nil {
			return nil, err
		}
	}
}

// postUnaVez: una llamada. causa != "" si el fallo admite reintento (red, 5xx, 429); retryAfter es el de un 429.
func (l *L3LLM) postUnaVez(ctx context.Context, cuerpo []byte) (b []byte, causa string, retryAfter time.Duration, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.url, bytes.NewReader(cuerpo))
	if err != nil {
		return nil, "", 0, errors.New("L3: petición no válida")
	}
	req.Header.Set("Content-Type", "application/json")
	if l.clave != "" {
		req.Header.Set("Authorization", "Bearer "+l.clave)
	}
	resp, err := l.cli.Do(req)
	if err != nil {
		var ue *url.Error
		if errors.As(err, &ue) {
			err = ue.Err
		}
		return nil, "red", 0, fmt.Errorf("L3: %w", err)
	}
	defer resp.Body.Close()
	b, err = io.ReadAll(io.LimitReader(resp.Body, maxRespuestaL3))
	if err != nil {
		return nil, "red", 0, fmt.Errorf("L3: %w", err)
	}
	switch {
	case resp.StatusCode/100 == 2:
		return b, "", 0, nil
	case resp.StatusCode == http.StatusTooManyRequests:
		return nil, "429", parseRetryAfter(resp.Header.Get("Retry-After"), time.Now()), fmt.Errorf("%w (%d)", errL3Estado, resp.StatusCode)
	case resp.StatusCode >= 500:
		return nil, "5xx", 0, fmt.Errorf("%w (%d)", errL3Estado, resp.StatusCode)
	}
	return nil, "", 0, fmt.Errorf("%w (%d)", errL3Estado, resp.StatusCode)
}

// parseRetryAfter: segundos (entero) o fecha HTTP; 0 si falta o no se entiende (se usa la espera exponencial).
func parseRetryAfter(v string, ahora time.Time) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	if n, err := strconv.ParseFloat(v, 64); err == nil {
		if n < 0 || math.IsNaN(n) || n > 86400 {
			return 0
		}
		return time.Duration(n * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(ahora); d > 0 {
			return d
		}
	}
	return 0
}

type plEntradaL3 map[string]struct {
	Logprob      float64 `json:"logprob"`
	DecodedToken string  `json:"decoded_token"`
}

// primerTokenLP: logprob de cada token del top 20 del primer token generado (variantes con espacio sumadas) y todos los del top.
func (l *L3LLM) primerTokenLP(ctx context.Context, msgs []msgL3) (map[string]float64, []float64, error) {
	c := l.cuerpoBaseLP(msgs)
	c.Logprobs, c.TopLogprobs = true, topLogprobsL3
	b, err := l.postLP(ctx, c)
	if err != nil {
		return nil, nil, err
	}
	var out struct {
		Choices []struct {
			Logprobs struct {
				Content []struct {
					TopLogprobs []struct {
						Token   string  `json:"token"`
						Logprob float64 `json:"logprob"`
					} `json:"top_logprobs"`
				} `json:"content"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if json.Unmarshal(b, &out) != nil || len(out.Choices) == 0 || len(out.Choices[0].Logprobs.Content) == 0 {
		return nil, nil, errors.New("L3: respuesta sin logprobs")
	}
	por := map[string]float64{}
	var todos []float64
	for _, t := range out.Choices[0].Logprobs.Content[0].TopLogprobs {
		todos = append(todos, t.Logprob)
		k := strings.TrimSpace(t.Token)
		if v, ok := por[k]; ok {
			por[k] = lseL3(v, t.Logprob)
		} else {
			por[k] = t.Logprob
		}
	}
	return por, todos, nil
}

// prefijoLP: logprob de la etiqueta cod como inicio de la respuesta, por prompt_logprobs.
func (l *L3LLM) prefijoLP(ctx context.Context, msgs []msgL3, cod string) (float64, error) {
	c := l.cuerpoBaseLP(append(append([]msgL3(nil), msgs...), msgL3{"assistant", cod}))
	cero, falso := 0, false
	c.PromptLogprobs, c.AddGenerationP, c.ContinueFinalMsg = &cero, &falso, true
	b, err := l.postLP(ctx, c)
	if err != nil {
		return 0, err
	}
	var out struct {
		Choices []struct {
			PromptLogprobs []plEntradaL3 `json:"prompt_logprobs"`
		} `json:"choices"`
		PromptLogprobs []plEntradaL3 `json:"prompt_logprobs"`
	}
	if json.Unmarshal(b, &out) != nil {
		return 0, errors.New("L3: respuesta sin prompt_logprobs")
	}
	pl := out.PromptLogprobs
	if len(pl) == 0 && len(out.Choices) > 0 {
		pl = out.Choices[0].PromptLogprobs
	}
	var no []plEntradaL3
	for _, e := range pl {
		if len(e) > 0 {
			no = append(no, e)
		}
	}
	if len(no) == 0 {
		return 0, errors.New("L3: prompt_logprobs vacío")
	}
	for _, v := range no[len(no)-1] { // el último token del prompt es la etiqueta
		if strings.TrimSpace(v.DecodedToken) == cod {
			return v.Logprob, nil
		}
	}
	return 0, errors.New("L3: prompt_logprobs no trae el token de la etiqueta")
}

func lseL3(a, b float64) float64 {
	m := math.Max(a, b)
	return m + math.Log(math.Exp(a-m)+math.Exp(b-m))
}

// puntuarLP devuelve las probs por etiqueta (en el orden de cods). ok=false si no se pudo puntuar ninguna. Si el contexto vence
// a mitad del relleno, usa lo que ya hay; si vence sin nada, devuelve el error del contexto. parcial: alguna etiqueta recibió el piso.
func (l *L3LLM) puntuarLP(ctx context.Context, msgs []msgL3, cods []string) (xs []float64, ok, parcial bool, err error) {
	top, todos, err := l.primerTokenLP(ctx, msgs)
	if err != nil {
		return nil, false, false, err
	}
	lp := map[string]float64{}
	for _, c := range cods {
		if v, esta := top[c]; esta {
			lp[c] = v
		}
	}
	extra, hechas := l.cfg.maxExtra(), 0
	for _, c := range cods {
		if hechas >= extra || ctx.Err() != nil {
			break
		}
		if _, esta := lp[c]; esta {
			continue
		}
		v, e := l.prefijoLP(ctx, msgs, c)
		if e != nil { // el relleno es opcional: se sigue con lo que hay (como el Python)
			break
		}
		lp[c] = v
		hechas++
	}
	if len(lp) == 0 {
		return nil, false, false, ctx.Err()
	}
	piso := pisoLP(todos, lp)
	xs = make([]float64, len(cods))
	vs := make([]float64, len(cods))
	for i, c := range cods {
		v, esta := lp[c]
		if !esta {
			v = piso
		}
		vs[i] = v
	}
	return softmaxLP(vs, xs), true, len(lp) < len(cods), nil
}

// pisoLP: el menor logprob conocido (top 20 y puntuados) menos 1 nat.
func pisoLP(todos []float64, lp map[string]float64) float64 {
	piso := math.Inf(1)
	for _, v := range todos {
		piso = math.Min(piso, v)
	}
	for _, v := range lp {
		piso = math.Min(piso, v)
	}
	return piso - 1
}

func softmaxLP(vs, xs []float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range vs {
		mx = math.Max(mx, v)
	}
	var s float64
	for i, v := range vs {
		xs[i] = math.Exp(v - mx)
		s += xs[i]
	}
	for i := range xs {
		xs[i] /= s
	}
	return xs
}

// entrarFila toma un hueco del semáforo de L3 o descarta la fila: sin hueco libre, espera en cola si hay sitio (max_cola); si la cola
// está llena, descarte inmediato (ErrColaLlena) en vez de esperar. libera() devuelve el hueco.
func (l *L3LLM) entrarFila(ctx context.Context) (libera func(), err error) {
	entra := func() func() {
		l.met.l3Uso(1)
		return func() { <-l.sem; l.met.l3Uso(-1) }
	}
	pr := presupuestoDe(ctx)
	select {
	case l.sem <- struct{}{}:
		l.met.l3EsperaCola(0)
		return entra(), nil
	default:
	}
	if n := l.cola.Add(1); int(n) > l.cfg.maxCola() {
		l.cola.Add(-1)
		l.met.l3Descarte("cola_llena")
		return nil, ErrColaLlena
	}
	l.met.l3ColaFija(l.cola.Load())
	te := time.Now()
	defer func() { l.cola.Add(-1); l.met.l3ColaFija(l.cola.Load()) }()
	select {
	case l.sem <- struct{}{}:
		l.met.l3EsperaCola(time.Since(te))
		pr.l3EsperaNs.Add(int64(time.Since(te)))
		return entra(), nil
	case <-ctx.Done():
		return nil, fmt.Errorf("%w: %v", ErrSinL3, ctx.Err())
	}
}

// decidirLogprobs es Decidir en modo logprobs; ordenadas ya viene en orden de nivel en score y con el presupuesto reservado.
func (l *L3LLM) decidirLogprobs(ctx context.Context, p Pregunta, ordenadas []Opcion) (Respuesta, error) {
	var msgs []msgL3
	var cods []string
	if p.Kind == KindMulti {
		if len(ordenadas) > l.cfg.multiMaxOpciones() {
			l.met.l3Cuenta("invalida")
			return Respuesta{Incompleto: true}, nil
		}
	} else {
		var err error
		if msgs, cods, err = mensajesLogprobsL3(p, ordenadas); err != nil {
			l.met.l3Cuenta("invalida")
			return Respuesta{Incompleto: true}, nil
		}
		if ds, ok := ctx.Value(claveDemosKNN{}).([]EjemploFS); ok && l.ejemplosPorPeticion() > 0 { // demostraciones por consulta: las k más cercanas que eligió L2
			if t, err := turnosEjemplosL3(ds); err == nil && len(t) > 0 {
				msgs = conEjemplosL3(msgs, t)
			}
		} else if len(p.Ejemplos) > 0 { // los examples de la petición mandan sobre el banco: son las demostraciones (ver DemosDePeticion)
			if ejs := DemosDePeticion(p, l.ejemplosPorPeticion()); len(ejs) > 0 {
				if t, err := turnosEjemplosL3(ejs); err == nil && len(t) > 0 {
					msgs = conEjemplosL3(msgs, t)
				}
			}
		} else if t := l.turnosParaL3(p.Texto); len(t) > 0 { // few-shot: ejemplos de oro de la tarea como turnos previos
			msgs = conEjemplosL3(msgs, t)
		}
	}
	ok, sonda := l.circ.permite()
	if !ok {
		l.met.l3Descarte("circuito_abierto")
		l.met.l3Cuenta("error")
		return Respuesta{}, ErrCircuitoAbierto
	}
	libera, err := l.entrarFila(ctx)
	if err != nil {
		l.circ.registra(true, sonda, true) // ni éxito ni fallo del LLM: solo se suelta la prueba
		if errors.Is(err, ErrColaLlena) {
			l.met.l3Cuenta("error")
		} else {
			l.met.l3Cuenta("timeout")
		}
		return Respuesta{}, err
	}
	defer libera()
	cf, cancel := context.WithTimeout(ctx, l.cfg.timeoutFila())
	defer cancel()
	t := time.Now()
	var r Respuesta
	var valida, parcial bool
	if p.Kind == KindMulti {
		r, valida, parcial, err = l.puntuarMulti(cf, p)
	} else {
		var xs []float64
		xs, valida, parcial, err = l.puntuarLP(cf, msgs, cods)
		if err == nil && !valida && errors.Is(cf.Err(), context.DeadlineExceeded) {
			err = cf.Err()
		}
		if err == nil && valida {
			r = respuestaDeProbsLP(p, ordenadas, xs)
			if p.Kind == KindScore && l.umbralesDe != nil { // score calibrado: esperanza ordinal + umbrales del banco en vez de la mediana
				if i, ok := nivelCalibrado(ordenadas, xs, l.umbralesDe(p.Texto)); ok {
					r.Answer = ordenadas[i].ID
					r.Conf = r.Probs[r.Answer]
				}
			}
		}
	}
	l.met.l3Latencia(time.Since(t))
	presupuestoDe(ctx).l3Ns.Add(int64(time.Since(t)))
	l.circ.registra(err == nil, sonda, err != nil && ctx.Err() != nil)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || os.IsTimeout(err) {
			l.met.l3Cuenta("timeout")
		} else {
			l.met.l3Cuenta("error")
		}
		return Respuesta{}, fmt.Errorf("%w: %v", ErrSinL3, err)
	}
	if !valida {
		l.met.l3Cuenta("invalida")
		return Respuesta{Incompleto: true}, nil
	}
	if parcial {
		l.met.l3Parcial()
	}
	l.met.l3Cuenta("ok")
	return r, nil
}

// respuestaDeProbsLP: answer = el más probable (primera opción en empate); en score, la mediana.
func respuestaDeProbsLP(p Pregunta, ordenadas []Opcion, xs []float64) Respuesta {
	probs := make(map[string]float64, len(ordenadas))
	for i, o := range ordenadas {
		probs[o.ID] = xs[i]
	}
	ans := ordenadas[0].ID
	if p.Kind == KindScore {
		var acc float64
		for _, o := range ordenadas {
			if acc += probs[o.ID]; acc >= 0.5 {
				ans = o.ID
				break
			}
		}
	} else {
		for _, o := range ordenadas {
			if probs[o.ID] > probs[ans] {
				ans = o.ID
			}
		}
	}
	return Respuesta{Answer: ans, Probs: probs, Conf: probs[ans]}
}

// Multi por logprobs (EXTENSIÓN SOLO-GO: llm_logprobs.py no tiene multi; allí «multi» se trata como una elección única, «elige la opción
// más probable»). Cada opción es un sí/no independiente: UNA llamada por opción (max_tokens=1, top_logprobs=20) con las etiquetas S/N;
// p(opción) = p(S) / (p(S) + p(N)), sumando las variantes con espacio. Se descartó «dos etiquetas por opción en una sola llamada»: el
// primer token solo da UNA distribución, no n sí/no independientes; encadenar prefijos costaría n llamadas igualmente. El coste es n
// llamadas por fila (multi_max_opciones, 16 por defecto), que van en paralelo dentro del hueco del semáforo (hasta 8 a la vez) y comparten
// el prefijo del prompt (--enable-prefix-caching). Si de S y N solo sale una en el top 20, la otra recibe el piso (menor logprob visto − 1).
// Si no sale ninguna la opción no se puede puntuar y la fila es inválida. θ = 0,5 como en la vía JSON (multiDeProbs).
const (
	instruccionMultiLPL3 = `Se te pregunta por UNA sola opción (campo "opcion"), independiente de las demás. Responde SOLO con una letra: S si esa opción corresponde al texto, ` +
		`N si no. Un único carácter, sin comillas, sin explicación y sin JSON.`
	multiParaleloL3 = 8
)

func mensajesMultiLP(p Pregunta, o Opcion) []msgL3 {
	op := `{"id": ` + jsonPyL3(o.ID)
	if o.Description != "" {
		op += `, "descripcion": ` + jsonPyL3(o.Description)
	}
	op += "}"
	u := `{"pregunta": ` + jsonPyL3(p.Texto) + `, "tipo": "multi", "opcion": ` + op + `, "texto": ` + jsonPyL3(delimitaTextoL3(p.Estado)) + "}"
	sistema := promptSistemaL3[:strings.Index(promptSistemaL3, "Responde SOLO")] + instruccionMultiLPL3
	return []msgL3{{"system", sistema}, {"user", u}}
}

func (l *L3LLM) puntuarMulti(ctx context.Context, p Pregunta) (r Respuesta, valida, parcial bool, err error) {
	n := len(p.Opciones)
	pr, th := make([]float64, n), make([]float64, n)
	pis := make([]bool, n) // opción puntuada con piso
	sin := make([]bool, n) // opción sin ninguna de las dos etiquetas
	errs := make([]error, n)
	var wg sync.WaitGroup
	par := make(chan struct{}, multiParaleloL3)
	for i, o := range p.Opciones {
		wg.Add(1)
		par <- struct{}{}
		go func() {
			defer func() { <-par; wg.Done() }()
			top, todos, e := l.primerTokenLP(ctx, mensajesMultiLP(p, o))
			if e != nil {
				errs[i] = e
				return
			}
			s, okS := top["S"]
			nn, okN := top["N"]
			if !okS && !okN {
				sin[i] = true
				return
			}
			piso := pisoLP(todos, map[string]float64{})
			if okS {
				piso = math.Min(piso, s-1)
			}
			if okN {
				piso = math.Min(piso, nn-1)
			}
			if !okS {
				s, pis[i] = piso, true
			}
			if !okN {
				nn, pis[i] = piso, true
			}
			pr[i], th[i] = softmaxLP([]float64{s, nn}, make([]float64, 2))[0], 0.5
		}()
	}
	wg.Wait()
	for _, e := range errs {
		if e != nil {
			return Respuesta{}, false, false, e
		}
	}
	for i := range sin {
		if sin[i] {
			return Respuesta{}, false, false, ctx.Err() // vencida: error de tope; si no, inválida
		}
		parcial = parcial || pis[i]
	}
	return multiDeProbs(p, pr, th), true, parcial, nil
}

func (c ConfigL3) maxIntentos() int {
	if c.MaxIntentos > 0 {
		return c.MaxIntentos
	}
	return 2
}

// espera es la espera antes del reintento número n (1 = el primero): espera_base · 2^(n−1) hasta espera_max, por un jitter en [0,5, 1).
func (c ConfigL3) espera(n int, azar float64) time.Duration {
	base, tope := time.Duration(c.EsperaBase), time.Duration(c.EsperaMax)
	if base == 0 {
		base = 50 * time.Millisecond
	}
	if tope == 0 {
		tope = time.Second
	}
	d := base
	for i := 1; i < n && d < tope; i++ {
		d *= 2
	}
	if d > tope {
		d = tope
	}
	return time.Duration(float64(d) * (0.5 + 0.5*azar))
}

func (c ConfigL3) maxCola() int {
	if c.MaxCola == nil {
		return L3MaxColaPorDefecto
	}
	return *c.MaxCola
}

func (c ConfigL3) multiMaxOpciones() int {
	if c.MultiMaxOpciones > 0 {
		return c.MultiMaxOpciones
	}
	return 16
}

func (c ConfigL3) maxExtra() int {
	if c.MaxExtra == nil {
		return maxExtraL3
	}
	return *c.MaxExtra
}

func (c ConfigL3) timeoutFila() time.Duration {
	if c.TimeoutFila > 0 {
		return time.Duration(c.TimeoutFila)
	}
	if c.Timeout > 0 {
		return time.Duration(c.Timeout)
	}
	return L3TimeoutPorDefecto
}
