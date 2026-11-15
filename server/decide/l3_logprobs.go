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
	"strings"
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
		return nil, nil, errMasDe62
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

// postLP hace la llamada con 1 reintento ante error de red o 5xx (no ante timeout de la fila ni 4xx).
func (l *L3LLM) postLP(ctx context.Context, c cuerpoLPL3) ([]byte, error) {
	cuerpo, _ := json.Marshal(c)
	var err error
	for intento := 0; intento < 2; intento++ {
		var b []byte
		var reintentable bool
		if b, reintentable, err = l.postUnaVez(ctx, cuerpo); err == nil {
			return b, nil
		}
		if !reintentable || ctx.Err() != nil {
			break
		}
	}
	return nil, err
}

func (l *L3LLM) postUnaVez(ctx context.Context, cuerpo []byte) ([]byte, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.url, bytes.NewReader(cuerpo))
	if err != nil {
		return nil, false, errors.New("L3: petición no válida")
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
		return nil, true, fmt.Errorf("L3: %w", err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxRespuestaL3))
	if err != nil {
		return nil, true, fmt.Errorf("L3: %w", err)
	}
	if resp.StatusCode/100 != 2 {
		return nil, resp.StatusCode >= 500, fmt.Errorf("%w (%d)", errL3Estado, resp.StatusCode)
	}
	return b, false, nil
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
// a mitad del relleno, usa lo que ya hay; si vence sin nada, devuelve el error del contexto.
func (l *L3LLM) puntuarLP(ctx context.Context, msgs []msgL3, cods []string) ([]float64, bool, error) {
	top, todos, err := l.primerTokenLP(ctx, msgs)
	if err != nil {
		return nil, false, err
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
		return nil, false, ctx.Err()
	}
	piso := math.Inf(1)
	for _, v := range todos {
		piso = math.Min(piso, v)
	}
	for _, v := range lp {
		piso = math.Min(piso, v)
	}
	piso--
	xs := make([]float64, len(cods))
	mx := math.Inf(-1)
	for i, c := range cods {
		v, esta := lp[c]
		if !esta {
			v = piso
		}
		xs[i] = v
		mx = math.Max(mx, v)
	}
	var s float64
	for i := range xs {
		xs[i] = math.Exp(xs[i] - mx)
		s += xs[i]
	}
	for i := range xs {
		xs[i] /= s
	}
	return xs, true, nil
}

// decidirLogprobs es Decidir en modo logprobs; ordenadas ya viene en orden de nivel en score y con el presupuesto reservado.
func (l *L3LLM) decidirLogprobs(ctx context.Context, p Pregunta, ordenadas []Opcion) (Respuesta, error) {
	msgs, cods, err := mensajesLogprobsL3(p, ordenadas)
	if err != nil {
		l.met.l3Cuenta("invalida")
		return Respuesta{Incompleto: true}, nil
	}
	// Semáforo: la espera por un hueco la acota el contexto de la petición; el tope de la fila empieza al obtenerlo.
	select {
	case l.sem <- struct{}{}:
		defer func() { <-l.sem }()
	case <-ctx.Done():
		l.met.l3Cuenta("timeout")
		return Respuesta{}, fmt.Errorf("%w: %v", ErrSinL3, ctx.Err())
	}
	cf, cancel := context.WithTimeout(ctx, l.cfg.timeoutFila())
	defer cancel()
	t := time.Now()
	xs, ok, err := l.puntuarLP(cf, msgs, cods)
	l.met.l3Latencia(time.Since(t))
	if err == nil && !ok && errors.Is(cf.Err(), context.DeadlineExceeded) {
		err = cf.Err()
	}
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || os.IsTimeout(err) {
			l.met.l3Cuenta("timeout")
		} else {
			l.met.l3Cuenta("error")
		}
		return Respuesta{}, fmt.Errorf("%w: %v", ErrSinL3, err)
	}
	if !ok {
		l.met.l3Cuenta("invalida")
		return Respuesta{Incompleto: true}, nil
	}
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
	l.met.l3Cuenta("ok")
	return Respuesta{Answer: ans, Probs: probs, Conf: probs[ans]}, nil
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
