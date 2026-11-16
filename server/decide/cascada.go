package decide

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"chispa/format/chispa3"
)

// Resultado es la decisión final de una pregunta.
type Resultado struct {
	ID     string             `json:"id"`
	Kind   string             `json:"kind"`
	Answer string             `json:"answer"`
	Probs  map[string]float64 `json:"probs"`
	// Answers: solo en multi, las opciones activas (puede ir vacía); en los demás tipos no sale.
	Answers  []string `json:"answers,omitempty"`
	Layer    string   `json:"layer"`
	Escalate bool     `json:"escalate"`
	// EscalateReason: por qué escaló (solo con escalate); ver MotivoSinOpinion, MotivoBajaConfianza, MotivoScoreIdsNoValidos y MotivoIncompleta.
	EscalateReason string `json:"escalate_reason,omitempty"`
	// Calibrada: la capa que contestó tiene calibración (CALIB.calibracion, o tipos_l2 en L2). Solo sale si la cascada sabe qué capas
	// lo están (Cascada.ConCalibracion) y una capa contestó; false = su confianza no está calibrada y el τ se cortó sobre la cruda.
	Calibrada *bool `json:"calibrada,omitempty"`
	// Truncado: alguna capa que corrió vio solo una parte del texto (pasaba de su max_tokens o max_len y la estrategia no es
	// «trozos»); TokensOriginales son los tokens del texto entero (el mayor entre las capas). Solo salen con truncado.
	Truncado         bool `json:"truncado,omitempty"`
	TokensOriginales int  `json:"tokens_originales,omitempty"`
	// UnkPct: fracción de tokens <unk> del texto (la mayor entre las capas que tokenizaron); no sale si ninguna tokenizó.
	UnkPct *float64 `json:"unk_pct,omitempty"`
	Ms     float64  `json:"ms"`
	Nonce  string   `json:"nonce,omitempty"`
	// Experimental: el tipo de la pregunta (score) no es estable: la exactitud medida está cerca del azar (T3: jql 0,15 y usajobs 0,15). Solo sale con true.
	Experimental bool `json:"experimental,omitempty"`
	// Incompleto: una capa falló o el ctx se canceló; no debe guardarse bajo un nonce.
	Incompleto bool `json:"-"`
}

// MarshalJSON añade «answers» solo en multi (y siempre, aunque vacía: [] = ninguna opción activa).
func (r Resultado) MarshalJSON() ([]byte, error) {
	type plano Resultado // sin métodos: evita la recursión
	out := struct {
		plano
		Answers *[]string `json:"answers,omitempty"`
	}{plano: plano(r)}
	if r.Kind == KindMulti {
		a := r.Answers
		if a == nil {
			a = []string{}
		}
		out.Answers = &a
	}
	return json.Marshal(out)
}

// CapaEscalar es el valor de «layer» cuando ninguna capa acepta: no contestó ninguna, así que no se atribuye a la última.
const CapaEscalar = "escalar"

// Valores de «escalate_reason». sin_opinion: ninguna capa opinó (todas «sin opinión», sin error: el modelo no puede decidir ese
// esquema: sin torre, tipo no soportado…) y las probs van vacías. score_ids_no_validos: sin opinión porque los ids del score no son
// 1..K. baja_confianza: alguna capa opinó pero ninguna llegó a su τ.
// incompleta: ninguna capa opinó y alguna falló o se quedó sin presupuesto (no se sabe si habría opinado).
const (
	MotivoSinOpinion           = "sin_opinion"
	MotivoSinPresupuestoTorres = "sin_presupuesto_torres" // ninguna capa opinó y L2 agotó el presupuesto de torres nuevas
	MotivoIncompleta           = "incompleta"             // una capa falló o se quedó sin presupuesto y ninguna otra opinó: no es «nadie opinó»
	MotivoScoreIdsNoValidos    = "score_ids_no_validos"
	MotivoBajaConfianza        = "baja_confianza"
)

type capa struct {
	d   Decisor
	tau float64
	// tauNuevo, si no es nil, sustituye a tau cuando la respuesta de la capa trae EsquemaNuevo.
	tauNuevo *float64
	// cal, si no es nil, calibra la confianza top-1 antes de compararla con τ (CALIB.calibracion de la capa).
	cal       *chispa3.CalibCapa
	calibrada bool
}

// Cascada pasa por las capas en orden mientras la confianza top-1 sea < τ.
type Cascada struct {
	capas []capa
	met   *Metricas
	// vistas son las vistas de CALIB (con los τ efectivos) preparadas al armar; PublicaCalib las pone en /metrics.
	vistas map[string]VistaCalib
	// tauNuevo es el τ de esquema nuevo de la config; el gauge lo fija Recargar al hacer el cambio (nuevoVivo), no el armado.
	tauNuevo *float64
	// marcaCalib: ConCalibracion se llamó y cada Resultado contestado lleva «calibrada».
	marcaCalib bool
	// omitidas son capas que el modelo no trae y la cascada se salta (L1 sin sección L1): Capacidades las da como inactivas.
	omitidas []string
	// saltadas: capas que la cascada no corre (config knn: L0 y L1, como en Python).
	saltadas map[string]bool
	// soloL3: prueba (config forzar_l3): se saltan las demás capas y L3 contesta con τ = 0. Ver SoloL3.
	soloL3 bool
}

// Saltar hace que la cascada no corra las capas dadas (config knn: L2(kNN)→L3 sin L0 ni L1).
func (c *Cascada) Saltar(nombres ...string) *Cascada {
	if c.saltadas == nil {
		c.saltadas = map[string]bool{}
	}
	for _, n := range nombres {
		c.saltadas[n] = true
	}
	return c
}

// SoloL3 salta todas las capas salvo L3 y acepta su respuesta sea cual sea su confianza (config forzar_l3, solo para pruebas).
func (c *Cascada) SoloL3() *Cascada {
	c.soloL3 = true
	return c
}

// ConOmitidas apunta capas que no están en la cascada porque el modelo no las trae (ver SinL1Ausente).
func (c *Cascada) ConOmitidas(nombres ...string) *Cascada {
	c.omitidas = append(c.omitidas, nombres...)
	return c
}

// ConCalibracion aplica la calibración por capa de CALIB (cal, por nombre de capa) y marca en cada respuesta si la capa que contestó
// está calibrada (calibradas). Sin llamarla, la cascada corta sobre la confianza cruda y no pone «calibrada». Antes de servir.
func (c *Cascada) ConCalibracion(cal map[string]chispa3.CalibCapa, calibradas map[string]bool) *Cascada {
	c.marcaCalib = true
	for i := range c.capas {
		n := c.capas[i].d.Nombre()
		if k, ok := cal[n]; ok {
			c.capas[i].cal = &k
		}
		c.capas[i].calibrada = calibradas[n]
	}
	return c
}

// NuevaCascada empareja decisores y umbrales por posición y nombre.
func NuevaCascada(cfg []Capa, ds []Decisor, met *Metricas) (*Cascada, error) {
	return NuevaCascadaConTauNuevo(cfg, ds, met, nil)
}

// NuevaCascadaConTauNuevo es NuevaCascada con un τ propio para L2 cuando la pregunta no estaba en ESQUEMAS
// (tauNuevo nil = el de la capa, como antes).
func NuevaCascadaConTauNuevo(cfg []Capa, ds []Decisor, met *Metricas, tauNuevo *float64) (*Cascada, error) {
	if len(cfg) != len(ds) || len(ds) == 0 {
		return nil, fmt.Errorf("cascada: %d capas en la config y %d decisores", len(cfg), len(ds))
	}
	c := &Cascada{met: met, tauNuevo: tauNuevo}
	// Las vistas de CALIB se preparan aparte: /metrics sigue mostrando las del modelo en servicio hasta PublicaCalib.
	met.iniciaPrepCalib()
	cerrada := false
	defer func() {
		if !cerrada {
			met.cierraPrepCalib()
		}
	}()
	for i, d := range ds {
		if u, ok := d.(UsaMetricas); ok && met != nil {
			u.UsaMetricas(met)
		}
		if d.Nombre() != cfg[i].Nombre {
			return nil, fmt.Errorf("cascada: la capa %d es %s pero la config dice %s", i, d.Nombre(), cfg[i].Nombre)
		}
		cp := capa{d: d, tau: cfg[i].Tau}
		if cfg[i].Nombre == "L2" {
			cp.tauNuevo = tauNuevo
		}
		c.capas = append(c.capas, cp)
	}
	c.publicaTaus(cfg)
	c.vistas = met.cierraPrepCalib()
	cerrada = true
	return c, nil
}

// Taus devuelve el τ efectivo de cada capa de la cascada, por nombre.
func (c *Cascada) Taus() map[string]float64 {
	m := make(map[string]float64, len(c.capas))
	for _, cp := range c.capas {
		m[cp.d.Nombre()] = cp.tau
	}
	return m
}

// PublicaCalib pone en /metrics la CALIB y los τ de esta cascada, y quita las series de las capas que ya no están. Se llama
// cuando la cascada ya está en servicio (arranque y Recargar tras sustituir el modelo), nunca al armarla.
func (c *Cascada) PublicaCalib() {
	if c == nil || c.met == nil {
		return
	}
	c.met.publicaCalib(c.vistas)
}

// publicaTaus pone en chispa_calib_tau los τ efectivos de L1 y L2 (config > CALIB > default) con su origen, y el τ de esquema nuevo.
func (c *Cascada) publicaTaus(cfg []Capa) {
	if c.met == nil {
		return
	}
	defecto := map[string]float64{}
	for _, d := range ConfigPorDefecto().Capas {
		defecto[d.Nombre] = d.Tau
	}
	vistas := c.met.calibPrep()
	for _, k := range cfg {
		v, ok := vistas[k.Nombre]
		if !ok || (k.Nombre != "L1" && k.Nombre != "L2") {
			continue
		}
		// candidatos de CALIB: los que la vista trae con origen calib
		var calib []float64
		for _, t := range v.Taus {
			if t.Origen == OrigenCalib {
				calib = append(calib, t.Valor)
			}
		}
		taus := []TauCalib{{Capa: k.Nombre, Clave: "tau_escalate", Valor: k.Tau, Origen: origenTau(k.Tau, defecto[k.Nombre], calib)}}
		// L2c con tau_l2c en servicio: k.Tau es el de la cabeza conjunta y tau_escalate es el de respaldo (L2 base).
		if k.Nombre == "L2" {
			for _, cp := range c.capas {
				if cr, ok := cp.d.(DecisorConRespaldo); ok && cp.d.Nombre() == "L2" {
					if tr, tc, ok := cr.TauRespaldo(); ok && cp.tau == tc {
						taus = []TauCalib{
							{Capa: k.Nombre, Clave: "tau_escalate", Valor: tr, Origen: origenTau(tr, defecto[k.Nombre], calib)},
							{Capa: k.Nombre, Clave: "tau_l2c", Valor: tc, Origen: OrigenCalib},
						}
					}
				}
			}
		}
		if k.Nombre == "L2" && c.tauNuevo != nil {
			var cn []float64
			for _, t := range v.Taus {
				if t.Clave == "tau_l2_esquema_nuevo" && t.Origen == OrigenCalib {
					cn = append(cn, t.Valor)
				}
			}
			taus = append(taus, TauCalib{Capa: k.Nombre, Clave: "tau_l2_esquema_nuevo", Valor: *c.tauNuevo, Origen: origenTau(*c.tauNuevo, TauL2EsquemaNuevoPorDefecto, cn)})
		}
		c.met.fijaTaus(k.Nombre, taus)
	}
}

// Decidir recorre las capas. Si una falla o no opina, se pasa a la siguiente.
// Si ninguna alcanza su τ, la cascada escala con layer «escalar»: la respuesta es la de mayor
// confianza vista (pista para quien reciba la escalada).
func (c *Cascada) Decidir(ctx context.Context, p Pregunta) Resultado {
	t0 := time.Now()
	res := Resultado{ID: p.ID, Kind: p.Kind, Nonce: p.Nonce, Probs: map[string]float64{}, Experimental: p.Kind == KindScore}
	var mejor Respuesta
	ninguna := len(c.capas) > 0 // sigue true mientras todas las capas vistas devuelvan «sin opinión» limpia (sin error ni Incompleto)
	sinPres := false            // alguna capa quedó sin presupuesto de torres nuevas
	otraFalla := false          // alguna capa falló por error o plazo, o quedó incompleta por otra causa que el presupuesto de torres
	previas := Previas{}
	var sen senalesRespuesta
	ctx = conPresupuesto(ctx)
	for _, l := range c.capas {
		if ctx.Err() != nil {
			res.Incompleto = true
			otraFalla = true // el plazo venció antes de correr esta capa: es incompleta, no sin_presupuesto_torres
			break
		}
		if (c.soloL3 && l.d.Nombre() != "L3") || c.saltadas[l.d.Nombre()] {
			continue
		}
		t := time.Now()
		var r Respuesta
		var err error
		respaldo := false
		if cr, ok := l.d.(DecisorConRespaldo); ok {
			r, respaldo, err = cr.DecidirConOrigen(ctx, p, previas)
		} else if cp, ok := l.d.(DecisorConPrevias); ok {
			r, err = cp.DecidirCon(ctx, p, previas)
		} else {
			r, err = l.d.Decidir(ctx, p)
		}
		c.met.ObservaCapa(l.d.Nombre(), time.Since(t), err)
		if err == nil && r.Senal != nil {
			sen.suma(*r.Senal)
		} else if cs, ok := l.d.(ConSenales); ok && err == nil {
			if s, ok := cs.Senales(p.Estado); ok {
				sen.suma(s)
			}
		}
		if err != nil || ctx.Err() != nil {
			res.Incompleto = true
			otraFalla = true
			r = Respuesta{}
		}
		if r.Incompleto {
			res.Incompleto = true
			sinPres = sinPres || r.SinPresupuestoTorres
			otraFalla = otraFalla || !r.SinPresupuestoTorres
		}
		previas[l.d.Nombre()] = r
		if r.Conf == 0 && !scoreValido(p) {
			c.met.ScoreIdsNoValidos(l.d.Nombre())
		}
		if r.Conf == 0 && err == nil && !r.Incompleto && ctx.Err() == nil {
			c.met.CapaNoOpina(l.d.Nombre(), causaNoOpina(l.d, p))
		} else if r.Incompleto && r.SinPresupuestoTorres {
			c.met.CapaNoOpina(l.d.Nombre(), CausaPresupuestoTorres)
			ninguna = false
		} else {
			ninguna = false
		}
		if r.Conf > mejor.Conf {
			mejor = r
		}
		tau := l.tau
		if c.soloL3 {
			tau = 0
		}
		if cr, ok := l.d.(DecisorConRespaldo); ok && respaldo && !(r.EsquemaNuevo && l.tauNuevo != nil) {
			// respaldo de L2c: τ de L2 base (tau_escalate_l2) salvo esquema nuevo con τ de esquema nuevo (igual que Kotlin/Swift), solo si el τ de la capa es el de la cabeza (tau_l2c de CALIB)
			if tr, tc, ok := cr.TauRespaldo(); ok && l.tau == tc {
				tau = tr
			}
		}
		if r.EsquemaNuevo {
			c.met.EsquemaNuevoPreguntas.Add(1)
			if l.tauNuevo != nil {
				tau = *l.tauNuevo
			}
		}
		if r.TauPropio != nil {
			tau = *r.TauPropio
		}
		conf := r.Conf
		if l.cal != nil && r.TauPropio == nil {
			conf = l.cal.Aplicar(conf)
		}
		if conf > 0 && conf >= tau {
			if r.EsquemaNuevo {
				c.met.EsquemaNuevoDecididas.Add(1)
			}
			res.Answer, res.Answers, res.Probs, res.Layer = r.Answer, r.Answers, r.Probs, l.d.Nombre()
			if c.marcaCalib {
				cb := l.calibrada
				res.Calibrada = &cb
			}
			c.met.Decidida(res.Layer, false)
			sen.aplica(&res)
			res.Ms = ms(time.Since(t0))
			return res
		}
	}
	if k, ok := previas["L2"]; ok && k.MotivoKNN != "" && res.Incompleto {
		c.met.knnIncompleta(k.Tarea) // el kNN escaló a L3 y L3 falló
	}
	res.Escalate = true
	res.Layer = CapaEscalar
	switch {
	case mejor.Conf > 0:
		res.EscalateReason = MotivoBajaConfianza
	case ninguna && !res.Incompleto:
		res.EscalateReason = MotivoSinOpinion
		if !scoreValido(p) {
			res.EscalateReason = MotivoScoreIdsNoValidos
		}
		c.met.NingunaCapaOpina()
	case sinPres && !otraFalla: // prioridad: incompleta (error/plazo de otra capa) gana a sin_presupuesto_torres
		res.EscalateReason = MotivoSinPresupuestoTorres
	case res.Incompleto:
		res.EscalateReason = MotivoIncompleta
	}
	res.Answer, res.Answers = mejor.Answer, mejor.Answers
	if mejor.Probs != nil {
		res.Probs = mejor.Probs
	}
	c.met.Decidida(res.Layer, true)
	sen.aplica(&res)
	res.Ms = ms(time.Since(t0))
	return res
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
