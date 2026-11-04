package decide

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"

	"chispa/format/chispa3"
	"chispa/runtime/l1"
	"github.com/juan52878911/chispa/server/torre"
)

// TokenizadorPasaje es el tokenizador de e5 con «passage: » (runtime/tok.TokenizarPasaje) que necesita la torre al vuelo.
// El Tokenizador que se pasa a ArmarDesde lo puede implementar además de IDs; sin él no se monta la torre.
type TokenizadorPasaje = torre.Tokenizador

// TokenizadorMaxLen lo implementa el tokenizador cuyo truncado se puede ajustar: ConMaxLen devuelve una copia que trunca la
// consulta (IDs) a consulta tokens y el pasaje (IDsPasaje) a pasaje. ArmarDesde lo llama con el `chispa.consulta.max_len` del
// ONNX de L2 y el `max_len` de TORR; un tokenizador que no lo implemente se queda con su truncado (128).
type TokenizadorMaxLen interface {
	ConMaxLen(consulta, pasaje int) Tokenizador
}

// ajustarMaxLen aplica el truncado de los metadatos al tokenizador. Con los dos en el defecto no toca nada.
func ajustarMaxLen(tok Tokenizador, consulta, pasaje int, avisos *[]string) Tokenizador {
	if tok == nil || (consulta == chispa3.MaxLenDefecto && pasaje == chispa3.MaxLenDefecto) {
		return tok
	}
	if t, ok := tok.(TokenizadorMaxLen); ok {
		return t.ConMaxLen(consulta, pasaje)
	}
	*avisos = append(*avisos, fmt.Sprintf("el tokenizador no admite truncado ajustable: sigue en %d (los metadatos piden consulta %d, pasaje %d)",
		chispa3.MaxLenDefecto, consulta, pasaje))
	return tok
}

// avisarCalib apunta y registra en el log los tipos que faltan en la CALIB de una capa.
func (m *Montaje) avisarCalib(v VistaCalib) {
	if a := v.AvisoSinCalib(); a != "" {
		m.AvisosCalib = append(m.AvisosCalib, a)
		log.Print(a)
	}
}

// OpcionArmado ajusta ArmarDesde; sirve para sustituir ORT por piezas falsas en las pruebas.
type OpcionArmado func(*opcionesArmado)

type opcionesArmado struct {
	ses    SesionS
	torres CodificadorTorre
}

// ConSesionS monta L2 sobre esta sesión en lugar de abrir el ONNX de la sección L2 con ORT.
func ConSesionS(s SesionS) OpcionArmado { return func(o *opcionesArmado) { o.ses = s } }

// ConCodificadorTorre usa este codificador de torres en lugar de abrir el ONNX de la sección TORR con ORT.
func ConCodificadorTorre(c CodificadorTorre) OpcionArmado {
	return func(o *opcionesArmado) { o.torres = c }
}

// Montaje es lo que sale de armar la cascada desde un .chispa v3 (o v2 envuelto).
type Montaje struct {
	L1, L2 Decisor
	Avisos []string // capas que se quedan como stub y por qué
	// AvisosCalib: tipos sin CALIB por capa (una línea por capa). ArmarDesde también los escribe en el log, así que salen al arrancar y en cada
	// recarga; van aparte de Avisos porque no son un fallo de montaje.
	AvisosCalib []string
	// Cerrar libera las sesiones (L2 y torre) que abrió ArmarDesde; nil si no hay nada que cerrar. Idempotente.
	Cerrar func() error
	// Calibracion es CALIB.calibracion (por capa) y Calibradas las capas con calibración: las de Calibracion y L2 si CALIB trae
	// tipos_l2. Se pasan a Cascada.ConCalibracion. Hoy el exportador solo calibra L2; L0, L1 y L3 salen con «calibrada»: false.
	Calibracion map[string]chispa3.CalibCapa
	Calibradas  map[string]bool
}

// ArmarDesde añade a l0 el modelo de la sección L0TF, o el de L0 si no hay, y los de L0MULTI (uno por pregunta few-shot), y monta L1 y L2 con las secciones presentes;
// las que faltan quedan como stub. Si el .chispa trae CALIB, el τ de L1 y L2 es siempre el de CALIB (tau_escalate; el de L2 es
// tau_escalate_l2 si existe, y tau_l2c si L2 es L2c y CALIB lo trae): cfg se modifica. La config solo puede subirlo, si su JSON trae
// «capas» (Config.CapasExplicitas), y eso se registra en el log; un τ de la config por debajo del de CALIB se ignora con aviso.
// tok es el tokenizador de e5 que necesita L2 (puede ser nil).
// Si el .chispa trae la sección TORR, L2 calcula al vuelo las torres de los esquemas que ESQUEMAS no trae (L2.ConTorres,
// caché de cfg.CacheTorres torres, tope de cfg.MaxTorresNuevas por petición); sin ella, un esquema nuevo queda sin opinión de L2 como antes. Hace falta que tok
// implemente TokenizadorPasaje: si no, la torre no se monta y se avisa.
// f debe seguir abierto mientras se use la cascada.
func ArmarDesde(f *chispa3.Fichero, l0 *L0, cfg *Config, tok Tokenizador, opts ...OpcionArmado) (Montaje, error) {
	var op opcionesArmado
	for _, o := range opts {
		o(&op)
	}
	mt := Montaje{L1: NuevoL1(), L2: NuevoL2()}
	esL2c := false
	// L0TF (el que entrenó la cabeza conjunta) manda sobre el L0 de kindling del mismo fichero; L0MULTI añade los demás, uno por pregunta.
	if err := l0.AnadirDe(f); err != nil {
		return mt, err
	}
	textoTorre := chispa3.TextoTorreV1
	motivoSinTorre := ""       // META.texto_torre y TORR discrepan: el modelo carga sin torre al vuelo, con aviso (L0/L1/L2 con ESQUEMAS siguen)
	if f.Tiene(chispa3.TORR) { // la versión efectiva sale de TORR; si META trae texto_torre y discrepa, se degrada
		v, err := f.TextoTorreEfectivo()
		switch {
		case errors.Is(err, chispa3.ErrTextoTorreInconsistente):
			motivoSinTorre = chispa3.MotivoTextoTorreInconsistente
			aviso := "AVISO: torre al vuelo desactivada (" + motivoSinTorre + "): " + err.Error() +
				"; L2 decide solo los esquemas de ESQUEMAS. Reentrene con `chispa train` (escribe META y TORR coherentes)"
			mt.Avisos = append(mt.Avisos, aviso)
			log.Print(aviso)
		case err != nil:
			return mt, err
		default:
			textoTorre = v
		}
	}
	var calib chispa3.Calib
	if f.Tiene(chispa3.CALIB) {
		var err error
		if calib, err = f.Calib(); err != nil {
			return mt, fmt.Errorf("sección CALIB: %w", err)
		}
		fijarTauDeCalib(cfg, "L1", "tau_escalate", calib.TauEscalate, &mt.Avisos)
		mt.Calibracion, mt.Calibradas = calib.Calibracion, map[string]bool{}
		for n := range calib.Calibracion {
			mt.Calibradas[n] = true
		}
		if calib.TiposL2 != nil {
			mt.Calibradas["L2"] = true
		}
	}
	var sopl *chispa3.Sopl1
	lambdaL1 := cfg.LambdaL1()
	if f.Tiene(chispa3.SOPL1) && cfg.L1ShrinkageLambda == nil && !cfg.L1ShrinkageApagado && calib.L1LambdaCalib == nil {
		// .chispa anterior a l1_lambda_calib: su CALIB se calibró sin encogimiento, así que el λ por defecto lo descalibraría.
		// Sin λ de L1 explícito en la config, el soporte de L1 queda apagado (el comportamiento de antes).
		lambdaL1 = nil
		mt.Avisos = append(mt.Avisos, fmt.Sprintf("AVISO: el .chispa trae SOPL1 pero su CALIB no trae l1_lambda_calib (calibrado sin encogimiento): "+
			"L1 no mezcla el soporte; ponga l1_shrinkage_lambda en la config para activarlo (λ=%g por defecto en modelos nuevos)", L1ShrinkageLambdaPorDefecto))
	}
	explicito := cfg.L1ShrinkageLambda != nil || cfg.L2ShrinkageLambda != nil // el operador pidió el soporte: SOPL1 ilegible es error
	if f.Tiene(chispa3.SOPL1) && explicito {
		var err error
		if sopl, err = f.Sopl1(); err != nil {
			return mt, fmt.Errorf("sección SOPL1: %w", err)
		}
	} else if f.Tiene(chispa3.SOPL1) {
		// Sin λ explícito (solo el default de L1, o apagado) una SOPL1 ilegible no impide arrancar: se ignora con aviso.
		if s, err := f.Sopl1(); err != nil {
			mt.Avisos = append(mt.Avisos, "SOPL1 inválida, se ignora: "+err.Error())
		} else {
			sopl = s
		}
	}
	if f.Tiene(chispa3.L1) {
		b, err := f.L1()
		if err != nil {
			return mt, fmt.Errorf("sección L1: %w", err)
		}
		m, err := l1.Desde(b)
		if err != nil {
			return mt, fmt.Errorf("sección L1: %w", err)
		}
		l1c := NuevoL1Modelo(m, calib, cfg.CacheEsquemas).ConSoporte(sopl, lambdaL1)
		l1c.sinCalibFichero = !f.Tiene(chispa3.CALIB)
		mt.avisarCalib(VistaCalibCapa("L1", calib, OrigenFichero, !l1c.sinCalibFichero))
		if cfg.TextoLargo != nil {
			l1c.ConTextoLargo(*cfg.TextoLargo)
		}
		mt.L1 = l1c
		if sopl != nil && lambdaL1 != nil {
			switch {
			case calib.L1LambdaCalib == nil:
				mt.Avisos = append(mt.Avisos, "AVISO: L1 mezcla SOPL1 con λ explícito pero la CALIB se calibró sin encogimiento (sin l1_lambda_calib): la cobertura y precisión prometidas no valen")
			case *calib.L1LambdaCalib != *lambdaL1:
				mt.Avisos = append(mt.Avisos, fmt.Sprintf("AVISO: λ de L1 del runtime (%g) distinto del de la CALIB (l1_lambda_calib=%g): τ de L1 descalibrado", *lambdaL1, *calib.L1LambdaCalib))
			}
		} else if sopl != nil && calib.L1LambdaCalib != nil {
			mt.Avisos = append(mt.Avisos, fmt.Sprintf("AVISO: L1 sin soporte (apagado) pero la CALIB se calibró con λ=%g (l1_lambda_calib): τ de L1 descalibrado", *calib.L1LambdaCalib))
		}
		if sopl != nil && sopl.Dim() != m.Dim() {
			mt.Avisos = append(mt.Avisos, fmt.Sprintf("SOPL1 de dimensión %d y L1 de %d: L1 no mezcla el soporte", sopl.Dim(), m.Dim()))
		}
	}
	if f.Tiene(chispa3.L2) && f.Tiene(chispa3.ESQUEMAS) {
		onnx, err := f.L2()
		if err != nil {
			return mt, fmt.Errorf("sección L2: %w", err)
		}
		esq, err := f.Esquemas()
		if err != nil {
			return mt, fmt.Errorf("sección ESQUEMAS: %w", err)
		}
		var conj *chispa3.Conj
		if f.Tiene(chispa3.CONJ) {
			if conj, err = f.Conj(); err != nil {
				return mt, fmt.Errorf("sección CONJ: %w", err)
			}
			if conj.Version() != esq.Version() {
				mt.Avisos = append(mt.Avisos, fmt.Sprintf("AVISO: ESQUEMAS v%d y CONJ v%d: sus claves no se mezclan y L2c no encontrará los esquemas de CONJ; reentrene con --conjunta", esq.Version(), conj.Version()))
			}
		}
		mq, err := chispa3.MaxLenConsulta(onnx)
		if err != nil {
			return mt, fmt.Errorf("sección L2: %w", err)
		}
		mp := chispa3.MaxLenDefecto
		if f.Tiene(chispa3.TORR) {
			if t, err := f.Torre(); err == nil { // si es ilegible, montarTorres da el error
				mp = t.MaxLen
			}
		}
		tok = ajustarMaxLen(tok, mq, mp, &mt.Avisos)
		base, err := montarL2(onnx, esq, calib.ParaL2(), tok, op.ses) // L2 lee tipos_l2 si existen; L1 siempre tipos
		if err != nil {
			mt.Avisos = append(mt.Avisos, "L2 queda como stub: "+err.Error())
		} else {
			base.sinCalibFichero = !f.Tiene(chispa3.CALIB)
			or := OrigenFichero
			if calib.TiposL2 != nil {
				or = OrigenTiposL2
			}
			base.origenCalib = or
			mt.avisarCalib(VistaCalibCapa("L2", calib.ParaL2(), or, !base.sinCalibFichero))
			cerrables := []any{base.ses}
			base.ConMotivoSinTorre(motivoSinTorre)
			if f.Tiene(chispa3.TORR) && motivoSinTorre == "" && cfg.maxOpcionesEfectivo() > MaxOpcionesTope {
				// la caché de torres guarda 16 × max_opciones: con más del máximo, la torre al vuelo se desactiva en vez de fallar
				motivoSinTorre = MotivoMaxOpcionesExcesivo
				base.ConMotivoSinTorre(motivoSinTorre)
				mt.Avisos = append(mt.Avisos, fmt.Sprintf("AVISO: torre al vuelo desactivada (%s): limites.max_opciones (%d) pasa del máximo %d para la caché de torres (16 × max_opciones); el modelo carga sin torres al vuelo",
					motivoSinTorre, cfg.maxOpcionesEfectivo(), MaxOpcionesTope))
			}
			if f.Tiene(chispa3.TORR) && motivoSinTorre == "" {
				base.ConTextoTorre(textoTorre)
				// Con TORR, la caché debe caber la pregunta más grande con margen por los fragmentos del LRU (16 × max_opciones):
				// si la config pide menos, se sube el valor efectivo en vez de fallar (sin TORR no se exige nada).
				cacheT, pedida := cfg.cacheTorresEfectiva(), 0
				if min := cfg.cacheTorresMinima(); cacheT < min {
					aviso := fmt.Sprintf("AVISO: cache_torres (%d) < %d (16 × max_opciones=%d, un fragmento del LRU debe admitir una pregunta entera): se sube a %d para que las preguntas de esquema nuevo con torres al vuelo progresen",
						cacheT, min, cfg.maxOpcionesEfectivo(), min)
					mt.Avisos = append(mt.Avisos, aviso) // lo escribe el llamador (main.go lo hace con todos los avisos)
					cacheT, pedida = min, cfg.cacheTorresEfectiva()
				}
				fuente := f
				if cfg.TorreFichero != "" && op.torres == nil {
					alt, err := abrirTorreAlterna(cfg, f, textoTorre)
					if err != nil {
						cierraTodo(cerrables...)() // la sesión de L2 ya está abierta
						return mt, err
					}
					fuente = alt
					cerrables = append(cerrables, alt)
					mt.Avisos = append(mt.Avisos, "torres al vuelo con la TORR de torre_fichero "+cfg.TorreFichero+
						" (p. ej. int8); las torres de ESQUEMAS siguen siendo las del -modelo")
				}
				if err := montarTorres(fuente, base, tok, op.torres, cacheT, cfg.MaxTorresNuevas, cfg.TorresPorSegundo); err != nil {
					mt.Avisos = append(mt.Avisos, "sin torres al vuelo: "+err.Error())
				} else {
					base.ConCacheTorresEfectiva(cacheT, pedida)
				}
			}
			cerrables = append(cerrables, base.torres)
			if cfg.TextoLargo != nil {
				base.ConTextoLargo(*cfg.TextoLargo)
			}
			mt.Cerrar = cierraTodo(cerrables...)
			mt.L2 = Decisor(base)
			if conj != nil {
				mt.L2 = NuevoL2cCon(base, base, conj).ConSoporte(sopl, cfg.L2ShrinkageLambda)
			}
			esL2c = conj != nil
			if cfg.TauL2EsquemaNuevo == nil && !cfg.TauL2EsquemaNuevoApagado {
				v := TauL2EsquemaNuevoPorDefecto
				if calib.TauL2EsquemaNuevo != nil {
					v = *calib.TauL2EsquemaNuevo
				}
				cfg.TauL2EsquemaNuevo = &v
			} else if cfg.TauL2EsquemaNuevo != nil && calib.TauL2EsquemaNuevo != nil && *cfg.TauL2EsquemaNuevo < *calib.TauL2EsquemaNuevo {
				// Como los τ de capa: la config solo puede subir el de CALIB.
				mt.Avisos = append(mt.Avisos, fmt.Sprintf("AVISO: tau_l2_esquema_nuevo: la config pide %g, por debajo de %g (CALIB): se ignora (la config solo puede subir el τ)",
					*cfg.TauL2EsquemaNuevo, *calib.TauL2EsquemaNuevo))
				v := *calib.TauL2EsquemaNuevo
				cfg.TauL2EsquemaNuevo = &v
			}
		}
	}
	if f.Tiene(chispa3.CALIB) {
		tauL2, clave := calib.ParaL2().TauEscalate, "tau_escalate"
		if calib.TauEscalateL2 != nil {
			clave = "tau_escalate_l2"
		}
		if esL2c && calib.TauL2c != nil {
			tauL2, clave = *calib.TauL2c, "tau_l2c" // el tau_escalate se calibró para L2 solo
		}
		fijarTauDeCalib(cfg, "L2", clave, tauL2, &mt.Avisos)
	}
	return mt, nil
}

// montarL2 monta el L2 base sobre ses, o sobre el ONNX de la sección L2 si ses es nil.
func montarL2(onnx []byte, esq *chispa3.Esquemas, calib chispa3.Calib, tok Tokenizador, ses SesionS) (*L2, error) {
	if tok == nil {
		return nil, ErrSinTokenizador
	}
	if ses == nil {
		var err error
		if ses, err = NuevaSesionORT(onnx); err != nil {
			return nil, err
		}
	}
	return NuevoL2Con(ses, esq, calib, tok), nil
}

// montarTorres lee la sección TORR y activa las torres al vuelo en l2; enc, si no es nil, sustituye al ONNX.
func montarTorres(f *chispa3.Fichero, l2 *L2, tok Tokenizador, enc CodificadorTorre, cache, maxNuevas int, torresPorSeg float64) error {
	t, err := f.Torre()
	if err != nil {
		return fmt.Errorf("sección TORR: %w", err)
	}
	if t.Dim != l2.esq.Dim() {
		return fmt.Errorf("sección TORR: dimensión %d, pero las ranuras de L2 son de %d", t.Dim, l2.esq.Dim())
	}
	b, err := f.Bytes(chispa3.TORR)
	if err != nil {
		return err
	}
	// TORR v2 exige el mismo prefijo que v1 (torre.ValidarPrefijo conoce las dos); una versión desconocida ya la rechazó Torre().
	vTorr := binary.LittleEndian.Uint32(b)
	if err := torre.ValidarPrefijo(vTorr, t.Prefijo); err != nil {
		return err
	}
	if enc == nil {
		tp, ok := tok.(TokenizadorPasaje)
		if !ok {
			return fmt.Errorf("el tokenizador no ofrece el modo passage")
		}
		if enc, err = torre.Abrir(t, tp); err != nil {
			return err
		}
	}
	if cache <= 0 {
		cache = CacheTorresPorDefecto
	}
	l2.ConTorres(enc, cache).ConMaxTorresNuevas(maxNuevas).ConTorresPorSegundo(torresPorSeg)
	return nil
}

// abrirTorreAlterna abre torre_fichero, un .chispa del que solo se usa la sección TORR (p. ej. la torre int8 del perfil movil-int8
// para las torres al vuelo del servidor). Su TORR debe ser compatible con la del -modelo: misma dimensión, max_len, prefijo y
// versión del texto de torre; si no, es un error de carga (no se degrada en silencio).
func abrirTorreAlterna(cfg *Config, f *chispa3.Fichero, textoTorre string) (*chispa3.Fichero, error) {
	alt, err := chispa3.AbrirCon(cfg.TorreFichero, chispa3.Limites{MaxFichero: cfg.MaxFicheroBytes()})
	if err != nil {
		return nil, fmt.Errorf("torre_fichero: %w", err)
	}
	falla := func(format string, a ...any) (*chispa3.Fichero, error) {
		alt.Close()
		return nil, fmt.Errorf("torre_fichero %s: "+format, append([]any{cfg.TorreFichero}, a...)...)
	}
	if !alt.Tiene(chispa3.TORR) {
		return falla("no trae sección TORR")
	}
	ta, err := alt.Torre()
	if err != nil {
		return falla("sección TORR: %v", err)
	}
	tm, err := f.Torre()
	if err != nil {
		return falla("TORR del -modelo: %v", err)
	}
	if ta.Dim != tm.Dim || ta.MaxLen != tm.MaxLen || ta.Prefijo != tm.Prefijo {
		return falla("TORR incompatible con la del -modelo (dim %d/%d, max_len %d/%d, prefijo %q/%q)", ta.Dim, tm.Dim, ta.MaxLen, tm.MaxLen, ta.Prefijo, tm.Prefijo)
	}
	if v, err := alt.TextoTorreEfectivo(); err != nil || v != textoTorre {
		return falla("texto de torre %q, el -modelo usa %q (%v)", v, textoTorre, err)
	}
	return alt, nil
}

// fijarTauDeCalib pone en la capa nombre el τ de CALIB (clave es su nombre en CALIB). La config solo puede subirlo: con
// CapasExplicitas y un τ mayor manda el de la config; con uno menor se ignora. Los dos casos van a avisos.
func fijarTauDeCalib(cfg *Config, nombre, clave string, tauCalib float64, avisos *[]string) {
	for i := range cfg.Capas {
		if cfg.Capas[i].Nombre != nombre {
			continue
		}
		pedido := cfg.Capas[i].Tau
		cfg.Capas[i].Tau = tauCalib
		if !cfg.CapasExplicitas || pedido == tauCalib {
			return
		}
		var aviso string
		if pedido > tauCalib {
			cfg.Capas[i].Tau = pedido
			aviso = fmt.Sprintf("τ de %s: la config lo sube de %g (%s de CALIB) a %g", nombre, tauCalib, clave, pedido)
		} else {
			aviso = fmt.Sprintf("AVISO: τ de %s: la config pide %g, por debajo de %g (%s de CALIB): se ignora (la config solo puede subir el τ)", nombre, pedido, tauCalib, clave)
		}
		*avisos = append(*avisos, aviso) // quien arma (chispa-server) escribe Avisos en el log
		return
	}
}

// cierraTodo devuelve una función que cierra, una sola vez, los io.Closer de la lista (los que no lo son se saltan).
func cierraTodo(xs ...any) func() error {
	var once sync.Once
	var err error
	return func() error {
		once.Do(func() {
			for _, x := range xs {
				if c, ok := x.(io.Closer); ok {
					err = errors.Join(err, c.Close())
				}
			}
		})
		return err
	}
}
