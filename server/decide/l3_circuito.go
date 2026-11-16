package decide

import (
	"fmt"
	"sync"
	"time"
)

// Errores de L3 que cuelgan de ErrSinL3 (errors.Is(err, ErrSinL3) sigue valiendo): la cascada trata cualquiera como «L3 no opina».
var (
	ErrColaLlena       = fmt.Errorf("%w: cola llena", ErrSinL3)
	ErrCircuitoAbierto = fmt.Errorf("%w: circuito abierto", ErrSinL3)
)

// Cortacircuitos de L3. Una «fila» es una pregunta entera (con sus reintentos y su relleno): cuenta como fallo si acaba en error de
// red, 5xx, 429 sin remedio o tope de la fila vencido sin nada puntuado; una respuesta inválida del LLM o parcial con piso NO es fallo
// (el LLM contestó).
//
//	cerrado ──(circuito_fallos seguidos, o tasa ≥ circuito_tasa en las últimas circuito_ventana filas)──▶ abierto
//	abierto ──(pasado circuito_espera)──▶ semiabierto: deja pasar UNA fila de prueba; las demás se descartan como si siguiera abierto
//	semiabierto ──prueba ok──▶ cerrado        semiabierto ──prueba fallida──▶ abierto (otra espera)
const (
	circCerrado = iota
	circAbierto
	circSemiabierto
)

const (
	CircuitoFallosPorDefecto  = 5
	CircuitoEsperaPorDefecto  = 10 * time.Second
	CircuitoVentanaPorDefecto = 20
)

type circuito struct {
	mu      sync.Mutex
	apagado bool
	maxSegu int
	espera  time.Duration
	tasa    float64
	ventana []bool // anillo de los últimos resultados (true = fallo)
	pos, n  int
	segu    int // fallos seguidos
	estado  int
	hasta   time.Time
	sonda   bool // hay una fila de prueba en vuelo
	ahora   func() time.Time
	met     *Metricas
}

func nuevoCircuito(c ConfigL3) *circuito {
	k := &circuito{apagado: c.CircuitoApagado, maxSegu: c.CircuitoFallos, espera: time.Duration(c.CircuitoEspera), tasa: c.CircuitoTasa, ahora: time.Now}
	if k.maxSegu == 0 {
		k.maxSegu = CircuitoFallosPorDefecto
	}
	if k.espera == 0 {
		k.espera = CircuitoEsperaPorDefecto
	}
	if k.tasa > 0 {
		v := c.CircuitoVentana
		if v == 0 {
			v = CircuitoVentanaPorDefecto
		}
		k.ventana = make([]bool, v)
	}
	return k
}

func (k *circuito) conMetricas(m *Metricas) {
	k.mu.Lock()
	k.met = m
	m.l3CircEstado.Store(int64(k.estado))
	k.mu.Unlock()
}

func (k *circuito) fijaEstado(e int) {
	k.estado = e
	if k.met != nil {
		k.met.l3CircEstado.Store(int64(e))
	}
}

func (k *circuito) abre() {
	k.fijaEstado(circAbierto)
	k.hasta = k.ahora().Add(k.espera)
	k.sonda = false
	if k.met != nil {
		k.met.l3CircAperturas.Add(1)
	}
}

// permite dice si la fila puede llamar al LLM; sonda=true si es la fila de prueba del semiabierto (debe llamar a registra).
func (k *circuito) permite() (ok, sonda bool) {
	if k == nil || k.apagado {
		return true, false
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	switch k.estado {
	case circAbierto:
		if k.ahora().Before(k.hasta) {
			return false, false
		}
		k.fijaEstado(circSemiabierto)
		k.sonda = true
		return true, true
	case circSemiabierto:
		if k.sonda {
			return false, false
		}
		k.sonda = true
		return true, true
	}
	return true, false
}

// registra anota el resultado de una fila que permite dejó pasar. cancelada: el cliente se fue (no cuenta ni como éxito ni como fallo;
// solo libera la prueba).
func (k *circuito) registra(ok, sonda, cancelada bool) {
	if k == nil || k.apagado {
		return
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if sonda {
		k.sonda = false
		switch {
		case cancelada: // se vuelve a probar con la siguiente fila
		case ok:
			k.fijaEstado(circCerrado)
			k.segu, k.pos, k.n = 0, 0, 0
		default:
			k.abre()
		}
		return
	}
	if k.estado != circCerrado || cancelada { // fila lanzada antes de abrir: su resultado ya no cambia nada
		return
	}
	if ok {
		k.segu = 0
	} else {
		k.segu++
	}
	if k.ventana != nil {
		k.ventana[k.pos] = !ok
		k.pos = (k.pos + 1) % len(k.ventana)
		if k.n < len(k.ventana) {
			k.n++
		}
	}
	if k.segu >= k.maxSegu {
		k.abre()
		return
	}
	if k.ventana != nil && k.n == len(k.ventana) {
		f := 0
		for _, x := range k.ventana {
			if x {
				f++
			}
		}
		if float64(f)/float64(k.n) >= k.tasa {
			k.abre()
		}
	}
}
