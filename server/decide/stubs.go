package decide

import "context"

// Stub es una capa sin modelo: siempre «sin opinión» (confianza 0). Se
// sustituye por el decisor real cuando llegue el modelo.
type Stub struct{ N string }

func (s Stub) Nombre() string { return s.N }

func (Stub) Decidir(context.Context, Pregunta) (Respuesta, error) { return Respuesta{}, nil }

func NuevoL1() Decisor { return Stub{"L1"} }

// SinL1Ausente quita la capa L1 de la cascada si su decisor es el stub (el .chispa no trae sección L1): la cascada se la salta en
// vez de llamar a una capa que nunca opina. Devuelve las capas y decisores que quedan y si quitó L1.
func SinL1Ausente(capas []Capa, ds []Decisor) ([]Capa, []Decisor, bool) {
	for i, d := range ds {
		if s, ok := d.(Stub); ok && s.N == "L1" && i < len(capas) && capas[i].Nombre == "L1" {
			return append(append([]Capa(nil), capas[:i]...), capas[i+1:]...), append(append([]Decisor(nil), ds[:i]...), ds[i+1:]...), true
		}
	}
	return capas, ds, false
}
func NuevoL2() Decisor { return Stub{"L2"} }
func NuevoL3() Decisor { return Stub{"L3"} }
