package main

import (
	"os"
	"path/filepath"
	"testing"

	"chispa/format/chispa3"
	"github.com/juan52878911/chispa/server/decide"
)

// Un solo origen de τ: el de CALIB manda con o sin -config. Antes, con -config (aunque fuera «{}») mandaban los τ fijos de
// ConfigPorDefecto (0,80/0,80/0,70/0,60) sin avisar. La config solo puede subir el τ, y solo si su JSON trae «capas».
func TestTauUnicoConYSinConfig(t *testing.T) {
	const modelo = "../../decide/testdata/dorados/l0l1.chispa"
	f, err := chispa3.Abrir(modelo)
	if err != nil {
		t.Fatal(err)
	}
	cal, err := f.Calib()
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	taus := func(cfg string) map[string]float64 {
		t.Helper()
		p := ""
		if cfg != "-" {
			p = filepath.Join(dir, "c.json")
			if err := os.WriteFile(p, []byte(cfg), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		a, err := armarModelo(decide.NuevasMetricas(), p, modelo, "")
		if err != nil {
			t.Fatal(err)
		}
		if a.Cerrar != nil {
			defer a.Cerrar()
		}
		return a.Cascada.Taus()
	}
	sin, vacia := taus("-"), taus("{}")
	if sin["L1"] != cal.TauEscalate {
		t.Fatalf("sin config, τ de L1 = %v; CALIB dice %v", sin["L1"], cal.TauEscalate)
	}
	for _, c := range []string{"L0", "L1", "L2", "L3"} {
		if sin[c] != vacia[c] {
			t.Errorf("τ de %s: %v sin config, %v con «{}»", c, sin[c], vacia[c])
		}
	}
	// Otra clave cualquiera tampoco cambia el τ.
	if otra := taus(`{"cache_esquemas":8}`); otra["L1"] != cal.TauEscalate {
		t.Errorf("con una config sin capas, τ de L1 = %v; CALIB dice %v", otra["L1"], cal.TauEscalate)
	}
	// Con «capas», un τ por debajo de CALIB se ignora y uno por encima lo sube.
	if baja := taus(`{"capas":[{"nombre":"L0","tau":0.8},{"nombre":"L1","tau":0},{"nombre":"L2","tau":0.7},{"nombre":"L3","tau":0.6}]}`); baja["L1"] != cal.TauEscalate {
		t.Errorf("capas con τ de L1 = 0: queda %v; debía seguir el de CALIB (%v)", baja["L1"], cal.TauEscalate)
	}
	if sube := taus(`{"capas":[{"nombre":"L0","tau":0.8},{"nombre":"L1","tau":1},{"nombre":"L2","tau":0.7},{"nombre":"L3","tau":0.6}]}`); sube["L1"] != 1 {
		t.Errorf("capas con τ de L1 = 1: queda %v; debía subirlo", sube["L1"])
	}
}
