package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/juan52878911/chispa/server/decide"
)

const cfgForzar = `{"forzar_l3": true, "l3": {"url": "http://127.0.0.1:1/v1", "modelo": "m", "modo": "logprobs"}}`

func escribeCfg(t *testing.T, s string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "c.json")
	if err := os.WriteFile(p, []byte(s), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// forzar_l3 solo arranca (y se recarga) con -pruebas; sin él el error lo dice claro. Sin forzar_l3 el flag no hace falta.
func TestForzarL3ExigeFlagPruebas(t *testing.T) {
	defer func(v bool) { pruebasPermitidas = v }(pruebasPermitidas)
	p := escribeCfg(t, cfgForzar)
	pruebasPermitidas = false
	_, err := armarModelo(decide.NuevasMetricas(), p, "", "")
	if err == nil || !strings.Contains(err.Error(), "-pruebas") || !strings.Contains(err.Error(), "forzar_l3") {
		t.Fatalf("sin -pruebas debía negarse con un mensaje claro: %v", err)
	}
	pruebasPermitidas = true
	a, err := armarModelo(decide.NuevasMetricas(), p, "", "")
	if err != nil {
		t.Fatalf("con -pruebas debía arrancar: %v", err)
	}
	if a.Cerrar != nil {
		a.Cerrar()
	}
	pruebasPermitidas = false
	ok := escribeCfg(t, `{"l3": {"url": "http://127.0.0.1:1/v1", "modelo": "m"}}`)
	if a, err := armarModelo(decide.NuevasMetricas(), ok, "", ""); err != nil {
		t.Fatalf("sin forzar_l3 no hace falta -pruebas: %v", err)
	} else if a.Cerrar != nil {
		a.Cerrar()
	}
}

// Extremo a extremo con el binario: el proceso hijo sale con FATAL y menciona -pruebas.
func TestHijoForzarL3SinPruebasNoArranca(t *testing.T) {
	p := escribeCfg(t, cfgForzar)
	cmd, log := hijo(t, "-addr 127.0.0.1:0 -config "+p)
	if err := cmd.Wait(); err == nil || !strings.Contains(log.String(), "-pruebas") {
		t.Fatalf("debía salir con error citando -pruebas (err=%v):\n%s", err, log)
	}
}
