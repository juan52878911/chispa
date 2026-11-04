package decide

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
)

// Con muchas opciones L3 solo ve las top_k de L2; las demás salen con p = 0. Sin guía o en otros tipos, ve todas.
func TestL3TopK(t *testing.T) {
	f := &llmFalso{contenido: `{"probs":{"o57":0.7,"o58":0.2,"o59":0.1}}`}
	l, _ := nuevoL3Test(t, f.arranca(t), func(c *ConfigL3) { c.TopK = 3 })
	var ids []string
	for i := range 60 {
		ids = append(ids, fmt.Sprintf("o%d", i))
	}
	p := pregL3(KindChoice, ids...)
	guia := map[string]float64{}
	for i, id := range ids {
		guia[id] = float64(i+1) / 1830 // suma 1; las mejores son o59, o58, o57
	}
	r, err := l.DecidirCon(context.Background(), p, Previas{"L2": {Answer: "o59", Probs: guia, Conf: guia["o59"]}})
	if err != nil || r.Answer != "o57" || len(r.Probs) != 60 || r.Probs["o0"] != 0 || absL3(r.Probs["o57"]-0.7) > 1e-9 {
		t.Fatalf("r=%+v err=%v", r, err)
	}
	var e entradaL3
	pe := f.ultima.Load().(peticionL3)
	if err := json.Unmarshal([]byte(pe.Messages[1].Content), &e); err != nil {
		t.Fatal(err)
	}
	if len(e.Opciones) != 3 || e.Opciones[0].ID != "o57" || e.Opciones[2].ID != "o59" || pe.MaxTokens != 16+24*3 {
		t.Fatalf("L3 debía ver o57, o58, o59 en orden: %+v (max_tokens %d)", e.Opciones, pe.MaxTokens)
	}
	// Sin capa anterior con probabilidades: todas las opciones (el LLM falso no las trae todas, así que no opina).
	if _, err := l.DecidirCon(context.Background(), p, Previas{}); err != nil {
		t.Fatal(err)
	}
	if pe := f.ultima.Load().(peticionL3); pe.MaxTokens != 16+24*60 {
		t.Fatalf("sin guía debía ver las 60: max_tokens %d", pe.MaxTokens)
	}
	// Defecto 8 y validación.
	if L3TopKPorDefecto != 8 {
		t.Fatal("el defecto de top_k cambió")
	}
	for _, k := range []int{-1, 1} {
		c := ConfigL3{URL: "http://127.0.0.1:1/v1", Modelo: "m", TopK: k}
		if err := c.validar(); err == nil {
			t.Errorf("top_k=%d debía ser inválido", k)
		}
	}
}
