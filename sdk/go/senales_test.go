package chispa

import (
	"encoding/json"
	"testing"
)

// Los campos nuevos de la respuesta son opcionales: un servidor antiguo da nil/cero y uno nuevo los rellena.
func TestResultadoCamposOpcionales(t *testing.T) {
	var viejo Resultado
	if err := json.Unmarshal([]byte(`{"id":"q","kind":"choice","answer":"a","probs":{"a":1},"layer":"L1","escalate":false,"ms":1}`), &viejo); err != nil {
		t.Fatal(err)
	}
	if viejo.Calibrada != nil || viejo.Truncado || viejo.TokensOriginales != 0 || viejo.UnkPct != nil {
		t.Fatalf("servidor antiguo: %+v", viejo)
	}
	var nuevo Resultado
	j := `{"id":"q","kind":"choice","answer":"a","probs":{"a":1},"layer":"L1","escalate":false,"ms":1,"calibrada":false,"truncado":true,"tokens_originales":900,"unk_pct":0.25}`
	if err := json.Unmarshal([]byte(j), &nuevo); err != nil {
		t.Fatal(err)
	}
	if nuevo.Calibrada == nil || *nuevo.Calibrada || !nuevo.Truncado || nuevo.TokensOriginales != 900 || nuevo.UnkPct == nil || *nuevo.UnkPct != 0.25 {
		t.Fatalf("servidor nuevo: %+v", nuevo)
	}
}
