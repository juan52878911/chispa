package decide

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// truncado, tokens_originales y unk_pct: salen en la respuesta cuando una capa que corrió tokenizó el texto.
func TestSenalesTruncadoYUnk(t *testing.T) {
	casc, _ := cascadaTiny(t, calibTiny(), ConfigPorDefecto())
	ops := []Opcion{{ID: "facturacion", Description: "factura cobro pago dinero reembolso"}, {ID: "logistica", Description: "envio paquete retraso pedido"}}
	pregunta := func(estado string) Resultado {
		return casc.Decidir(context.Background(), Pregunta{ID: "q", Kind: KindChoice, Texto: "¿Qué área?", Estado: estado, Opciones: ops})
	}
	corto := pregunta("quiero un reembolso de la factura")
	if corto.Truncado || corto.TokensOriginales != 0 || corto.UnkPct == nil {
		t.Fatalf("texto corto: %+v", corto)
	}
	largo := pregunta(strings.Repeat("quiero un reembolso de la factura ", 400))
	if !largo.Truncado || largo.TokensOriginales <= 512 || largo.UnkPct == nil {
		t.Fatalf("texto largo: truncado=%v tokens=%d unk=%v", largo.Truncado, largo.TokensOriginales, largo.UnkPct)
	}
	b, _ := json.Marshal(largo)
	if !strings.Contains(string(b), `"truncado":true`) || !strings.Contains(string(b), `"tokens_originales":`) {
		t.Fatalf("JSON: %s", b)
	}
	raro := pregunta("ꙮꙮꙮ ꙮꙮꙮ ꙮꙮꙮ ꙮꙮꙮ")
	if raro.UnkPct == nil || *raro.UnkPct <= *corto.UnkPct {
		t.Fatalf("un texto de símbolos raros debía tener más <unk>: %v frente a %v", raro.UnkPct, corto.UnkPct)
	}
	// Sin capas que tokenicen (todo stub) no sale ninguna señal.
	c, err := NuevaCascada([]Capa{{"L0", 0.8}}, []Decisor{&falso{n: "L0", r: Respuesta{Answer: "a", Conf: 0.9}}}, NuevasMetricas())
	if err != nil {
		t.Fatal(err)
	}
	if r := c.Decidir(context.Background(), Pregunta{ID: "q", Kind: KindChoice, Estado: "x"}); r.UnkPct != nil || r.Truncado {
		t.Fatalf("sin tokenizar: %+v", r)
	}
}
