package tok

import (
	"strings"
	"testing"
)

// Senales cuenta los tokens sin truncar (con <s> y </s>) y dice si TokenizarMax recortaría.
func TestSenales(t *testing.T) {
	tk := abrir(t)
	texto := "quiero un reembolso de la factura"
	n, _, trunc := tk.Senales(Prefijo, texto, MaxTok)
	if want := len(tk.TokenizarMax(Prefijo, texto, MaxTok)); n != want || trunc {
		t.Fatalf("corto: %d tokens (quiero %d), truncado=%v", n, want, trunc)
	}
	largo := strings.Repeat(texto+" ", 100)
	n, _, trunc = tk.Senales(Prefijo, largo, MaxTok)
	if n <= MaxTok || !trunc {
		t.Fatalf("largo: %d tokens, truncado=%v", n, trunc)
	}
}
