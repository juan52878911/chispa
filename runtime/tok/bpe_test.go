package tok

import (
	"bufio"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func leerJSONL(t *testing.T, ruta string, f func(linea []byte)) {
	t.Helper()
	fh, err := os.Open(ruta)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<22), 1<<22)
	for sc.Scan() {
		f(sc.Bytes())
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
}

// El pretokenizador a mano (sin mirada adelante) parte igual que `regex.findall` de Python con el patrón de Qwen2.
func TestPretokenizarQwen2ContraRegex(t *testing.T) {
	n := 0
	leerJSONL(t, "testdata/qwen/pretok_casos.jsonl", func(l []byte) {
		var c struct {
			S     string
			Trozo []string
		}
		var par []json.RawMessage
		if err := json.Unmarshal(l, &par); err != nil || len(par) != 2 {
			t.Fatal(err)
		}
		json.Unmarshal(par[0], &c.S)
		json.Unmarshal(par[1], &c.Trozo)
		n++
		got := pretokenizarQwen2(c.S)
		if len(c.Trozo) == 0 {
			c.Trozo = nil
		}
		if !reflect.DeepEqual(got, c.Trozo) {
			t.Fatalf("caso %d: %+q → %+q, quiero %+q", n, c.S, got, c.Trozo)
		}
		if strings.Join(got, "") != c.S {
			t.Fatalf("las piezas no cubren %+q", c.S)
		}
	})
	if n < 6000 {
		t.Fatalf("solo %d casos", n)
	}
}

// Ids idénticos a transformers (AutoTokenizer, padding_side="left", truncation, max_length 320) con el tokenizador de juguete: el fixture de 200
// textos (20 de ellos truncados) y 1 500 cadenas de borde. Es el mismo pipeline que el de Qwen3-Embedding; el vocabulario es otro.
func TestBPEIgualTransformersMini(t *testing.T) {
	b, err := AbrirBPE("testdata/qwen/mini.chispa-bpe")
	if err != nil {
		t.Fatal(err)
	}
	n, truncados := 0, 0
	leerJSONL(t, "testdata/qwen/ids_mini.jsonl", func(l []byte) {
		var r struct {
			ID      string  `json:"id"`
			Entrada string  `json:"entrada"`
			IDs     []int32 `json:"ids"`
			Largo   int     `json:"largo_sin_truncar"`
		}
		if err := json.Unmarshal(l, &r); err != nil {
			t.Fatal(err)
		}
		n++
		got := b.IDs(r.Entrada, 320)
		if !reflect.DeepEqual(got, r.IDs) {
			t.Fatalf("%s: %+q\n go  %v\n ref %v", r.ID, r.Entrada, got, r.IDs)
		}
		if full := b.IDs(r.Entrada, 0); len(full) != r.Largo {
			t.Fatalf("%s: sin truncar %d ids, quiero %d", r.ID, len(full), r.Largo)
		}
		if r.Largo > 320 {
			truncados++
		}
	})
	if n != 1700 || truncados < 20 {
		t.Fatalf("%d textos, %d truncados", n, truncados)
	}
	t.Logf("%d textos idénticos a transformers (%d truncados a 320)", n, truncados)
}

func TestBPEPlantillaYTruncadoLimite(t *testing.T) {
	b, err := AbrirBPE("testdata/qwen/mini.chispa-bpe")
	if err != nil {
		t.Fatal(err)
	}
	ids := b.IDs("hola mundo", 0)
	if ids[len(ids)-1] != b.PadID() {
		t.Fatalf("la plantilla debe terminar en <|endoftext|> (%d): %v", b.PadID(), ids)
	}
	for _, m := range []int{1, 2, 3, 5} {
		got := b.IDs("hola mundo cruel y largo", m)
		if len(got) != m || got[m-1] != b.PadID() {
			t.Fatalf("max %d: %v", m, got)
		}
	}
	if got := b.IDs("", 320); len(got) != 1 {
		t.Fatalf("texto vacío: %v", got)
	}
	// especial escrito en el texto → un solo id
	if got := b.IDs("a<|im_end|>b", 0); len(got) != 4 {
		t.Fatalf("especial literal: %v", got)
	}
	// UTF-8 inválido no entra en pánico
	_ = b.IDs("a\xffb\xc3", 320)
}

func TestBPEErroresClaros(t *testing.T) {
	if _, err := AbrirBPE("testdata/qwen/no-existe.chispa-bpe"); err == nil || !strings.Contains(err.Error(), "no-existe") {
		t.Fatalf("falta el fichero: %v", err)
	}
	if _, err := BPEDesde([]byte("{")); err == nil {
		t.Fatal("JSON roto")
	}
	if _, err := BPEDesde([]byte(`{"formato":"otro"}`)); err == nil || !strings.Contains(err.Error(), "formato") {
		t.Fatalf("formato: %v", err)
	}
	if _, err := BPEDesde([]byte(`{"formato":"chispa-bpe-1","normalizador":"NFC","regex":"qwen2","vocab":["61"],"fusiones":[]}`)); err == nil || !strings.Contains(err.Error(), "byte") {
		t.Fatalf("faltan bytes: %v", err)
	}
}

func BenchmarkBPELargo(b *testing.B) {
	t, err := AbrirBPE("testdata/qwen/mini.chispa-bpe")
	if err != nil {
		b.Fatal(err)
	}
	s := strings.Repeat("Hola, ¿qué tal? Esto es una prueba larga con ñ y 123. ", 200)
	for i := 0; i < b.N; i++ {
		t.IDs(s, 320)
	}
}
