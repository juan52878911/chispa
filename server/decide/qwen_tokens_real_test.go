package decide

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"chispa/runtime/tok"
)

// Paridad con el modelo REAL (la corrida del ud; se salta sin las variables):
//
//	CHISPA_QWEN_TOK=qwen3emb.chispa-bpe  CHISPA_QWEN_REF=DIR (ids.jsonl y vectores.npy de model/alt/qwen3emb_referencia.py)  [CHISPA_QWEN_SALIDA=DIR]
//	go test ./decide -run QwenTokensReal -v        (y, con -tags ort, QwenCosenoReal con CHISPA_QWEN_ONNX)
//
// Criterio congelado (model/alt/crit-l2-qwen3emb.json): tokens 200/200 idénticos a transformers y coseno ≥ 0,999 en cada texto.
type resumenTokensQwen struct {
	N         int      `json:"n"`
	Iguales   int      `json:"iguales"`
	Truncan   int      `json:"truncados_a_320"`
	Fallos    []string `json:"primeros_fallos,omitempty"`
	Veredicto bool     `json:"pasa"`
}

type refIDsQwen struct {
	ID      string  `json:"id"`
	Entrada string  `json:"entrada"`
	IDs     []int32 `json:"ids"`
	Largo   int     `json:"largo_sin_truncar"`
}

func leerRefIDsQwen(t *testing.T, ruta string) []refIDsQwen {
	t.Helper()
	f, err := os.Open(ruta)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []refIDsQwen
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<22), 1<<22)
	for sc.Scan() {
		var r refIDsQwen
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		out = append(out, r)
	}
	return out
}

func compararTokensQwen(t *testing.T, b *tok.BPE, refs []refIDsQwen) resumenTokensQwen {
	r := resumenTokensQwen{N: len(refs)}
	for _, x := range refs {
		got := b.IDs(x.Entrada, QwenMaxTok)
		if reflect.DeepEqual(got, x.IDs) {
			r.Iguales++
		} else if len(r.Fallos) < 5 {
			r.Fallos = append(r.Fallos, x.ID)
			t.Logf("%s: %+q\n go  %v\n ref %v", x.ID, x.Entrada, got, x.IDs)
		}
		if x.Largo > QwenMaxTok {
			r.Truncan++
		}
	}
	r.Veredicto = r.N == 200 && r.Iguales == r.N
	return r
}

func escribirJSONQwen(t *testing.T, nombre string, v any) {
	dir := os.Getenv("CHISPA_QWEN_SALIDA")
	if dir == "" {
		return
	}
	b, _ := json.MarshalIndent(v, "", " ")
	if err := os.WriteFile(filepath.Join(dir, nombre), append(b, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestQwenTokensReal(t *testing.T) {
	rutaTok, ref := os.Getenv("CHISPA_QWEN_TOK"), os.Getenv("CHISPA_QWEN_REF")
	if rutaTok == "" || ref == "" {
		t.Skip("falta CHISPA_QWEN_TOK o CHISPA_QWEN_REF")
	}
	b, err := tok.AbrirBPE(rutaTok)
	if err != nil {
		t.Fatal(err)
	}
	r := compararTokensQwen(t, b, leerRefIDsQwen(t, filepath.Join(ref, "ids.jsonl")))
	escribirJSONQwen(t, "paridad-tokens.json", r)
	t.Logf("tokens: %d/%d idénticos a transformers (%d textos truncados a 320)", r.Iguales, r.N, r.Truncan)
	if !r.Veredicto {
		t.Fatalf("tokens: %d/%d", r.Iguales, r.N)
	}
}
