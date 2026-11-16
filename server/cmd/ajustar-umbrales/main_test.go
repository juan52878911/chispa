package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/juan52878911/chispa/server/decide"
)

func escribe(t *testing.T, ruta string, filas []map[string]any) {
	t.Helper()
	var b strings.Builder
	for _, f := range filas {
		j, _ := json.Marshal(f)
		b.Write(j)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(ruta, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestAjustarYCargar(t *testing.T) {
	d := t.TempDir()
	var oro, pred []map[string]any
	for n := 1; n <= 3; n++ {
		for i := 0; i < 8; i++ {
			id := fmt.Sprintf("n%d-%d", n, i)
			oro = append(oro, map[string]any{"id": id, "etiqueta": fmt.Sprint(n), "opciones": []string{"1", "2", "3"}, "split": "train"})
			p := map[string]float64{"1": 0.05, "2": 0.05, "3": 0.05}
			p[fmt.Sprint(n)] = 0.9
			pred = append(pred, map[string]any{"id": id, "probs": p, "valida": true})
		}
	}
	escribe(t, filepath.Join(d, "oro.jsonl"), oro)
	escribe(t, filepath.Join(d, "pred.jsonl"), pred)
	banco := filepath.Join(d, "banco-t")
	if err := run(banco, filepath.Join(d, "oro.jsonl"), filepath.Join(d, "pred.jsonl"), "n1-0, n2-0"); err != nil {
		t.Fatal(err)
	}
	u, err := decide.CargarUmbralesScore(banco)
	if err != nil || u == nil || len(u.Umbrales) != 2 || len(u.Excluidas) != 2 || u.Umbrales[0] >= u.Umbrales[1] {
		t.Fatalf("%+v %v", u, err)
	}
	// 7 por nivel: error claro y no se escribe nada
	escribe(t, filepath.Join(d, "oro7.jsonl"), oro[:len(oro)-1])
	banco7 := filepath.Join(d, "banco-7")
	if err := run(banco7, filepath.Join(d, "oro7.jsonl"), filepath.Join(d, "pred.jsonl"), ""); err == nil || !strings.Contains(err.Error(), "al menos 8") {
		t.Fatalf("debía exigir 8 por nivel: %v", err)
	}
	if _, err := os.Stat(decide.RutaUmbrales(banco7)); err == nil {
		t.Error("no debía escribir el fichero")
	}
}
