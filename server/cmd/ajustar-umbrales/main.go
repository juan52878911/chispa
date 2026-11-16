// ajustar-umbrales ajusta los umbrales del score calibrado de una tarea y los escribe junto a su banco kNN (`<banco>.umbrales.json`).
//
//	ajustar-umbrales -banco DIR/banco-<tarea> -oro oro.jsonl -pred pred-soporte.jsonl [-excluir id1,id2,...]
//
//	-oro   el mismo jsonl de oro que lee `model/alt/generar_banco.py` ({"id","etiqueta","opciones",...}; solo cuentan las filas con split "train" o sin split);
//	-pred  las predicciones de L3 sobre ESAS filas de oro (la salida de `model/alt/llm_logprobs.py`: {"id","probs","valida"}), con la misma rúbrica con que
//	       L3 contestará en producción;
//	-excluir  ids del oro que sirvieron de ejemplo en la rúbrica: no entran al ajuste.
//
// Exige al menos 8 filas de oro por nivel y falla con un mensaje claro si no las hay. El ajuste no puede ir dentro de generar_banco.py porque necesita
// las predicciones del LLM sobre el soporte, que el generador (solo e5) no tiene.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/juan52878911/chispa/server/decide"
)

func leerJSONL[T any](ruta string) ([]T, error) {
	f, err := os.Open(ruta)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []T
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if len(strings.TrimSpace(sc.Text())) == 0 {
			continue
		}
		var v T
		if err := json.Unmarshal(sc.Bytes(), &v); err != nil {
			return nil, fmt.Errorf("%s línea %d: %w", ruta, len(out)+1, err)
		}
		out = append(out, v)
	}
	return out, sc.Err()
}

type filaOro struct {
	ID       string   `json:"id"`
	Etiqueta string   `json:"etiqueta"`
	Opciones []string `json:"opciones"`
	Split    string   `json:"split"`
}

type filaPred struct {
	ID     string             `json:"id"`
	Probs  map[string]float64 `json:"probs"`
	Valida *bool              `json:"valida"`
}

func main() {
	banco := flag.String("banco", "", "ruta base del banco (se escribe <banco>.umbrales.json)")
	oro := flag.String("oro", "", "jsonl de oro")
	pred := flag.String("pred", "", "jsonl con las predicciones de L3 sobre el oro")
	excluir := flag.String("excluir", "", "ids del oro excluidos del ajuste (ejemplos de la rúbrica), separados por comas")
	flag.Parse()
	if *banco == "" || *oro == "" || *pred == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*banco, *oro, *pred, *excluir); err != nil {
		fmt.Fprintln(os.Stderr, "ajustar-umbrales:", err)
		os.Exit(1)
	}
}

func run(banco, oro, pred, excluir string) error {
	filas, err := leerJSONL[filaOro](oro)
	if err != nil {
		return err
	}
	ps, err := leerJSONL[filaPred](pred)
	if err != nil {
		return err
	}
	porID := make(map[string]filaPred, len(ps))
	for _, p := range ps {
		porID[p.ID] = p
	}
	var niveles []string
	var sop []decide.PrediccionSoporte
	for _, r := range filas {
		if r.Split != "" && r.Split != "train" {
			continue
		}
		if niveles == nil {
			niveles = r.Opciones
		}
		p := porID[r.ID]
		sop = append(sop, decide.PrediccionSoporte{ID: r.ID, Etiqueta: r.Etiqueta, Probs: p.Probs, Valida: p.Probs != nil && (p.Valida == nil || *p.Valida)})
	}
	if len(sop) == 0 {
		return fmt.Errorf("%s: no hay filas de oro", oro)
	}
	ex := map[string]bool{}
	for _, id := range strings.Split(excluir, ",") {
		if id = strings.TrimSpace(id); id != "" {
			ex[id] = true
		}
	}
	u, err := decide.AjustarUmbrales(niveles, sop, ex)
	if err != nil {
		return err
	}
	if err := decide.EscribirUmbralesScore(banco, u); err != nil {
		return err
	}
	fmt.Printf("%s: umbrales %v\n", decide.RutaUmbrales(banco), u.Umbrales)
	return nil
}
