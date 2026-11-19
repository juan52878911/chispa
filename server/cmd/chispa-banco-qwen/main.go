// chispa-banco-qwen construye los bancos del kNN de L2 con Qwen3-Embedding en Go (sin torch): lee el soporte ya elegido (8 de oro por clase, como
// `muestrear(ys, 8, 0)` de R11; lo escribe model/alt/aws/qwen3emb_filas.py con numpy) y escribe banco-<tarea>.npy y banco-<tarea>.jsonl, el formato
// de siempre (decide.CargarBancoKNN): float32 (n × 1024) ya normalizado y una línea {"etiqueta", "es_oro": true} por fila, en el orden de entrada.
//
//	chispa-banco-qwen -onnx qwen3emb-fp32.onnx -tok qwen3emb.chispa-bpe -soporte soporte.jsonl -salida DIR [-ort-lib libonnxruntime.so] [-sesiones 1]
//
// soporte.jsonl: una fila por línea con tarea, etiqueta, pregunta y texto. Cada fila se codifica con la MISMA ruta que el servidor
// (decide.CodificadorQwen.CodificarConsulta: instrucción + pregunta, truncado a 320, pooling del último token, L2), así que el banco y las consultas
// comparten codificador por construcción. Hay que compilarlo con -tags ort; sin él falla con un mensaje claro.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"chispa/runtime/tok"
	"github.com/juan52878911/chispa/server/decide"
)

type fila struct {
	Tarea    string `json:"tarea"`
	Etiqueta string `json:"etiqueta"`
	Pregunta string `json:"pregunta"`
	Texto    string `json:"texto"`
	// Opcionales, para <banco>.ejemplos.jsonl (el formato de generar_banco.py): id, tipo, opciones (ids) y split (solo cuenta como oro "train" o ausente).
	ID       string   `json:"id,omitempty"`
	Tipo     string   `json:"tipo,omitempty"`
	Opciones []string `json:"opciones,omitempty"`
	Split    string   `json:"split,omitempty"`
}

// Semilla de la elección de ejemplos: la de generar_banco.py.
const semillaPorDefecto = 20261115

func main() {
	onnx := flag.String("onnx", "", "ONNX de Qwen3-Embedding (model/alt/exportar_qwen3emb_onnx.py)")
	rutaTok := flag.String("tok", "", ".chispa-bpe (runtime/tok/exportar_bpe.py)")
	soporte := flag.String("soporte", "", "soporte.jsonl (tarea, etiqueta, pregunta, texto)")
	salida := flag.String("salida", "", "directorio de salida")
	lib := flag.String("ort-lib", os.Getenv("CHISPA_ORT_LIB"), "libonnxruntime (o CHISPA_ORT_LIB)")
	ses := flag.Int("sesiones", 1, "sesiones ORT")
	k := flag.Int("k", -1, "ejemplos del few-shot de L3 en <banco>.ejemplos.jsonl (como generar_banco.py --k); 0 = sin fichero; -1 = 8 si las filas traen id, tipo y opciones y, si no, sin fichero (con aviso)")
	semilla := flag.Uint64("semilla", semillaPorDefecto, "semilla de la elección de ejemplos (la de generar_banco.py)")
	flag.Parse()
	if *onnx == "" || *rutaTok == "" || *soporte == "" || *salida == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := ejecutar(*onnx, *rutaTok, *soporte, *salida, *lib, *ses, *k, *semilla); err != nil {
		log.Fatal(err)
	}
}

func ejecutar(onnx, rutaTok, soporte, salida, lib string, ses, k int, semilla uint64) error {
	filas, err := leerFilas(soporte)
	if err != nil {
		return err
	}
	if len(filas) == 0 {
		return fmt.Errorf("%s: sin filas", soporte)
	}
	t, err := tok.AbrirBPE(rutaTok)
	if err != nil {
		return err
	}
	fijarRutaORT(lib)
	c, err := decide.NuevoCodificadorQwenORT(onnx, t, ses)
	if err != nil {
		return err
	}
	defer c.Close()
	if err := os.MkdirAll(salida, 0o755); err != nil {
		return err
	}
	porTarea := map[string][]fila{}
	for _, f := range filas {
		porTarea[f.Tarea] = append(porTarea[f.Tarea], f)
	}
	var nombres []string
	for n := range porTarea {
		nombres = append(nombres, n)
	}
	sort.Strings(nombres)
	resumen := map[string]any{}
	for _, n := range nombres {
		fs := porTarea[n]
		var vs [][]float32
		for _, f := range fs {
			v, err := c.CodificarConsulta(context.Background(), f.Pregunta, f.Texto)
			if err != nil {
				return fmt.Errorf("tarea %q: %w", n, err)
			}
			for _, x := range v {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					return fmt.Errorf("tarea %q: vector con NaN o Inf", n)
				}
			}
			vs = append(vs, v)
		}
		base := filepath.Join(salida, "banco-"+n)
		if err := escribirBanco(base, fs, vs); err != nil {
			return err
		}
		nej, err := escribirEjemplos(base, fs, k, semilla)
		if err != nil {
			return fmt.Errorf("tarea %q: %w", n, err)
		}
		resumen[n] = map[string]int{"filas": len(fs), "dim": len(vs[0]), "ejemplos_fs": nej}
		log.Printf("banco-%s: %d filas × %d", n, len(fs), len(vs[0]))
	}
	b, _ := json.MarshalIndent(resumen, "", " ")
	return os.WriteFile(filepath.Join(salida, "bancos.json"), append(b, '\n'), 0o644)
}

func leerFilas(ruta string) ([]fila, error) {
	f, err := os.Open(ruta)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []fila
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for n := 1; sc.Scan(); n++ {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var r fila
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("%s línea %d: %w", ruta, n, err)
		}
		if r.Tarea == "" || r.Etiqueta == "" {
			return nil, fmt.Errorf("%s línea %d: faltan tarea o etiqueta", ruta, n)
		}
		out = append(out, r)
	}
	return out, sc.Err()
}

// escribirBanco escribe <base>.npy (v1, '<f4', C-order) y <base>.jsonl.
func escribirBanco(base string, fs []fila, vs [][]float32) error {
	dim := len(vs[0])
	cab := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", len(vs), dim)
	for (10+len(cab)+1)%64 != 0 {
		cab += " "
	}
	cab += "\n"
	buf := append([]byte("\x93NUMPY\x01\x00"), 0, 0)
	binary.LittleEndian.PutUint16(buf[8:], uint16(len(cab)))
	buf = append(buf, cab...)
	for _, v := range vs {
		for _, x := range v {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(x))
		}
	}
	if err := os.WriteFile(base+".npy", buf, 0o644); err != nil {
		return err
	}
	var js []byte
	for _, f := range fs {
		l, _ := json.Marshal(map[string]any{"etiqueta": f.Etiqueta, "es_oro": true})
		js = append(append(js, l...), '\n')
	}
	return os.WriteFile(base+".jsonl", js, 0o644)
}

// escribirEjemplos escribe <base>.ejemplos.jsonl con la elección de generar_banco.py y devuelve cuántos ejemplos lleva (0 = no escribe fichero).
func escribirEjemplos(base string, fs []fila, k int, semilla uint64) (int, error) {
	if k == 0 {
		return 0, nil
	}
	completas := true
	for _, f := range fs {
		if f.ID == "" || f.Tipo == "" || len(f.Opciones) < 2 {
			completas = false
			break
		}
	}
	if !completas {
		if k > 0 {
			return 0, fmt.Errorf("-k %d exige que cada fila del soporte lleve id, tipo y opciones (ids de las opciones, al menos 2)", k)
		}
		log.Printf("AVISO: %s: las filas no traen id, tipo y opciones: no se escribe .ejemplos.jsonl y L3 contestará zero-shot", base)
		return 0, nil
	}
	if k < 0 {
		k = 8
	}
	for _, f := range fs {
		ok := false
		for _, o := range f.Opciones {
			ok = ok || o == f.Etiqueta
		}
		if !ok {
			return 0, fmt.Errorf("%s: la etiqueta %q no es una de sus opciones", f.ID, f.Etiqueta)
		}
	}
	var b strings.Builder
	idx := elegirEjemplos(fs, k, semilla)
	for _, i := range idx {
		b.WriteString(lineaEjemplo(fs[i]))
	}
	if len(idx) == 0 {
		return 0, nil
	}
	return len(idx), os.WriteFile(base+".ejemplos.jsonl", []byte(b.String()), 0o644)
}
