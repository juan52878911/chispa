# runtime/tok — tokenizador de multilingual-e5-small en Go

`tok.Abrir("e5.chispa-tok").Tokenizar(texto) []int32` devuelve los ids de `AutoTokenizer(...)("query: " + texto,
truncation=True, max_length=128)` de `transformers` (128 es el defecto; `TokenizarMax(prefijo, texto, maxLen)` trunca a `maxLen` ≤ `TopeMaxLen` = 512, y el truncado real lo dan los metadatos del modelo: `chispa.consulta.max_len` del ONNX de L2 y `max_len` de TORR) (con `<s>` y `</s>`), que es lo que hace `model/l2/modelo.py` y lo que espera el ONNX de L2.

- **Motor**: el de `runtime/l1` (Viterbi sobre unigram, charsmap de sentencepiece, tabla hash FNV). `l1.DesdeTokenizador` lee un
  `.chispa-tok` (misma cabecera y secciones que el `.chispa-l1` con magia `CHTK` y sin pesos) y activa el modo `xlmr`: Precompiled →
  WhitespaceSplit → Metaspace(`always`, partiendo antes de cada `▁`) → Unigram palabra a palabra. No toca el modo de potion (su test dorado
  sigue igual).
- **Referencia**: el pipeline de `transformers` 5.x, que NO es el de `tokenizer.json` tal cual (el JSON normaliza `" {2,}"` y hace
  Metaspace sobre toda la frase; transformers usa WhitespaceSplit). Difieren en 38 de 2.315 frases de borde (un `▁` final suelto);
  `referencia.py --tokenizers` genera la referencia del otro pipeline. Si AWS instala otra versión de transformers y esto cambia, el test lo dice.
- **Exportar**: `python runtime/tok/exportar.py <dir-e5> /tmp/e5.chispa-tok` (3 s, determinista).
- **Pruebas** (`CHISPA_TOK`, por defecto `/tmp/e5.chispa-tok`; se saltan si falta): `go test ./...` compara con `testdata/ids_e5.jsonl`
  (2.274 frases del sondeo + 41 de borde, ids de transformers; 390 KB); `CHISPA_TOK_FUZZ=f.jsonl` con `referencia.py <dir> f.jsonl --fuzz N`;
  `go test -fuzz FuzzTokenizar -fuzztime 2m` comprueba forma, rango de ids y determinismo.

## Modo «passage: » (torre de esquemas de L2)
`TokenizarPasaje(texto)` = `TokenizarCon("passage: ", texto)`: mismo pipeline, prefijo de pasaje, truncado a 128 por defecto. `referencia.py <dir> <salida> --pasaje`
genera los ids de transformers con ese prefijo (más 9 textos con el formato `pregunta · opción · descripción`, U+00B7 y descripción vacía);
`testdata/ids_e5_passage.jsonl` (2.324 frases) lo comprueba `TestIgualdadTransformersPasaje`.

## BPE byte-level de Qwen2/Qwen3 (`bpe.go`; Qwen3-Embedding-0.6B)
`tok.AbrirBPE("qwen3emb.chispa-bpe").IDs(texto, 320)` da los ids de `AutoTokenizer.from_pretrained(dir, padding_side="left")(texto, truncation=True, max_length=320)` de transformers 5.x: NFC (`nfc.go`, con tablas
generadas por `gen_nfc.py`; igual que `unicodedata.normalize('NFC')` en 4 015 cadenas de prueba, Unicode 14.0), separación de los tokens añadidos, pretokenizador regex de Qwen2 hecho a mano (RE2 no tiene mirada adelante;
igual que `regex.findall` en 6 000 cadenas), fusiones por rango con cola de prioridad, y la plantilla de especiales que aplique transformers (p. ej. `<|endoftext|>` al final), con el truncado recortando el cuerpo.
- **Exportar**: `python runtime/tok/exportar_bpe.py DIR_MODELO SALIDA.chispa-bpe` (JSON determinista; lee el tokenizador EFECTIVO de transformers y falla si no es el esperado: regex, ignore_merges, añadidos con lstrip…).
- **Referencia**: `python runtime/tok/referencia_bpe.py DIR_MODELO testdata/qwen/fixture_paridad.jsonl SALIDA.jsonl [--fuzz N]` (ids de transformers; sin torch). El fixture son 200 textos inventados (`gen_fixture_qwen.py`).
- **Pruebas**: `go test ./...` compara con `testdata/qwen/ids_mini.jsonl` (200 del fixture + 1 500 de borde) usando un tokenizador de juguete con el MISMO pipeline que el de Qwen (`model/alt/qwen3emb_mini.py`); con el vocabulario
  real lo repite el ud (`server/decide/qwen_tokens_real_test.go`). Lo que NO cubre el juguete: las fusiones y los especiales del vocabulario real de Qwen.
