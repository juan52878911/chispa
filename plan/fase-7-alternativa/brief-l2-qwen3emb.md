# L2-qwen3emb: el codificador L2 del servidor Go pasa a ser intercambiable (e5-small o Qwen3-Embedding-0.6B)

Encargo: `plan/fase-7-alternativa/encargo-l2-qwen3emb.md`. El criterio de aceptación está congelado en `model/alt/crit-l2-qwen3emb.json` (sha256 58a62d618a4f917c1c4c232934a31394874a59f4cb5b589e945efd91511ade0b). No lo cambies tú.

## Contexto
- R11 midió en Python que Qwen3-Embedding-0.6B gana a e5 como codificador del kNN de L2, con un macro en dev de 0,4675 (32 vecinos) y 0,4683 (8 vecinos).
  - Código: `model/alt/alternativas_r11.py`, en `cmd_vec_qwen`, `EmbQwen` y `consulta_emb`.
  - Resultado: `model/alt/resultados-20261113-p0-p2/r11-alternativas-ba463e8.json`; el LEEME está en la misma carpeta, §18.
- Hoy el servidor solo sabe codificar con e5. El código está en:
  - `server/decide/e5_ort.go`, `e5.go`, `e5_stub.go` y `tokenizador_e5.go`;
  - `l2knn.go` (la interfaz `Codificador`, y `ConfigKNN` con `e5_onnx` y `e5_sesiones`) y `knn.go`;
  - `server/cmd/chispa-server/ort_si.go` y `ort_no.go`;
  - el tokenizador unigram de e5 en `runtime/tok` (motor `runtime/l1`).
- La ronda T5b hizo lo mismo para e5, y es el patrón a seguir:
  - criterio `model/alt/crit-r10-t5b.json` y exportación `model/alt/exportar_e5_onnx.py`;
  - ud `model/alt/aws/r10_t5e2e.sh`, `r10_t5e2e_fases.sh` y `empaquetar_r10_t5e2e.sh`;
  - cliente HTTP corregido `model/alt/aws/r10_t5_cliente.py`.

## Qué construir
1. **Config.** En `knn`, añade `codificador: "e5-small" | "qwen3-emb"`.
   - El valor por defecto es e5-small, con el comportamiento de hoy byte a byte. `e5_onnx` y `e5_sesiones` siguen valiendo.
   - Añade la ruta del ONNX y del tokenizador de Qwen con nombres claros, y `vecinos` (32 u 8).
   - Una config vieja tiene que arrancar igual que antes; un test lo demuestra.
2. **Codificador Qwen en Go**, con lo que midió R11 y nada más:
   - revisión 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3;
   - consulta «Instruct: Classify the text to answer this question: <pregunta>\nQuery:<texto>» en soporte y test, y las descripciones de etiqueta sin instrucción;
   - truncado a 320 tokens, pooling del último token (con relleno a la izquierda en lote) y normalizado L2.
   - **Ojo:** el camino de e5 normaliza mayúsculas del mensaje antes de codificar. R11 no lo hacía con Qwen; reproduce R11, no e5.
3. **Tokenizador BPE byte-level de Qwen en Go.** Puede ir en `runtime/tok`, si encaja, o en un paquete hermano.
   - Incluye NFC, el pretokenizador regex de Qwen2 (Go no tiene lookahead: impleméntalo a mano), las fusiones por rango y los tokens especiales tal como los pone transformers 5.22.2 con `AutoTokenizer(padding_side="left")` (por ejemplo, si añade `<|endoftext|>` al final).
   - Añade un exportador del tokenizador a un fichero propio (determinista) y un `referencia.py` que genere los ids de transformers.
4. **Exportación a ONNX:** `model/alt/exportar_qwen3emb_onnx.py`, con revisión, opset y versiones fijadas, fp32 y además int8 dinámico de ORT. Los metadatos llevan el max_len y el pooling.
5. **Bancos:** el soporte (8 de oro por clase, como `muestrear(ys,8,0)` en R11) se codifica con Qwen.
   - Decide si los construye una herramienta en Go (preferible: producción sin torch) o Python, y justifícalo en el informe.
   - El formato `<banco>.npy` y `.jsonl` no cambia.
6. **Fixture de paridad:** 200 textos inventados por ti (es y en, con acentos, ñ, emojis, saltos de línea, espacios raros y unos 20 que pasen de 320 tokens), en `testdata`.
   - Nada de PQRS, nada de selladas ni datos de clientes.
   - Los ids y vectores de referencia los genera transformers 5.22.2: offline si puedes instalar `transformers==5.22.2` sin torch (el tokenizador no necesita torch); si no, en el ud.
7. **ud de medida para c7i.xlarge** (CPU, 4 hilos), preparado y **NO lanzado**, como el de T5b:
   - plantilla con `__CHISPA_S3__`, `shutdown -h +N`, `listo`/`verificado`, imports comprobados antes de bajar nada y sha del criterio comprobado;
   - empaquetador que escribe `model/alt/aws/lanzados/ud-l2-qwen3emb-SHA.sh` y los tgz, sin subir nada;
   - lo que hace en la máquina:
     - baja el modelo en la revisión fijada, exporta fp32 e int8 y calcula la referencia de transformers sobre el fixture;
     - paridad Go de tokens y coseno, con un JSON por condición;
     - construye los bancos de las 6 tareas dev (`bateria-v2/dev`, el empaquetador falla si trae algo que no sea esos 6 jsonl, como `empaquetar_r11_alternativas.sh`);
     - levanta el servidor con L3 apagado y sin escalado (que conteste solo L2) y corre el e2e con `r10_t5_cliente.py` en v32 y v8 × fp32 e int8;
     - mide la latencia p50/p95 por consulta, el RSS del servidor y el tamaño del binario, del ONNX y de la imagen (o una estimación clara si no hay docker);
     - escribe un `cifras.json` con el veredicto de cada condición del criterio.
   - El ud cabe en 16 KB; lo largo va en el código.
8. **Documentación:** `server/PRODUCTO.md` y `server/deploy/config.ejemplo.json` explican cómo elegir el codificador y la vuelta atrás. Las cifras quedan como «pendiente de medir» hasta la corrida.
9. **Tests en Go:** config vieja = e5 sin cambios, selección del codificador, tokenizador contra un fixture, pooling del último token con un ONNX sintético diminuto (como `server/decide/testdata/l2_sintetico.py`, sin torch) y errores claros si falta un fichero.

## Hipótesis y cómo refutarla rápido
- El servidor con Qwen da el mismo kNN que el Python de R11.
- Lo que puede romperlo: tokens distintos (regex, NFC, especiales), el pooling equivocado, la normalización de mayúsculas o el truncado.
- Refútalo primero offline: compara ids con transformers y el pooling con un ONNX sintético. Un fallo de tokens se ve sin el modelo.

## Terminado cuando
- `cd server && go test ./...` (con y sin `-tags ort`, si el nodo tiene ORT) pasa, y `make test` y `make smoke` en la raíz pasan.
- Si pudiste instalar transformers 5.22.2: tokens 200/200 offline.
- `bash -n` del ud pasa, el empaquetador genera el ud con una prueba en seco y no quedan placeholders.
- **Parada:** si algo falla dos veces con el mismo enfoque, manda un `blocker` y para.

## Zona
- Puedes tocar:
  - `server/decide/`, `server/cmd/chispa-server/`, `server/PRODUCTO.md`, `server/deploy/config.ejemplo.json` y `server/deploy/Dockerfile.producto` (si hace falta);
  - `runtime/tok/` (o un paquete nuevo en `runtime/`);
  - `model/alt/` (scripts nuevos, `aws/` y tests).
- **No toques:** `contracts/` ni `model/alt/crit-*.json`.
- No cargues en contexto `data/`, `models/` ni `data/ledger.jsonl`.
- Reglas del CLAUDE.md del repo. Commitea en tu rama; sin push ni merge.

## Entorno
- Línea base: la rama `l2-qwen3emb` en su HEAD (48e9a2c + R11).
- Nada de torch en el nodo. Las descargas grandes de modelos (más de 50 MB) no se hacen aquí: van al ud.

## Tope
- ≤ 240 min de reloj. Cómputo de pago: 0 (no lances nada en AWS).
- Sin subagentes.

## Informe
Termina con un informe de ≤ 15 líneas:
- qué cambiaste, con la rama y el sha;
- qué verificaste ejecutando, con números (tests, tokens offline);
- la ruta y el sha256 del ud y de los tgz;
- qué quedó sin verificar;
- qué recomiendas a continuación.
