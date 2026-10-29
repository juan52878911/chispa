# Servidor de Chispa (T4.1)

`POST /v1/decide` decide preguntas con una cascada de capas: L0 (Chispa lineal real, `pkg/chispa`
de kindling v0.17.0), L1 real (embedding estático de `runtime/l1`), L2 (ONNX, solo con la etiqueta `ort`) y L3 (un LLM
opcional, ver «L3»). Las capas sin sección en el modelo, y L3 sin config, son stubs «sin opinión» (confianza 0).
Tipos soportados: `choice`, `score`, `noul` y `multi` (ver la tabla de abajo). Esquema de preguntas: `model/esquema.md`;
contrato completo en [`docs/03-api.md`](../docs/03-api.md) y [`api/openapi.yaml`](api/openapi.yaml).

```
make -C server test     # go vet + go test -race
make -C server bench    # benchmarks (1 y 4 núcleos)
make -C server serve    # arranca con testdata/config.json (3 modelos de prueba)
make -C server carga    # generador de carga contra :8090
```

Módulos locales por `replace` en `go.mod`: `../format/chispa3` y `../runtime/l1` (sin red).
Sin red: `GOFLAGS=-mod=mod GOPROXY=off` (ya puesto en el Makefile). `go.sum` es el de `arena/`.

## Contrato

### Entrada
```json
{
  "state": "texto que se clasifica",
  "questions": [
    {"id": "u1", "question": "¿Qué tan urgente es?", "kind": "choice",
     "options": [{"id": "alta", "description": "Requiere atención inmediata"}, {"id": "baja"}],
     "nonce": "abc-1"}
  ]
}
```
- `state` (string, ≤ 256 KiB): el texto. Es el mismo para todas las preguntas de la petición.
- `questions` (1..256). Cada una: `id` (obligatorio, único en la petición), `question` (obligatorio),
  `kind` ∈ `choice|score|noul|multi`, `options` y `nonce` (opcional).
  - `choice`: ≥ 2 opciones (≤ 256; clinc150 trae 150) con `id` único y `description` opcional.
  - `score`: ≥ 2 opciones ordinales, ids `"1"…"K"`.
  - `noul`: opciones opcionales; si van, solo `"false"` y `"true"`. Por defecto las dos.
  - `multi` (multietiqueta): ≥ 2 opciones (≤ 256) con `id` único; cada una es una etiqueta que puede estar activa o no.
- Cuerpo ≤ 1 MiB; campos desconocidos → 400.

### Salida (200)
```json
{"results": [
  {"id": "u1", "kind": "choice", "answer": "alta", "probs": {"alta": 0.9, "baja": 0.1},
   "layer": "L0", "escalate": false, "ms": 0.012, "nonce": "abc-1"}
]}
```
- `answer`: id de la opción elegida (en `noul`, `"true"`/`"false"`; en `score`, `"1"…"K"`; en `multi`, la activa de mayor probabilidad, o `""` si no hay ninguna).
- `answers`: **solo en `multi`**: los ids de las opciones activas (p ≥ θ de la opción), en el orden de la pregunta; `[]` si ninguna
  (respuesta válida, con `escalate: false`).
- `probs`: id de opción → probabilidad calibrada (la de la capa que respondió). Suman 1 salvo en `multi`, donde cada opción lleva la suya en [0, 1].
- `layer`: la capa que contestó; si `escalate` es true, `"escalar"` (ninguna capa aceptó; `answer` y `probs` son la mejor respuesta vista).
- `escalate`: ninguna capa llegó a su τ. `answer`/`probs` son entonces la mejor pista vista
  (la capa con más confianza) y quedan vacíos (`""`, `{}`) si ninguna opinó.
- `ms`: tiempo de la cascada para esa pregunta. `nonce` se devuelve tal cual.
- Los resultados van en el orden de `questions`.

### Errores
`400 {"error": {"code": "invalid_kind", "message": "...", "question_id": "..."}}`: errores tipados, con la ruta del campo en `message` (`questions[2].kind: "raro" no válido (choice, score, noul o multi)`). Lista de códigos en [`docs/03-api.md`](../docs/03-api.md) y [`api/openapi.yaml`](api/openapi.yaml).
También 405 (método ≠ POST) y 413 (cuerpo > 1 MiB). No hay errores por pregunta: la validación es
todo o nada, y un fallo interno de una capa solo la salta (se cuenta en `chispa_capa_errores_total`).

### Nonce
Clave de idempotencia: la misma petición con el mismo `nonce` devuelve el resultado guardado sin
volver a decidir (`ms` incluido). La clave es `nonce + estado + esquema + id`: el mismo nonce con
otro estado o pregunta se recalcula. Sin `nonce` no se cachea. LRU de `cache_respuestas` entradas.

### Cascada
L0 → L1 → L2 → L3: se pasa a la siguiente mientras la confianza top-1 (en `multi`, la de la opción más dudosa) sea < τ de la capa (una capa
sin opinión o con error vale 0). Con confianza ≥ τ responde esa capa. Si la última no llega,
`escalate: true`. Capas y τ se leen del JSON de configuración (`-config`):

```json
{"capas": [{"nombre":"L0","tau":0.8},{"nombre":"L1","tau":0.8},{"nombre":"L2","tau":0.7},{"nombre":"L3","tau":0.6}],
 "modelos": ["testdata/intencion.chispa"], "cache_esquemas": 1024, "cache_respuestas": 4096}
```
Interfaz por capa: `type Decisor interface { Decidir(ctx, Pregunta) (Respuesta, error); Nombre() string }`
(`decide/types.go`). Para sustituir un stub basta pasar otro `Decisor` con el mismo nombre en `main.go`.

### Tipos soportados y qué capa opina en cada uno
| `kind` | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| `choice` | sí, si hay un modelo con las mismas etiquetas | softmax(coseno / T) | MaxSim contra la torre de cada opción, softmax | un LLM reparte la probabilidad entre opciones |
| `noul` | ídem | σ(a·coseno + b) sobre la opción `true` | σ(a·MaxSim + b) | ídem |
| `score` | ídem | softmax sobre los niveles; `answer` = nivel mediano | CORAL (solo con ids `"1"…"K"`) | ídem |
| `multi` | **no opina** (el L0 es multiclase) | σ(a·coseno + b) por opción, θ por opción | σ(a·MaxSim + b) por opción, una torre por opción | una p independiente por opción (se activan las de p ≥ 0,5) |

En `multi`, una capa solo acepta si **todas** sus decisiones por opción superan τ (confianza = la de la opción más dudosa: el mínimo de
p si está activa o 1 − p si no); si no, pasa a la siguiente y al final escala (`escalate: true`, `probs: {}`). Calibración en
`CALIB.tipos.multi` y `multi#<id>`; detalle en [`docs/03-api.md`](../docs/03-api.md).

### Esquemas nuevos: τ y λ
- **τ de esquema nuevo** (`tau_l2_esquema_nuevo`): τ de L2 solo para preguntas cuyas torres **no estaban** en `ESQUEMAS` (salieron al vuelo desde la
  sección `TORR`). Prioridad: la config > `tau_l2_esquema_nuevo` de `CALIB` (cuenta aunque haya `-config`) > 0,846 por defecto (precisión ≈ 0,95 con ≈ 97 % de escalado en tareas
  no vistas). `"tau_l2_esquema_nuevo_apagado": true` vuelve al τ normal de L2 (junto a un valor, error). Métricas: `chispa_esquema_nuevo_preguntas_total`,
  `chispa_esquema_nuevo_decididas_total`. Las torres al vuelo se cachean (`cache_torres`, por defecto 16384) y una petición calcula como mucho
  `max_torres_nuevas` (por defecto 256) sin cachear (el `.chispa` de `-modelo` puede pesar hasta `max_fichero` bytes, 1 GiB por defecto); si el tope corta, cachea las que caben y escala con `sin_presupuesto_torres` (`chispa_torres_*`).
- **λ** (`l1_shrinkage_lambda`, `l2_shrinkage_lambda`): encogimiento del soporte few-shot (sección `SOPL1`): w = k/(k+λ) mezcla la media de los k ejemplos con el
  prototipo de la descripción. L1: por defecto 4; `"l1_shrinkage_apagado": true` lo apaga. L2: sin valor, no hay mezcla (λ = ∞). Sin `SOPL1` en el `.chispa` no tiene efecto.
  Un λ negativo es error de config. Con `-config` hay que repetir los de `CALIB` que se quieran conservar.

### L3 (LLM opcional)
Con `"l3": {...}` en la config **y** una capa `"L3"` en `capas`, las preguntas que L0–L2 no resolvieron se mandan a un endpoint compatible con OpenAI
(`POST {url}/chat/completions`, p. ej. Ollama en `http://127.0.0.1:11434/v1`). Pide un JSON estricto con una probabilidad por opción, lo valida y lo normaliza (en `multi`
no normaliza y deben venir todas); si la respuesta no sirve, L3 no opina y la pregunta escala.
Campos de `l3`: `url`, `modelo`, `api_key_env` (nombre de la variable de entorno con la clave: nunca la clave en la config; si la variable está vacía, falla al armar),
`timeout` (`"10s"` o segundos; por defecto 10 s), `min_por_peticion` y `max_por_peticion` (cupo de filas con L3 por petición HTTP: el tamaño del lote acotado a [mín 8, máx 256]; ver «Endurecimiento») y `tau` (si falta, el de la capa `L3`).
**Los textos del cliente salen del servidor hacia ese endpoint**; L3 no los registra. Métrica: `chispa_l3_llamadas_total`.

#### L3 por logprobs (`"modo": "logprobs"`)
Alternativa al JSON pensada para vLLM con Qwen3-4B-Instruct-2507 (0,729 agrupada frente a 0,650 con JSON; 33 ms p50 con un hilo). Es el espejo en Go de
`model/alt/llm_logprobs.py` (código en `decide/l3_logprobs.go`; test de paridad del prompt con el Python en `decide/testdata/l3_logprobs_paridad.json`).
Sin `modo` (o `"json"`) todo sigue como arriba. Se activa en la config:
```json
{"l3": {"url": "http://vllm:8000/v1", "modo": "logprobs", "modelo": "qwen3-4b", "max_concurrente": 4, "timeout_fila": "2s", "max_extra": 4, "sin_pensar": false}}
```
Cada opción recibe una etiqueta de un carácter (A–Z, a–z, 0–9; en `score` con niveles de una cifra, el propio nivel) y se pide `max_tokens=1`, `logprobs=true`,
`top_logprobs=20`. Las variantes `"A"` y `" A"` se suman. Las etiquetas que no salen en el top 20 se puntúan con `prompt_logprobs` (`add_generation_prompt=false`,
`continue_final_message=true`) para las primeras `max_extra`; el resto recibe un piso (mínimo conocido − 1). probs = softmax sobre las opciones. Más de 62 opciones,
o ninguna etiqueta puntuada, = respuesta inválida (L3 no opina y escala). `multi` también va por logprobs (ver abajo). El servidor vLLM debe tener `--max-logprobs` ≥ 20.
- `max_concurrente`: filas puntuándose a la vez contra el LLM (semáforo; por defecto 4).
- `timeout_fila`: tope de cada fila incluido el relleno (por defecto, el `timeout`). Si vence con algo puntuado se devuelven las probs parciales con piso; si no, error `ErrSinL3` y la pregunta escala.
- `timeout`: tope de cada llamada HTTP; los reintentos se describen en «Endurecimiento».
- `max_extra`: etiquetas fuera del top 20 que se puntúan por prefijo (por defecto 4; 0 = ninguna).
- `sin_pensar`: manda `chat_template_kwargs: {"enable_thinking": false}`; solo se manda si es `true`.

#### Endurecimiento de L3 (todo configurable en `l3`; los valores son los de por defecto)
Cortacircuitos, cola acotada, reintentos y multi valen en modo `logprobs`; el cortacircuitos y el cupo por lote valen también en modo `json`.
- **Cupo por lote** (`min_por_peticion` 8, `max_por_peticion` 256). Antes `max_por_peticion` era un tope fijo de 8 llamadas para TODA la petición HTTP:
  el presupuesto lo comparten las preguntas de la petición (`conPresupuesto`) y `reservar` rechazaba desde la 9.ª pregunta («presupuesto», sin L3),
  así que en un lote de 32 las filas 9..32 se quedaban sin L3. Ahora el cupo es el número de preguntas del lote acotado a [`min`, `max`]
  (`min` > `max` ⇒ vale `max`; una llamada directa sin lote usa `min`). Para el comportamiento antiguo: `"min_por_peticion": 8, "max_por_peticion": 8`.
- **Cortacircuitos** (`circuito_fallos` 5, `circuito_espera` `"10s"`, `circuito_tasa` 0 = apagada, `circuito_ventana` 20, `circuito_apagado` false). Cuenta filas
  (pregunta entera con reintentos y relleno): fallo = error de red, 5xx, 429 sin remedio o tope de la fila vencido sin nada puntuado; una respuesta inválida
  o parcial NO es fallo. Tras `circuito_fallos` fallos seguidos, o con `circuito_tasa` > 0 una tasa de fallos ≥ ese valor en las últimas `circuito_ventana` filas,
  se **abre** `circuito_espera`: L3 devuelve `ErrCircuitoAbierto` (cuelga de `ErrSinL3`) sin llamar al LLM. Pasada la espera queda **semiabierto**: deja pasar UNA fila
  de prueba (las demás se descartan); si va bien cierra, si falla vuelve a abrir. Métricas: `chispa_l3_circuito_estado` (0 cerrado, 1 abierto, 2 semiabierto),
  `chispa_l3_circuito_aperturas_total`.
- **Reintentos** (`max_intentos` 2 = un reintento, `espera_base` `"50ms"`, `espera_max` `"1s"`). Se reintenta ante error de red, 5xx y 429 (no ante otros 4xx), con espera
  `espera_base · 2^(n-1)` hasta `espera_max` por un jitter en [0,5, 1). Un 429 espera su `Retry-After` (segundos o fecha HTTP). Si lo que queda de `timeout_fila`
  no alcanza para la espera, no se espera ni se reintenta: la fila nunca pasa de su tope. Métrica: `chispa_l3_reintentos_total{causa="red|5xx|429"}`.
- **Cola acotada** (`max_cola` 64; `0` = sin cola). Además de `max_concurrente`, como mucho `max_cola` filas esperan un hueco; la siguiente se descarta al instante con
  `ErrColaLlena` (cuelga de `ErrSinL3`, motivo «cola llena») en vez de esperar. Métricas: `chispa_l3_semaforo_en_uso` y `_capacidad`, `chispa_l3_cola_longitud`,
  `chispa_l3_descartes_total{causa="cola_llena|circuito_abierto"}`, `chispa_l3_espera_segundos`.
- **Parciales**: `chispa_l3_filas_parciales_total` cuenta las filas con alguna opción puntuada con el piso (relleno a medias por `max_extra` o por el tope de la fila).
- **`multi` por logprobs** (`multi_max_opciones` 16): **extensión solo-Go**; `model/alt/llm_logprobs.py` no tiene multi (allí «multi» es «elige la opción más probable», una elección única).
  Cada opción es un sí/no independiente: UNA llamada por opción (`max_tokens=1`, `top_logprobs=20`) con etiquetas `S`/`N`; p(opción) = p(S)/(p(S)+p(N)) sumando las variantes con espacio;
  si solo sale una de las dos, la otra recibe el piso (menor logprob visto − 1); si no sale ninguna, la fila es inválida. θ = 0,5 como en la vía JSON. Se descartó «dos etiquetas por
  opción en una llamada»: el primer token da UNA sola distribución, no n sí/no independientes. Coste: n llamadas por fila (hasta 8 en paralelo dentro del mismo hueco del
  semáforo; comparten prefijo con `--enable-prefix-caching`); más de `multi_max_opciones` opciones = no opina. El prompt es el del Python (mismas dos primeras frases, mismo escapado
  del texto y mismo formato `json.dumps`) con la instrucción `S`/`N` y un campo `"opcion"` en lugar de `"opciones"`.
- **`forzar_l3`** (solo pruebas): salta L0–L2 y manda todo a L3. El servidor **se niega a arrancar** (y a recargar) con él si no se le pasa `-pruebas`.

### L2 por kNN por tarea (`knn`): la cascada L2→L3 de R8/R10
Con los τ de CALIB del `.chispa`, la cascada L0/L1/L2→L3 escala ≈ 98,8 % de las filas a L3. La que funciona en Python (`model/alt/escalado.py`,
`exportar_banco.py`) es **L2 = kNN por tarea → L3**: con `"knn": {...}` en la config, el servidor sustituye el L2 de ONNX por `L2KNN`
(`decide/knn.go`, `decide/l2knn.go`) y, salvo `conservar_l0_l1`, salta L0 y L1 (como en Python).
```json
{"knn": {"umbral": 0.06641, "siempre": ["fakenews-es", "hatecheck-es"],
         "tareas": {"massive-es": {"banco": "bancos/banco-massive-es", "preguntas": ["¿Qué intención tiene el mensaje?"]},
                    "lexboe":     {"banco": "bancos/banco-lexboe", "preguntas": ["…"], "umbral": 0.08}}},
 "l3": {"url": "http://vllm:8000/v1", "modelo": "qwen3-4b", "modo": "logprobs"}}
```
- **Tareas y preguntas.** Una pregunta se asigna a la tarea cuyo texto de `question` coincide exactamente con uno de sus `preguntas`; si no coincide con ninguna, L2 no opina y sigue L3.
  Vale para `choice`, `noul` y `score`; `multi` no opina.
- **Banco** (formato propio, el del exportador de Python): para cada tarea, `<banco>.npy` (matriz float32 little-endian en orden C, n × dim, .npy v1/v2/v3; vectores e5 YA
  normalizados: si alguno no tiene norma 1 se normaliza al cargar) y `<banco>.jsonl` con una línea por fila, `{"etiqueta": "...", "es_oro": true|false}`, **el oro primero** y luego el fondo
  etiquetado por el LLM (si el oro no va primero, el servidor no arranca). Las rutas son relativas al directorio de trabajo. Va como ficheros, no como sección del `.chispa`
  (haría falta ampliar `format/chispa3`); un banco vacío o sin oro es válido (escala siempre).
- **El kNN** es el de `escalado.l2`: similitud coseno (producto escalar) con el banco; si el oro más cercano tiene similitud ≥ `s_oro` (0,90) votan solo los ejemplos de oro; los
  `vecinos` (32) más similares (a igualdad, el de menor índice) votan con peso exp((s − smax)/`temperatura`) (0,05) y solo cuentan los votos de las opciones de la fila; sin votos, uniforme.
  Margen = p1 − p2.
- **Escalado.** Se escala a L3 si margen < `umbral` (global, o `tareas.<t>.umbral`); SIEMPRE si la tarea no tiene oro, o si está en `siempre` (o `tareas.<t>.siempre`). En Python `umbral` es el cuantil 0,25 del margen
  en la mitad de calibración sin hatecheck-es, y «siempre» son las tareas sin oro o con AUROC del margen < 0,55 en calibración; aquí ambos se configuran a mano (el fixture de paridad trae los valores).
  Si L3 no contesta, la respuesta de la cascada es el top-1 del kNN, escalada.
- **Motivos y métricas.** `chispa_knn_filas_total{tarea}` cuenta las filas que decidió el kNN y `chispa_knn_escaladas_total{tarea,motivo}` las que escalaron, con motivo `margen`, `sin_oro` o `siempre`;
  `incompleta` cuenta las escaladas a las que L3 no pudo contestar (es un subconjunto de las anteriores, no se suma). Fracción escalada de una tarea = (margen + sin_oro + siempre) / filas.
  `chispa_knn_sin_vector_total` cuenta las filas sin vector de consulta (L2 no opina y sigue L3).
- **Vector de consulta y codificador e5.** El kNN necesita el vector e5 de la consulta (multilingual-e5-small sin ajustar, media de tokens con máscara normalizada a norma 1,
  prefijo «query: », 256 tokens, con la normalización de mayúsculas del mensaje). El ONNX del `.chispa` es el L2 ajustado (ranuras S), no e5 puro, así que el codificador es otro ONNX:
  `knn.e5_onnx` apunta al e5 exportado con `model/alt/exportar_e5_onnx.py` (revisión 614241f6, licencia MIT, `last_hidden_state`, opset 17, fp32; sha256 en `e5-onnx.json`).
  Con `knn.e5_onnx` el servidor monta `decide.CodificadorE5` (ONNX Runtime: hay que compilar con `-tags ort`; y pasar `-tok e5.chispa-tok`, el tokenizador de e5, que
  se trunca a 256 tokens como los bancos) y el kNN contesta por HTTP; sin `knn.e5_onnx` el vector se inyecta (`decide.ConVectorConsulta(ctx, v)` o un `decide.Codificador` propio) y,
  sin vector, L2 no opina y todo escala a L3. Una consulta por llamada (lote 1) y una sesión ORT serializada: el codificador es un cuello compartido por todas las peticiones.
  *Paridad de tokenización:* el tokenizador de Go (`runtime/tok`) descarta los espacios finales como sentencepiece, y el rápido de HuggingFace con el que se calcularon los vectores de los
  bancos deja un token «▁» (id 6) antes de `</s>`: «No la soporto. » da 11 tokens en HF y 10 en Go (coseno 0,96 en hatecheck-es). `CodificadorE5` añade ese token cuando el texto acaba en
  espacio (`conEspacioFinal`); `runtime/tok` no se toca (el L2 ajustado se sirve con su comportamiento actual).
  *Paridad de vectores medida* (4 244 consultas de banco-paridad, e5 fp32, ORT arm64): coseno Go/Python mínimo 1,000000, p01 1,000000; top-1 del kNN con el vector de Go = Python en el 100 %; escala y motivo 100 %.
  El criterio (coseno p01 ≥ 0,999 y top-1 ≥ 99,5 %) está congelado en `model/alt/crit-r10-t5b.json`. El test es `TestE5ParidadVectoresYDecision` (`-tags ort`; variables `CHISPA_PARIDAD_DIR`,
  `CHISPA_CONSULTAS_TEXTO`, `CHISPA_E5_ONNX`, `CHISPA_E5_TOK`, `CHISPA_ORT_LIB`; `consultas_con_texto.py` añade el texto al fixture).
  Cabeceras y métricas del codificador: `X-Chispa-Cod-Ms` y `X-Chispa-KNN-Ms` (suma de las filas de la petición), `chispa_knn_codificador_segundos` y `chispa_knn_votacion_segundos`.
  **Pool de sesiones** (`knn.e5_sesiones`, por defecto min(4, núcleos)): una sola sesión ORT serializada era el cuello con 16 clientes (e5 p50 202 ms, p95 1,3 s en la g5, frente a 8,6 ms en serie).
  Con N sesiones corren N consultas a la vez; cada una carga su copia de los pesos (≈ 470 MB en fp32; ≈ 120 MB en int8) y usa núcleos/N hilos intra-operador. Medido en local
  (Mac, 10 núcleos, 16 clientes, 400 consultas, `TestE5ConcurrenciaLatenciaPorSesiones`): 1 sesión 23 consultas/s, p50 358 ms, p95 2 279 ms; 2 sesiones 69/s, p50 137, p95 556; 4 sesiones 104/s, p50 56, p95 521.
- **Codificador alternativo: Qwen3-Embedding-0.6B** (`knn.codificador: "qwen3-emb"`; por defecto y vuelta atrás `e5-small`). `decide.CodificadorQwen` (`qwen.go`, `qwen_ort.go`): tokenizador BPE byte-level de Go (`runtime/tok/bpe.go`),
  consulta «Instruct: Classify the text to answer this question: <pregunta>\nQuery:<texto>» (L2KNN llama a `CodificarConsulta(pregunta, texto)` si el codificador la implementa), truncado a 320, pooling del último token y L2, sin normalizar mayúsculas.
  El ONNX (fp32 con `.data` al lado, o int8) se abre por ruta y sus metadatos `chispa.pooling` y `chispa.max_len` se comprueban; el servidor comprueba al arrancar que los bancos tienen la dimensión del codificador (1024). Una sola sesión ORT por defecto (≈ 2,4 GB cada una).
  Bancos: `go run -tags ort ./cmd/chispa-banco-qwen …`. Pruebas con un Qwen3 en miniatura (`testdata/qwen`, `model/alt/qwen3emb_mini.py`); la paridad con el modelo real la mide el ud (`model/alt/aws/l2qwen3emb.sh`). Ver `PRODUCTO.md`.
- **El prompt de L3 en cascada es el de `llm_logprobs.py`** (T5c; `TestL3PromptCascadaIdenticoAPython`, 192 de 192 filas escaladas del fixture con el prompt idéntico byte a byte, 45 en el
  testdata). Causas del −3,2 puntos de la primera ronda e2e en g5: (1) **las consultas se mandaron sin las descripciones de las opciones** (`consultas_con_texto.py` no las traía) y el prompt de
  Python las lleva (`"descripcion"` en cada opción): sin ellas el prompt cambia en todas las filas con descripciones (`TestL3PromptSinDescripcionesNoEsElDePython`); es la causa de las 229 filas sin timeout que
  respondían distinto (fakenews 2 opciones con descripción cada una, lexboe, massive-esc, tuits, hatecheck). Para quien llama: el prompt solo es el de Python si las opciones llevan la misma `description`.
  (2) El recorte a `top_k` (8 por defecto) y la guía de la capa previa (aquí, las probabilidades del kNN) se aplican a las preguntas `choice` con más opciones: con `knn` y sin `l3.top_k` quedan APAGADOS
  (`Config.DecisorL3` fija un top_k sin límite); con `l3.top_k` explícito vuelven a mandar. (3) El `timeout_fila` de 10 s dejaba parciales con piso a las filas de massive-es (60 opciones, relleno por `prompt_logprobs`):
  la ronda e2e usa 60 s.
- **Few-shot de L3 (fs8) y producto.** Con `l3.ejemplos` (por defecto 8; 0 = apagado), cuando L3 contesta una pregunta de una tarea con banco de oro, el prompt lleva como turnos previos usuario/asistente
  los 8 ejemplos de oro de la tarea que guarda `<banco>.ejemplos.jsonl` (los elige `model/alt/generar_banco.py` con `llm_logprobs.elegir_ejemplos`: estratificados por clase, semilla fija; el servidor no
  los vuelve a elegir, y por eso hay paridad exacta con `llm_logprobs.py --ejemplos --k 8`: sistema + nota, opciones del ejemplo solo con id y etiqueta, texto recortado a 1 200 caracteres,
  etiqueta correcta como respuesta, y el caso nuevo con sus descripciones al final). Tareas sin oro, o sin fichero de ejemplos (aviso en el arranque), contestan zero-shot. `l3.ejemplos` = N > 0 falla si el fichero trae más de N.
  Tests: `TestL3FewShotIdenticoAPython` (testdata/l3_fs_*, generado con el Python; 192 de 192 filas idénticas byte a byte con `--por-tarea 30`: 162 con fs8 y 30 zero-shot de una tarea sin oro).
  **Estabilidad por tipo:** `choice` y `noul` estables; `score` EXPERIMENTAL (la respuesta lleva `"experimental": true`; la exactitud en score medida en dev está cerca del azar: T3, jql 0,15 y usajobs 0,15);
  `multi` es una extensión solo de Go. **Límites:** máximo 62 opciones (etiquetas de un carácter); con más de ~20, preseleccione con el kNN con `l3.top_k` (implementado, sin medir con fs8); con 60 opciones L3 cuesta ≈ 1,2 s por fila en una A10G.
  El empaquetado (imagen, config de ejemplo, generador del banco, cifras y pendientes) está en `server/PRODUCTO.md`.
- **Paridad.** `decide/knn_test.go`: `TestKNNSinteticoParidadConPython` (fixture sintético en `decide/testdata/knn`, generado por `generar.py`, una reimplementación literal de `escalado.l2` en Python; corre siempre) y
  `TestKNNParidadFixtureReal` (con `CHISPA_PARIDAD_DIR=<banco-paridad>`: 4 244 consultas de 8 tareas; compara L2 top-1, escala sí/no y motivo con Python, y la exactitud de la cascada usando la `l3` del fixture).

### Flags de `chispa-server`
| flag | por defecto | uso |
|---|---|---|
| `-addr` | `127.0.0.1:8090` | dirección de escucha |
| `-config` | — | JSON de configuración (capas, τ, modelos, cachés, claves, L3) |
| `-modelo` | — | `.chispa` v3 (o v2 envuelto) del que armar L0, L1 y L2 |
| `-tok` | — | `.chispa-tok` de e5; sin él, L2 queda en stub |
| `-ort-lib` | — | `libonnxruntime.so` (precede a `CHISPA_ORT_LIB`) |
| `-timeout` | `30s` | tope por petición (503 `timeout` con `Retry-After`); 0 = sin tope |
| `-max-conexiones` | `1024` | conexiones TCP abiertas a la vez (0 = sin tope) |
| `-max-en-curso` | `64` | peticiones a `/v1/decide` en curso (0 = sin tope); al llenarse, 429 `rate_limited` |
| `-plazo-cuerpo` | `0` (= 10 s) | plazo para recibir el cuerpo de `/v1/decide`; negativo = sin plazo |
| `-pre-drenaje` | `0` | tras SIGTERM/SIGINT, `/healthz` responde 503 `shutting_down` este tiempo antes de apagar |
| `-apagado` | `25s` | plazo para terminar las peticiones en vuelo tras SIGTERM/SIGINT |
| `-pruebas` | `false` | permite las opciones solo-pruebas de la config (`forzar_l3`); sin él, el servidor no arranca con ellas |
| `-abierto` | `false` | permite escuchar fuera de loopback sin `api_keys` (o `api_keys_abierto` en la config) |
| `-admin-addr` | — | listener aparte para `POST /v1/admin/recargar` (la pública deja de servirla: 404) |
| `-metricas-protegidas` | `false` | `/metrics` exige `Authorization: Bearer` con una api_key válida (o el `admin_token`) |
| `-hash-clave` | `false` | lee una clave de la entrada estándar, imprime su SHA-256 y sale |

### Claves de API
Con `api_keys` no vacío en la config, `/v1/decide` exige `Authorization: Bearer <clave>` (401 `unauthorized` si falta o no vale) y limita por clave con un cubo
de fichas (429 `rate_limited` con `Retry-After`). Cada entrada: `{"id": "cliente-a", "hash_sha256": "<64 hex>", "rps": 20, "rafaga": 40}`; el servidor solo guarda el SHA-256,
nunca la clave en claro: `printf '%s' "$CLAVE" | chispa-server -hash-clave`. `id` (`[A-Za-z0-9_.-]`, ≤ 64) es la etiqueta de las métricas (`chispa_clave_peticiones_total`,
`chispa_auth_rechazos_total`), nunca el secreto. `rafaga` 0 = ⌈rps⌉. Escuchar fuera de loopback sin claves exige `-abierto` o `api_keys_abierto`; una recarga sin claves
tampoco las quita sin `api_keys_abierto`.

### Recarga en caliente
Dos formas, ambas sin cortar peticiones en curso: `kill -HUP <pid>` o `POST /v1/admin/recargar` con `Authorization: Bearer <admin_token>`
(la ruta solo existe si la config trae `admin_token` —mínimo 32 bytes— o `admin_token_sha256`; si no, 404). Relee la config y el `.chispa`, arma un modelo completo nuevo
(con cachés nuevas: un nonce del modelo anterior no devuelve su respuesta) y lo sustituye cuando está listo; las peticiones que ya corrían terminan con el viejo.
Se relee también `api_keys` (los cubos de las claves que siguen se conservan) y el token de admin. Si algo falla (`.chispa` corrupto, config inválida) se sigue sirviendo el modelo
anterior y se cuenta `chispa_recargas_total{resultado="error"}`. Con `-admin-addr` el endpoint sale del puerto público. Cubierto por `server/e2e/correr.sh` (pasos 5 y 8).

### L0 y caché de esquemas
L0 carga uno o varios `.chispa` y elige el modelo cuyo **conjunto de etiquetas es igual al de ids de las
opciones** (un modelo por esquema; dos modelos con las mismas etiquetas se rechazan al cargar). Sin
modelo para el esquema, L0 no opina. El modelo ve solo `state` (no la pregunta ni las descripciones, y
sin `fields`). La clave de la caché es un hash FNV-128a estable de `kind + question + options`
(ordenadas por id, con descripciones); LRU de `cache_esquemas` entradas que guarda el modelo
resuelto (o «ninguno»). Los LRU están fragmentados en 16: el tamaño es aproximado.

### Otros endpoints
`GET /healthz` → `{"status":"ok"}`. `GET /metrics`: formato de texto Prometheus escrito a mano
(`prometheus/client_golang` no está en la caché de módulos): `chispa_peticiones_total{codigo}`,
`chispa_preguntas_total`, `chispa_decididas_total{capa}`, `chispa_escalados_total`,
`chispa_capa_latencia_segundos` (histograma por capa), `chispa_capa_errores_total{capa}`,
`chispa_cache_esquemas_total{resultado}`, `chispa_cache_nonce_aciertos_total`, y, según el caso, `chispa_recargas_total{resultado}`,
`chispa_torres_*`, `chispa_esquema_nuevo_*`, `chispa_l3_llamadas_total`, `chispa_clave_peticiones_total`, `chispa_auth_rechazos_total`, `chispa_timeouts_total`.

## Relación con `/v1/systemone` de Jev — SUPOSICIÓN, sin verificar
No tengo la especificación de `/v1/systemone`. Lo que supuse, por el encargo:
- Se parece: petición con `state` + lista `questions` de `{id, question, kind, options[{id, description}], nonce?}`,
  y `nonce` devuelto tal cual. Se parecen los tres `kind`.
- **Suposición**: la respuesta es una lista por pregunta con el mismo `id`. Aquí va envuelta en `{"results": [...]}`;
  si systemone devuelve un array desnudo o usa otra clave, hay que cambiar `respuestaHTTP` (`decide/http.go`).
- No se parece / es nuestro: `probs`, `layer`, `escalate`, `ms`, y el 400 todo-o-nada con `{"error": {"code", "message", "question_id"}}`.
  Los nombres de campo de salida (`answer`, `probs`) son los del encargo, no los de Jev.

## Pruebas y modelos de prueba
`decide/testdata/*.req.json|resp.json`: golden del contrato (`go test ./decide -update` los reescribe;
`ms` se normaliza a 0). `testdata/*.chispa` (`intencion`, `injection`, `urgencia`) son los
`*-chispa-words` que entrena la Arena (`arena dataset` + `arena train`, datos sintéticos) copiados tal cual;
el `.gitignore` de `server/` deshace el `*.chispa` de la raíz para ellos.

## Medidas (microVM kindling de 4 vCPU, Xeon 3,3 GHz; cota del i3, no el objetivo)
Modelo `intencion` (4 clases), un mensaje de ~70 caracteres, L1–L3 stubs, caché de esquemas caliente.
Servidor y generador comparten la misma VM (el generador, con GOMAXPROCS 1–2, también consume CPU).

| Prueba | Núcleos del servidor | decisiones/s | p50 | p99 |
|---|---|---|---|---|
| HTTP, 1 pregunta/petición, 8 conexiones | 1 | 2 176 | 2,6 ms | 17,0 ms |
| ídem | 4 | 1 891 | 3,1 ms | 18,8 ms |
| HTTP, 16 preguntas/petición, 8 conexiones | 1 | 11 323 | 9,6 ms | 38,1 ms |
| ídem | 4 | 13 011 | 6,9 ms | 60,2 ms |
| `go test -bench L0` (sin HTTP) | 1 | ~160 k (≈6 µs/op) | | |
| ídem | 4 | ~130 k (≈7–9 µs/op) | | |

Latencias de las filas HTTP: por petición (con 16 preguntas, p50 es de la petición entera).
Los 4 vCPU de la VM **no escalan**: el modelo suelto (`PredictFull` sin caché) hace ≈3,7 µs/op con 1 y
con 4 núcleos, así que «4 núcleos» aquí no es una medida fiable de paralelismo. Medidas ruidosas
(máquina compartida): ±30 % entre repeticiones. El coste lo dominan HTTP/JSON, no L0: con la
petición de 1 pregunta el servidor hace ≈2 k/s por conexión-ciclo; el lote amortiza. El objetivo del
plan (≥ 5 k/s con L1 en el i7) no se puede confirmar desde aquí.

## T4.1b: `--modelo x.chispa` (L1 real y L2)
`chispa-server --modelo x.chispa` abre un `.chispa` v3 (o un v2, que `format/chispa3` envuelve con solo L0) y
arma la cascada con las secciones presentes (`decide.ArmarDesde`); las que faltan quedan como stub.
- **L0**: la sección `L0` se añade a los modelos de `config.modelos`.
- **L1** (`decide/l1.go`, sección `L1` con `runtime/l1`): embedding del mensaje y de `pregunta · opción · descripción`
  de cada opción (cacheados por `ClaveEsquema`, por id de opción: la clave ignora el orden). `choice`: softmax(coseno / T)
  con la `T` de `CALIB.tipos.choice`. `noul`: la consulta es la opción `true`; p(true) = σ(a·coseno + b) con `a`, `b`, θ de
  `CALIB.tipos.noul`. `score`: L1 sí opina: softmax(coseno / T) sobre los niveles (`T` de `CALIB.tipos.score`, o la de choice si falta) y `answer` es el nivel mediano (primero con masa acumulada ≥ 0,5), con su probabilidad como `conf`; una masa repartida entre niveles lejanos baja la confianza y escala. Sin tipo en CALIB: T = 0,05, a = 4, b = −2, θ = 0,5.
- **τ**: sin `--config`, el τ de L1 y L2 es el `tau_escalate` de CALIB; con `--config`, manda la config.
- **L2** (`decide/l2.go` + `l2_ort.go`, secciones `L2` y `ESQUEMAS`): el ONNX da las ranuras S (K×d) del mensaje; la torre de cada
  opción sale de ESQUEMAS (`HashEsquema(pregunta, [id])` en `choice`; la lista completa de ids, `["false","true"]` en `noul`
  y `["1".."K"]` en `score`, para el vector de consulta). Puntuación MaxSim `max_k softplus(h·s_k)`, h = int8·escala;
  luego `choice` softmax(s/T), `noul` σ(a·s+b), `score` CORAL con `scoreK` / `scoreK#j` (`model/l2/README.md`).
  La puntuación (`l2.go`) no depende de ORT y está probada con una sesión falsa; solo `l2_ort.go` toca ONNX Runtime.
  Sin `-tags ort`, `l2_stub.go` devuelve `ErrSinORT` y L2 queda como stub (el servidor avisa en el log).

### T2.6b: L2c, «L2 conjunta» (sección `CONJ`)
Si el `.chispa` trae `CONJ`, `ArmarDesde` monta `L2c` (`decide/l2c.go`) en lugar de L2. Sigue llamándose `L2` en la cascada, la
config y `layer`: mismo contrato `/v1/decide` y misma τ (la de CALIB sin `--config`). La cascada le pasa lo que dijeron las capas
anteriores (`DecisorConPrevias`), y L2c usa las probabilidades de L0 con:
- `a = log P_L0`, `b = simproto` (media sobre las ranuras del mensaje de la similitud máxima con las ranuras del prototipo de la
  clase; prototipos int8 de `CONJ`), `c = log P_L2` con `P_L2 = softmax((MaxSim + β·simproto)/T)` (β y T del modelo, en `CONJ`);
  cada una centrada por fila y dividida por su desviación de train, más las marcas «es la primera de L0 / de L2 / de ambas».
- `softmax(w · [a, b, c, top0, top2, top0·top2])` con los 6 pesos de la cabeza `lr` de `model/l2/conjunta.py`.
- La clave de `CONJ` es `HashEsquema(pregunta, ids ordenados)`: la pregunta debe ser la misma que al entrenar.
- Cae a L2 de siempre (sin tocar la cabeza) si: la pregunta no es `choice`, `CONJ` no tiene ese esquema, L0 no dio
  probabilidades (L0 no tiene modelo para el esquema), o las ranuras no tienen la `d` de los prototipos. Una opción sin clase
  en `CONJ` vale simproto 0, como en Python.
- El codificador va detrás de `Similitudes` (el ONNX y las torres): `l2c_test.go` prueba con similitudes falsas sin ORT.
- **Paridad con Python**: `TestL2cParidadConPython` lee `model/l2/tests/fixtures_conjunta/paridad.json` (lo genera
  `sdk/python/tests/fixture_conjunta.py` con `conjunta.Cabeza`): |Δp| máx 8e-8 en 8 casos (tolerancia 1e-4).
- **L0TF (T4.5c)**: si el `.chispa` trae `L0TF`, `ArmarDesde` registra ese L0 TF-IDF (`l0.go`, `chispa3.L0TFModelo.Probs`) y no el de la sección
  `L0`; la capa L0 de la cascada y la entrada de L0 de L2c usan sus probabilidades. Sin `L0TF`, el L0 de Chispa como antes.
  Paridad con sklearn: `TestL0TFParidadConSklearn` (`testdata/l0tf`, 270 frases con tildes, emojis, mayúsculas y espacios raros; lo genera
  `sdk/python/tests/fixture_l0tf.py`): |Δp| máx 4e-8 en f32 (tolerancia 1e-5); con datos reales (`L0TF_REAL=dir`), 4e-8 en 1500 frases de
  massive y de clinc150. Latencia: `go test ./decide -bench L0TF` (80 000 n-gramas × 150 clases, f32, frase de 14 palabras): 0,10 ms por consulta.
- **Latencia** (medida antes de T4.5; después, ver «T4.5»): la cabeza no añade llamadas al ONNX (L2c usa las mismas ranuras S que L2); el coste extra es el simproto, que va
  lineal con las clases: C·K·K·d multiplicaciones (150 clases, K=16, d=256: ≈ 10 ms en esta microVM de 4 vCPU, Go escalar,
  `go test -bench L2c ./decide`; con 2–10 clases, < 1 ms). El resto de la cabeza (logits y softmax) son µs. Frente a los ~30 ms de
  L2 no son despreciables con muchas clases; si hace falta, el bucle por clases se puede paralelizar o vectorizar.
- Sin L0 en la cascada, L2c no puede usar la cabeza y se comporta como L2.

### Compilar y probar con ORT
`go.mod` trae `github.com/yalue/onnxruntime_go v1.38.0` (probado con `libonnxruntime.so.1.31.1`). Sin `-tags ort` no se usa
(`l2_stub.go`, `torre/ort_stub.go`). Receta sin red (nodo-b; la caché de módulos de `~/go/pkg/mod` tiene el módulo):
```
. ~/orquestador/herr/env.sh                      # Go, venv, GOPROXY=off
export CC=~/orquestador/herr/zigcc CGO_ENABLED=1 # para -race
export CHISPA_ORT_LIB=~/orquestador/herr/venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.1
cd server && go test -tags ort -race -count=1 ./...
```
Ruta de la .so en `chispa-server`, de mayor a menor precedencia: `-ort-lib` > `CHISPA_ORT_LIB` > `-ldflags -X …/server/decide.RutaORT=` >
búsqueda por defecto del sistema. `main` la asigna a `decide.RutaORT` y `torre.RutaORT` (`torre` no puede importar `decide`), así que L2 y la
torre usan la misma. Al arrancar, si el `.chispa` trae L2 o TORR y ORT no carga, el log dice `AVISO: ONNX Runtime no carga (ruta …, origen)`
(y L2 queda en stub / `chispa_torres_calculadas_total` en 0). Test: `cmd/chispa-server/ort_si_test.go` (un proceso hijo por origen de la ruta).
Para (re)añadir el módulo sin red: `GOPROXY=file://$HOME/go/pkg/mod/cache/download GOSUMDB=off GOFLAGS=-mod=mod go get github.com/yalue/onnxruntime_go@v1.38.0`.
Tests ORT: `torre/ort_test.go` (`sintetico.onnx` y su referencia) y `decide/l2_ort_test.go` (`TestL2TorreORTEsquemaNuevo`: L2 con
`decide/testdata/l2_sintetico.onnx` + torre al vuelo + `/v1/decide` con un esquema nuevo, un solo entorno ORT; se regenera con
`python decide/testdata/l2_sintetico.py`). Los ONNX son sintéticos (pesos aleatorios): prueban la mecánica, no la calidad.
El tokenizador de e5 llegó en T4.5 (abajo): `chispa-server -tok e5.chispa-tok`; sin `-tok`, L2 queda en stub aun con `ort`.

### Medidas de L1 real (microVM kindling de 4 vCPU, Xeon 3,3 GHz; cota del i3)
Potion multilingüe (`runtime/l1/exportar.py`, 155 MB, en un v3 en memoria), `go test ./decide -bench L1Real -benchtime 3s -cpu 1,4`:
cascada completa (L0 sin modelo para el esquema → L1) en lotes de 16 preguntas de ~70 caracteres, 3 esquemas con caché
de esquemas caliente, **sin HTTP/JSON**, 4 hilos de goroutine solo en la fila `-4`:

| Núcleos | decisiones/s | p99 por lote de 16 |
|---|---|---|
| 1 | ≈ 24 000 | 1,2 ms |
| 4 | ≈ 49 000 | 5,4 ms |

Con HTTP y JSON el techo medido en T4.1 era ≈ 13 k/s con L0, así que el servidor completo con L1 queda entre eso y estas cifras;
no se midió por HTTP con el potion. El objetivo del plan (≥ 5 k/s en el i7) tiene margen en CPU, pero no se confirma aquí.
Máquina compartida: ±30 %. Sin caché fría de páginas del mmap (no se puede vaciar en la VM).

### T4.5: tokenizador de e5 en Go, simproto entero y e2e
- **Tokenizador** (`runtime/tok`, Go puro, `README.md` allí): `chispa-server -tok e5.chispa-tok` lo pasa a `ArmarDesde`, que monta L2 / L2c
  con el ONNX. El `.chispa-tok` se exporta con `runtime/tok/exportar.py <dir-e5> x.chispa-tok` (12 MB, no va a git).
- **simproto entero** (`decide/l2c.go`): cada ranura S se cuantiza a int8 (escala max|S|/127) y el producto con los prototipos int8 se
  hace en enteros, tres elementos por multiplicación de 64 bits y cuatro ranuras por pasada; los prototipos se empaquetan una vez por
  entrada de CONJ. `go test ./decide -bench 'Simproto150|L2cCabeza'` (150 clases, K=16, d=256, 16 ranuras; microVM Xeon 3,3 GHz, 4 vCPU):
  | | antes (float32 escalar) | después |
  |---|---|---|
  | `BenchmarkSimproto150` | 10,5 ms | 2,3 ms |
  | `BenchmarkL2cCabeza` (cabeza entera) | 10,1 ms | 2,4 ms |
  Coste: la cuantización de S mueve las probabilidades (|Δp| ≤ 3e-3 con el fixture de Python, que tiene d pequeña) frente a los 8e-8 del
  cálculo en float; ese cálculo sigue en `simprotoRef` y las pruebas de paridad estricta lo usan. El efecto en macro-F1 se mide en `e2e.sh`.
- **`MaxOpciones` 64 → 256**: clinc150 trae 150 clases (151 con `escalate`).
- **`deploy/aws/e2e.sh`**: prueba de punta a punta en una c6a.xlarge spot (`__CHISPA_S3__`, `empaquetar_e2e.sh > codigo.tgz`). Pasos en la
  cabecera del script. En seco: `CHISPA_LOCAL=1 bash server/deploy/aws/e2e.sh` (fixtures, sin ORT).
