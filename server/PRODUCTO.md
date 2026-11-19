# Chispa como servidor general de clasificación: dos modos

Chispa es un servidor de clasificación general (choice, score, sí/no) con **dos modos** de despliegue («opción A»). El Chispa clásico (`.chispa` entrenado, L0/L1/torres, móvil) queda congelado: no cambia
y sus configs siguen arrancando igual. En uso general L0 y L1 están apagados (con la sección `knn` la cascada es L2→L3 salvo `knn.conservar_l0_l1`) y el escalado depende del modo.

| | **Modo GPU** (`deploy/config.modo-gpu.json`) | **Modo CPU** (`deploy/config.modo-cpu.json`) |
|---|---|---|
| Quién contesta | L3 = Qwen3-4B por logprobs con 8 demostraciones (fs8), siempre | solo el kNN (e5 por defecto), siempre; sin LLM |
| `knn.modo` | `"gpu"`: todo escala a L3; el kNN elige las demostraciones de cada consulta (codifica la consulta) y sirve para el top-k de las preguntas con más de 20 opciones | `"cpu"`: contesta L2 siempre, sin escalado por margen ni `siempre` |
| Ejemplos | del banco (`<banco>.ejemplos.jsonl`, el pool de demos fs8) o de la petición (`examples[]`); las 8 demos son **las más cercanas a cada consulta** (kNN), ver «Demostraciones por consulta» | **obligatorios**: en la petición o de un banco; si no, 400 `examples_required` |
| Exactitud macro sellada (§17.26) | **0,501** (final todo-L3) | **0,402** (kNN) |
| Hardware | una GPU (A10G: 10,6 a 12,4 peticiones/s) + vLLM | CPU: e5 ≈ 9 ms por consulta; Qwen3-Embedding ≈ 1 s en CPU |

- **Cuándo elegir cada uno**: GPU si hay una GPU y se quiere la mejor exactitud (+9,9 puntos de macro sobre el kNN); CPU si no hay GPU, la latencia manda o el coste de una GPU no compensa. En `choice` la brecha casi desaparece:
  **kNN 0,483 frente a 0,485 del 4B zero-shot** (selladas), así que en choice el modo CPU pierde poco; la ventaja del modo GPU está en `score` y en las tareas con pocos ejemplos o con muchas opciones.
- **e5-small sigue por defecto** (`config.ejemplo.json`, `config.modo-cpu.json`); Qwen3-Embedding (`config.modo-gpu.json`, `config.ejemplo-qwen3emb.json`) es la opción con GPU o sin prisa. Cada modo es una decisión por despliegue.
- Las cifras selladas son las de §17.26 (LEEME de `model/alt/resultados-20261113-p0-p2/`); las medidas del modo GPU por el servidor (e2e fs8 con vLLM real) están pendientes: ver «Pendiente».

## Respuesta del kNN: la regla de `score`
En el camino kNN (e5 y Qwen), `answer` en las tareas `score` es la **mediana ordinal (CORAL)** de las `probs` del kNN, exactamente `model/alt/lineas_base_v2.top1(probs, r)` con `esp=False` (el primer nivel, ordenados por su valor
entero, cuya cola acumulada no pasa de 0,5); `choice` y sí/no siguen con el argmax (en un empate, la primera opción). `probs` no cambia. En dev (L2-qwen3emb, 6 tareas, 8 vecinos) leer las mismas `probs` con la regla da
**0,4700 frente a 0,4609** (v8; 0,4603 con v32) del argmax que devolvía el servidor. Paridad Go↔Python con `server/testdata/w1_score_regla.json` (419 filas generadas por `top1`: empates, colas exactamente en 0,5, opciones desordenadas);
las filas guardadas del e2e de L2-qwen3emb no están en el repo, así que el 0,4700 se reproduce solo en el sentido de que la regla Go da, fila a fila, lo mismo que Python sobre un fixture sintético.
Si los ids de `score` no son enteros o las probabilidades suman 0, el servidor se queda con el argmax.

## Demostraciones por consulta (modo gpu, fs8-kNN)
En modo gpu las 8 demostraciones del prompt de L3 no son fijas por tarea: son **las 8 del soporte más cercanas a cada consulta**, como `lineas_base_v2_pred demos` / `lineas_base_v2d.py` (fs8-knn) y la apertura sellada (`apertura_v2.py`, LEEME §17.24 D y §17.26).
El soporte es el pool de `<banco>.ejemplos.jsonl` (todo el oro del banco) o, si la pregunta trae `examples[]`, esos examples. L2 codifica la consulta (un coste de codificador por fila: con Qwen en CPU ≈ 0,2 a 1 s) y deja las demos elegidas a L3 (`Respuesta.Demos`).
- **Regla** (`SeleccionarDemos`, idéntica a `lineas_base_v2_pred.demos_fila`; fixture `testdata/w1_demos_knn.json` de `model/alt/w1_fixture_demos.py`, 8 casos con empates exactos, soportes menores que k, k clases justas y etiquetas que ordenan distinto como texto y como número):
  similitud = coseno de la consulta con cada ejemplo (float64 redondeado a float32); orden estable por similitud descendente y, a igualdad, posición en el soporte; si el soporte tiene **k clases o menos**, la más cercana de cada clase (clases en orden alfabético) y el resto por cercanía hasta k (cuota por clase);
  si tiene más, las k más cercanas; **orden en el prompt: de menor a mayor similitud** (la más parecida pegada a la pregunta; a igualdad, el orden de elección). Con menos de k ejemplos salen todos. `k = l3.ejemplos` (8).
- **Formato pool de `<banco>.ejemplos.jsonl`**: una línea por ejemplo de oro, las de siempre (`id, pregunta, tipo, texto, opciones, etiqueta`) más `fila` (índice de su vector en el `.npy`, del oro; el servidor comprueba que la etiqueta coincide con la del banco y no arranca si no) y, en las elegidas para el modo fijo, `fs` (su posición, desde 1).
  `chispa-banco-qwen` lo escribe por defecto (`-ejemplos-fijos` escribe el formato viejo: solo las k fijas, sin `fila`). Un fichero del formato viejo (los de `generar_banco.py` de e5, que no cambia) se sigue leyendo y esa tarea usa sus ejemplos fijos; para tener kNN con un banco de e5 hay que regenerarlo con el pool.
- **Modo fijo** (opción, no por defecto): `l3.demos_fijas: true` (o bancos/ejemplos sin codificador: sin vector de consulta no hay kNN) deja las demos como antes: las del fichero marcadas `fs` (o todas, en el formato viejo) y, con `examples[]`, una vuelta por clase en orden de aparición. Fuera de modo gpu todo es fijo.
- **Diagnóstico**: `chispa-server -demos-log FICHERO` escribe una línea por pregunta `{id, estado_sha, tarea, demos: [ids]}`; `-pruebas -codificador-prueba N` pone vectores inventados por hash (solo la prueba en seco del ud).
- Los examples de la petición cuestan ahora su codificación aunque todo escale (se guardan en la caché de vectores igual); el tope `max_ejemplos*` sigue valiendo.

## Ejemplos en la petición (`examples[]`)
Cada pregunta puede traer `"examples": [{"text": "…", "label": "<id de una opción>"}, …]` (no en `multi`). Validación (400, `question_id` y la posición del ejemplo en el mensaje): `invalid_example` (texto vacío, sin etiqueta),
`invalid_example_label` (la etiqueta no es una opción), `too_many_examples` (por defecto 256 por pregunta y 1 024 por petición; `limites.max_ejemplos` y `limites.max_ejemplos_peticion`: cada ejemplo sin caché es una codificación) y `field_too_long` (16 KiB por `text`; `limites.max_ejemplo_texto`).
- **Regla si llegan banco y examples**: los de la petición **sustituyen** al banco de esa pregunta (no se mezclan; el fondo y los umbrales de score del banco no se usan). La pregunta no tiene que estar en `knn.tareas`: sin tarea se usan el umbral global y el kNN cuenta como tarea `(peticion)` en las métricas.
  Con examples, L3 usa como demostraciones fs8 hasta `l3.ejemplos` de ellos: en modo gpu las más cercanas a la consulta (ver arriba); en el resto, una vuelta por clase en orden de aparición, sin azar (mismo prompt para la misma petición). Sin `examples` la respuesta es la de siempre, byte a byte (test), y los bancos fuera de línea no cambian.
- **Caché de vectores**: los vectores de los ejemplos se guardan por clave sha256 del texto ya preparado para el codificador (Qwen: la consulta con instrucción y pregunta; e5: texto con las mayúsculas normalizadas) + codificador + revisión del modelo,
  en una LRU con tope de memoria (`knn.cache_vectores_mb`, 64 por defecto) y sin codificar dos veces lo mismo si dos peticiones lo piden a la vez. La primera petición con N ejemplos nuevos cuesta N codificaciones (con Qwen en CPU ≈ 1 s cada una); las siguientes, ninguna. El caché de respuestas por `nonce` incluye los ejemplos en su clave.
- `knn.tareas` puede estar vacío si hay codificador en el servidor (`e5_onnx` o `qwen3-emb`): entonces solo hay kNN con examples de la petición.

# Chispa como producto: «trae entre 8 y 32 ejemplos por clase»

El cliente trae unos pocos ejemplos de oro por clase; el servidor responde con un clasificador de dos pasos:

1. **L2 = kNN por tarea** sobre esos ejemplos (vectores e5), con el **margen** p1 − p2 como confianza. Si el margen es ≥ u, contesta L2 (≈ 1 ms).
2. Si no, **L3 = Qwen3-4B-Instruct-2507 por logprobs** con 8 demostraciones de oro en el prompt (**fs8**). Una tarea sin oro contesta siempre así, zero-shot.

## Qué entra
- **Ejemplos de oro** (el cliente): un `jsonl` por tarea, una fila por ejemplo, con `id`, `pregunta`, `tipo`, `texto`, `opciones` (ids), `etiqueta` (id correcto) y, opcional, `descripciones` {id: texto}.
- **Fondo etiquetado por el LLM** (opcional): textos del cliente con la etiqueta que les puso L3 (`llm_logprobs.py`); votan en el kNN detrás del oro. Sin él, el banco es solo oro.
- **Generar el banco** (una vez por tarea, con torch 2.5.1 y transformers 4.46.3; los vectores son los del servidor):
  ```
  python3 -m model.alt.generar_banco --tarea mi-tarea --oro oro.jsonl [--fondo fondo.jsonl] --salida bancos/ [--k 8] [--cache CACHE_HUB_HF]
  ```
  Escribe `banco-<tarea>.npy` (float32 n × 384, e5, normalizado, oro primero), `banco-<tarea>.jsonl` (`{"etiqueta","es_oro"}`), `banco-<tarea>.ejemplos.jsonl` (los k ejemplos de fs8,
  elegidos por clase con semilla fija: el servidor los usa tal cual y el prompt es el de `llm_logprobs.py --ejemplos --k 8` byte a byte) y `banco-<tarea>.meta.json`.
  Probado: los vectores de las 16 filas de oro de hatecheck-es coinciden exactamente (diferencia máxima 0) con los del banco de referencia de R10.
- **El servidor** (`server/deploy/Dockerfile.producto`, `config.ejemplo.json`): `knn` (bancos, `e5_onnx`, umbral u), `l3` en modo `logprobs` con `ejemplos: 8`, `timeout_fila` 30 s,
  `max_concurrente` 32, `max_cola` 64, `tau` 0, y `api_keys`. `e5.onnx` sale de `model/alt/exportar_e5_onnx.py` (revisión 614241f6, MIT, fp32) y `e5.chispa-tok` de `runtime/tok/exportar.py`.
  **u hay que calibrarlo por cliente** (el 0,06641 del ejemplo es el de la batería de R10: el cuantil 0,25 del margen sin hatecheck). Una tarea se fuerza a L3 con `knn.siempre`.

## Qué sale
`POST /v1/decide` con el contrato de siempre: `answer`, `probs` (una por opción), `layer` (`L2` o `L3`), `escalate`, `ms`. En `score` sale además `experimental: true`.
Cabeceras de diagnóstico: `X-Chispa-Cod-Ms` (codificador e5), `X-Chispa-KNN-Ms`, `X-Chispa-L3-Espera-Ms` y `X-Chispa-L3-Ms`. Métricas Prometheus: `chispa_knn_*` (filas y escaladas por tarea y motivo
`margen`/`sin_oro`/`siempre`/`incompleta`), `chispa_l3_*` (semáforo, cola, circuito, reintentos, descartes, parciales).

## Cifras medidas (R10)
- **Exactitud** de la cascada por HTTP con el codificador y L3 reales (4 244 consultas, 8 tareas): macro sin hatecheck **0,7218 frente a 0,7233 de Python** (−0,15); L2 y la decisión de escalar coinciden con Python al 100 %,
  los vectores del e5 en Go coinciden con los de Python (coseno 1,000000; top-1 del kNN 100 %), y el prompt de L3 es idéntico byte a byte al de `llm_logprobs.py` (192 de 192 filas). Escala el 47,6 % de las filas.
  El fs8 está probado en paridad de PROMPT (192 de 192 filas con la elección de ejemplos de Python); su exactitud con vLLM real no está medida en una ronda e2e (pendiente).
- **Caudal**: 10,6 a 12,4 peticiones/s por g5 (A10G) con la mezcla real, limitado por vLLM; el servidor Go y el kNN no son el cuello.
- **Latencia**: kNN ≈ 1 ms; codificador e5 ≈ 8,6 ms en serie y, con 4 sesiones ORT y 16 clientes, p50 56 ms en local (con una sola sesión era 202 ms p50 y 1,3 s p95 en la g5);
  L3 ≈ 100 ms p50 por fila con pocas opciones y **≈ 1,2 s por fila con 60 opciones** en una A10G (relleno por `prompt_logprobs` de las etiquetas que no salen en el top 20).

## Elegir el codificador del kNN: e5-small (por defecto) o Qwen3-Embedding-0.6B
**qwen3-emb: recomendado con GPU o cuando la latencia no importe.** En CPU (c7i.xlarge, 4 hilos) da p50 ≈ 1 s por consulta en textos largos, 2,2 GB de RAM y una imagen de ≈ 2,4 GB; a cambio, +2,6 puntos de macro sobre e5 con 8 vecinos en R11 (0,4683 frente a 0,4429), que el servidor Go reproduce por HTTP (0,4700). e5-small sigue por defecto.
`knn.codificador` vale `"e5-small"` (por defecto: una config sin el campo arranca exactamente como antes; hay un test que lo prueba) o `"qwen3-emb"`. **Es una decisión por despliegue y los bancos no son intercambiables**
(e5 da vectores de 384, Qwen de 1024): al cambiar de codificador hay que regenerar los bancos con el nuevo, y el servidor se niega a arrancar si la dimensión de algún banco no es la del codificador.
- **Por qué existe**: en R11 (Python, 6 tareas de dev, solo kNN con 8 de oro por clase) Qwen3-Embedding-0.6B ganó a e5-small: macro 0,4675 con 32 vecinos y 0,4683 con 8 (+7,7 con 32 vecinos; +2,6 frente a e5 con 8).
- **Config** (ejemplo completo: `server/deploy/config.ejemplo-qwen3emb.json`):
  ```
  "knn": {"codificador": "qwen3-emb", "qwen_onnx": "/modelos/qwen3emb-fp32.onnx", "qwen_tok": "/modelos/qwen3emb.chispa-bpe", "qwen_sesiones": 1, "vecinos": 8, ...}
  ```
  `vecinos` (con qwen3-emb, 8 por defecto: ganó en dev, 0,4700 frente a 0,4686 con 32; con e5 siguen 32) y `umbral` valen igual que con e5; `e5_onnx` y `e5_sesiones` no se admiten con Qwen (error de config); `-tok` no se usa. Binario con `-tags ort`.
- **Artefactos** (todos por revisión fijada `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, Apache-2.0):
  - ONNX: `python model/alt/exportar_qwen3emb_onnx.py --modelo DIR --salida DIR_SALIDA --comprobar` → `qwen3emb-fp32.onnx` + `qwen3emb-fp32.onnx.data` (los pesos pasan de 2 GB: el `.data` va SIEMPRE al lado y el servidor abre el ONNX por ruta) y `qwen3emb-int8.onnx` (int8 dinámico de ORT). Los metadatos llevan `chispa.max_len=320` y `chispa.pooling=last_token`; el servidor los comprueba al abrir.
  - Tokenizador: `python runtime/tok/exportar_bpe.py DIR_MODELO qwen3emb.chispa-bpe` (BPE byte-level en Go puro, `runtime/tok/bpe.go`; NFC, el regex de Qwen2 a mano, fusiones y los especiales que ponga transformers).
  - Bancos: `chispa-banco-qwen -onnx … -tok … -soporte soporte.jsonl -salida DIR` (Go, sin torch: usa la misma ruta de codificación que el servidor, así que banco y consultas comparten codificador por construcción; el `soporte.jsonl` lleva `tarea, etiqueta, pregunta, texto` por fila y el formato de salida es el de siempre, `.npy` + `.jsonl`). Elegir el oro (8 por clase) queda fuera: en la medida lo hace `model/alt/aws/qwen3emb_filas.py` con `muestrear(ys, 8, 0)`.
    La herramienta escribe también `banco-<tarea>.ejemplos.jsonl` (los ejemplos fs8 de L3) con el formato y la elección de `generar_banco.py` (estratificados por clase con la semilla 20261115; el Mersenne Twister de CPython está reimplementado en Go y un test lo compara byte a byte con ficheros de Python).
    Las filas del soporte deben llevar además `id`, `tipo` y `opciones` (ids) (y `split`: solo cuenta `train`); flags `-k` (8 por defecto; 0 = sin fichero) y `-semilla`. Si las filas no traen esos campos, avisa y no escribe el fichero (L3 contesta zero-shot); con `-k` explícito es un error.
    El fichero es ahora el pool (todo el oro con `fila`, las k fijas con `fs`; ver «Demostraciones por consulta»); en modo gpu las demostraciones son las k más cercanas por fila, como `lineas_base_v2_pred demos`. La medida de L2-qwen3emb es solo L2 (L3 apagado).
- **Qué reproduce el codificador** (y nada más): consulta de soporte y de test «Instruct: Classify the text to answer this question: <pregunta>\nQuery:<texto>» (`<pregunta>` es el `question` de la petición); descripciones de etiqueta sin instrucción; truncado a 320 tokens; pooling del último token y norma L2. **No** normaliza las mayúsculas del mensaje (el camino de e5 sí; R11 no lo hacía con Qwen).
- **Memoria y CPU**: cada sesión ORT carga su copia de los pesos (≈ 2,4 GB fp32, ≈ 1,2 GB int8): `qwen_sesiones` vale 1 por defecto (usa todos los núcleos en cada consulta), no el min(4, núcleos) de e5.
- **Reglas de elección** (criterio congelado `model/alt/crit-l2-qwen3emb.json`): int8 solo si su macro e2e queda a ≤ 0,5 puntos del fp32 con los mismos vecinos; si la p50 por consulta en CPU (c7i.xlarge, 4 hilos) del modo elegido pasa de 150 ms, esta nota pasa a ser «**recomendado con GPU o cuando la latencia no importe**».
- **Cifras de la medida en c7i.xlarge** (EC2 i-052a22f433f725fe2, 4 vCPU Xeon 8488C, 4 hilos, 1 sesión ORT, fp32; detalle en `model/alt/resultados-20261113-p0-p2/LEEME.md` §19 y `l2-qwen3emb-5d9e86c.json`):
  - Paridad con transformers 5.22.2 y el modelo real: tokens 200/200 iguales, coseno mínimo 1,000.
  - Macro e2e por HTTP en dev (6 tareas, 2930 filas, solo L2): **0,4700 con 8 vecinos** (R11 0,4683) y 0,4686 con 32 (R11 0,4675). Cobertura 100 %.
  - Latencia por consulta: p50 1000 ms, p95 1026 ms; casi todo es el codificador (el kNN, 0,07 ms). Crece con la longitud: mediana 242 ms en reseñas cortas, 730 ms en quejas y ≈ 1000 ms en los textos que llegan al tope de 320 tokens. Con textos cortos (fixture de paridad), p50 212 ms.
  - RAM: 2,2 GB de RSS con una sesión (e5: ≈ 0,5 GB por sesión). Tamaños: ONNX fp32 2,38 GB, tokenizador 3,9 MB, libonnxruntime 29 MB, binario 12,9 MB. Imagen estimada ≈ 2,4 GB frente a ≈ 0,5 GB con e5.
  - **int8 no viable**: la cuantización dinámica (por canal, solo MatMul, sin las `down_proj`) da coseno 0,980 < 0,99 frente a torch; solo se ofrece fp32.
  - Sin medir: bajar `max_len` (128 o 192) recortaría la latencia, pero cambia la precisión.
- **Vuelta atrás**: poner `"codificador": "e5-small"` (o quitar el campo y los `qwen_*`), `e5_onnx` y `-tok` de e5 y los bancos de e5, y reiniciar. No hay migración de bancos en ningún sentido. Con la imagen `Dockerfile.producto`, el `CMD` lleva `-tok` de e5: con Qwen se arranca con `docker run … chispa-producto -config /etc/chispa/config.json -timeout 60s`.

## Límites
- **Máximo 62 opciones** por pregunta (etiquetas de un carácter A–Z, a–z, 0–9); con más, la fila es inválida y escala. En `score` con niveles de una cifra la etiqueta es el nivel.
- **Con más de 20 opciones, L3 ve solo el top-10 del kNN** (`l3.top_k_knn`):
  - Con `knn` el valor por defecto es 10; `0` lo apaga, y un `l3.top_k` explícito también lo deja apagado.
  - El ranking va por probabilidad del kNN y, a igualdad, por similitud al centroide de oro.
  - Las opciones se muestran en el orden de la pregunta.
  - Cifra de T3b en finanzas (59 opciones): el 4B zero-shot pasa de 0,703 a **0,813** (+10,9 [+6,6, +15,6]) con cobertura del 100 %, y el recall@10 del kNN es 0,989.
  - En Go el ranking solo está probado con datos sintéticos: no hay vectores e5 en el fixture.
  - Las 10 candidatas de Python por fila están exportadas en el fixture de paridad.
- Cada ejemplo del prompt se recorta a 1 200 caracteres; los ejemplos van solo con id y etiqueta (sin descripciones).
- Una tarea sin oro (o sin `<banco>.ejemplos.jsonl`) contesta zero-shot; un banco vacío escala siempre (`sin_oro`).
- El codificador carga una copia del e5 (≈ 470 MB) por sesión; `knn.e5_sesiones` (por defecto min(4, núcleos)). Con Qwen, ver arriba (≈ 2,4 GB por sesión).

## Estabilidad por tipo
- **`choice` y `noul` (sí/no): estables, con demostraciones por kNN (fs8, `<banco>.ejemplos.jsonl`).**
  - En la apertura única de las 8 tareas selladas de la batería v2 (R10, §17.26), el 4B con 8 demostraciones elegidas por kNN dio **0,614** de macro en choice, frente a 0,485 sin ejemplos: +12,9, IC por tareas [−1,6, +28,0] con 4 tareas.
  - En dev (D), las demostraciones por kNN ganaron +5,5 a las elegidas al azar.
- **`score`: EXPERIMENTAL.** La respuesta siempre lleva `experimental: true`.
  - La calibración (esperanza ordinal + umbrales isotónicos del banco) **solo ayuda con una rúbrica por nivel escrita por el cliente**, nunca con una mecánica:
    - **Dev (T3b), rúbrica escrita a partir del soporte:** +15,1 de exactitud macro y +0,24 de QWK frente al zero-shot (jql 0,149 → 0,478; usajobs 0,146 → 0,222; horeca 0,506 → 0,556).
    - **Selladas (§17.26), rúbrica mecánica** (descripción de la opción + primer ejemplo del soporte): **−2,7** frente al zero-shot, IC [−12,4, +4,4]. En disco pierde 16,4 puntos, y con 4 niveles dos umbrales quedan casi pegados.
  - Cómo se usa:
    - La rúbrica va como descripción de cada opción, con un ejemplo corto.
    - Los umbrales se ajustan con `cmd/ajustar-umbrales`, a partir del oro del soporte y de las predicciones de L3 sobre él (mediana de la esperanza por nivel, PAV y puntos medios). Exige ≥ 8 de oro por nivel y deja fuera las filas de ejemplo de la rúbrica.
    - Config: `knn.tareas.<t>.score_umbrales`. Sin la clave, se usan si existe `<banco>.umbrales.json`; con `true` se exigen; con `false` se ignoran.
  - Paridad Go = Python: el nivel coincide al 100 % en las 1 469 filas de test de T3b; los umbrales son idénticos y la esperanza difiere menos de 4e-15.
  - Sin rúbrica ni calibración, la exactitud en dev está cerca del azar (T3: jql y usajobs, 0,15 las dos).
- **`multi`: extensión solo de Go.** Sí/no independiente por opción (una llamada por opción con etiquetas S/N); `llm_logprobs.py` no lo tiene. Sin medir.

## Pendiente para producción
- Medir el e2e fs8-kNN por el servidor con vLLM real, Qwen3-4B y banco Qwen con el pool de `ejemplos.jsonl` frente a Python: ud preparado y NO lanzado (`model/alt/aws/empaquetar_w1_fs8.sh`, criterio propuesto `model/alt/crit-w1-fs8-e2e.json`, sin congelar; 100 filas por tarea: el servidor codifica cada consulta con Qwen en CPU).
- Medir fs8 con vLLM real (exactitud y latencia: los 8 turnos de ejemplo alargan el prompt; con 60 opciones y prefijo común, `--enable-prefix-caching` debería amortizarlo, sin medir).
- Paridad fila a fila del top-k en Go con los vectores e5 reales de finanzas, que hoy solo está probado con datos sintéticos.
- Calibrar u por cliente (hoy a mano); automatizarlo con un jsonl de validación.
- Construir y probar la imagen `Dockerfile.producto` (aquí no hay demonio de Docker); firmar/versionar `e5.onnx` y los bancos; quitar `-pruebas` del arranque de cualquier ud (forzar_l3 es solo de pruebas).
- Quedan sin cubrir: `score` (experimental; solo calibrable con rúbrica escrita por el cliente), `multi` (sin medir), tokenizador de Go frente a HF con espacios finales (ya compensado en el codificador, `runtime/tok` sin tocar), autenticación por cliente en los bancos
  (hoy un banco por tarea, compartido por todas las claves), y el cuello del codificador (una consulta por llamada; sin agrupar en lotes).
