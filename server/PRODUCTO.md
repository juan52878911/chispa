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
    **Limitación**: la herramienta no escribe `<banco>.ejemplos.jsonl` (los ejemplos fs8 de L3, que hoy salen de `generar_banco.py`, solo e5): sin ese fichero L3 contesta zero-shot (el servidor avisa). La medida de Qwen es solo L2 (L3 apagado).
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
- Medir fs8 con vLLM real (exactitud y latencia: los 8 turnos de ejemplo alargan el prompt; con 60 opciones y prefijo común, `--enable-prefix-caching` debería amortizarlo, sin medir).
- Paridad fila a fila del top-k en Go con los vectores e5 reales de finanzas, que hoy solo está probado con datos sintéticos.
- Calibrar u por cliente (hoy a mano); automatizarlo con un jsonl de validación.
- Construir y probar la imagen `Dockerfile.producto` (aquí no hay demonio de Docker); firmar/versionar `e5.onnx` y los bancos; quitar `-pruebas` del arranque de cualquier ud (forzar_l3 es solo de pruebas).
- Quedan sin cubrir: `score` (experimental; solo calibrable con rúbrica escrita por el cliente), `multi` (sin medir), tokenizador de Go frente a HF con espacios finales (ya compensado en el codificador, `runtime/tok` sin tocar), autenticación por cliente en los bancos
  (hoy un banco por tarea, compartido por todas las claves), y el cuello del codificador (una consulta por llamada; sin agrupar en lotes).
