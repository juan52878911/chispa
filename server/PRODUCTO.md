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
- El codificador carga una copia del e5 (≈ 470 MB) por sesión; `knn.e5_sesiones` (por defecto min(4, núcleos)).

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
