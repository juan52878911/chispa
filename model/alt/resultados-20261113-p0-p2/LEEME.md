# P0 y P2 de la revisión de arquitectura (2026-11-13): PARCIAL

**Estado: parcial.** Juan pidió liberar el PC de pruebas antes de acabar. Está medido: batería no vista, L2 (M1), cross-encoder del hito 4 (ONNX fp32 e int8), kNN-e5, McNemar y latencia con el PC cargado. **Falta** el techo de Qwen3-4B en JevBench (67 de 231 casos), el cross-encoder en PyTorch fp32 y la latencia con la máquina libre. Lo pendiente está al final, con lo que necesita.

Cómputo: «el PC de pruebas» (i7-1255U, 12 hilos, sin GPU, ~7 GB) y una microVM de kindling para el LLM. Cero USD. Rama `arquitectura-p0-p2`.

## 0. Hallazgo que condiciona todo: «no vista» no es lo que dijo la revisión

`massive`, `inyecciones` (deepset) y `tweet_sentiment` **sí están en el train de tsjev** que usan los dos modelos (`manifiesto-entrenamiento.json`, tareas `multilingual/massive`, `prompt-injections`, `multilingual/tweet_sentiment_multilingual`; en el PC hay 750, 606 y 452 filas de train+dev). Lo que no entró son los splits de test que se usan aquí. Comprobado fila a fila contra el texto de tsjev (train y dev): solape de 3 filas en massive-es y 2 en massive-esc (descartadas), 0 en el resto. Así que solo `hatecheck-es` es una tarea y una fuente nuevas; el resto son «misma tarea, filas nuevas». Lo único que no pude comprobar: el meta de L2 (tsinst, synzs, pawsx) no está en el PC.

## 1. Batería (6 651 filas de evaluación, 92,2 % en español; sin clinc150)

Fuente, revisión y licencia en `bateria-manifiesto.json`. Se construye con `python -m model.alt.bateria_no_vista` (tope 2 000 filas por tarea, semilla 0).

| conjunto | n | idioma | fuente @ revisión | licencia | tarea en tsjev |
|---|---|---|---|---|---|
| hatecheck-es | 2000 | es | Paul/hatecheck-spanish @ a7ea759535 | CC-BY-4.0 | no |
| massive-es (60 intenciones) | 1503 | es | mteb/amazon_massive_intent @ 940fd47a81 | Apache-2.0 | sí (muestra) |
| massive-esc (18 escenarios; enunciados disjuntos de massive-es) | 1438 | es | mteb/amazon_massive_scenario @ 58871793b9 | Apache-2.0 | sí (muestra) |
| tuits-es (sentimiento, test+validación) | 1194 | es | mteb/tweet_sentiment_multilingual @ bceb37eb62 | CC-BY-3.0 | sí (muestra) |
| jailbreak | 400 | en | jackhhao/jailbreak-classification @ 2f2ceeb396 | Apache-2.0 | familia vista (prompt-injection) |
| inyecciones (deepset, test) | 116 | en/de | deepset/prompt-injections @ 4f61ecb038 | Apache-2.0 | sí (el train) |

Descartada: CataloniaTweet (la tarjeta no da el significado de las etiquetas 0/1/2) y las de licencia desconocida.

## 2. Resultados: exactitud con IC95 (bootstrap de 2 000 remuestreos, semilla 0)

| conjunto | n | L2 (M1) | cross-encoder fp32 (ONNX) | cross-encoder int8 | kNN-e5, 8 por clase |
|---|---|---|---|---|---|
| hatecheck-es | 2000 | 0,514 [0,493–0,536] | **0,616** [0,594–0,637] | 0,498 [0,475–0,520] | **0,693** [0,672–0,713] |
| massive-es | 1503 | 0,411 [0,386–0,437] | 0,454 [0,429–0,478] | 0,394 [0,369–0,417] | **0,500** [0,474–0,526] |
| massive-esc | 1438 | 0,563 [0,538–0,588] | **0,622** [0,596–0,647] | 0,574 [0,549–0,600] | 0,570 [0,543–0,595] |
| tuits-es | 1194 | 0,554 [0,527–0,580] | 0,554 [0,526–0,583] | 0,472 [0,443–0,498] | 0,482 [0,453–0,511] |
| jailbreak | 400 | 0,653 [0,608–0,700] | 0,665 [0,618–0,708] | 0,628 [0,580–0,673] | 0,565 [0,518–0,613] |
| inyecciones | 116 | 0,491 [0,397–0,578] | 0,707 [0,621–0,793] | 0,647 [0,552–0,733] | 0,767 [0,690–0,845] |
| **agrupada** | 6651 | 0,516 [0,504–0,528] | 0,574 [0,562–0,586] | 0,497 [0,484–0,508] | 0,578 [0,566–0,590] |
| **media de las 6 tareas** | | 0,531 | 0,603 | 0,535 | 0,596 |
| macro-F1 medio de las 6 | | 0,448 | 0,596 | 0,527 | 0,573 |
| JevBench | 231 | 0,498 [0,433–0,563] | 0,511 [0,446–0,576] | 0,446 [0,381–0,507] | no medido |

Comprobaciones de validez: L2 reproduce al decimal las predicciones de M1 en JevBench (231 de 231 en el top-1, diferencia máxima 2e-6); el cross-encoder ONNX fp32 da 0,5108 en JevBench, igual que el informe oficial de PyTorch (`g5-2058ffa`).

**kNN-e5 con 8 ejemplos por clase iguala al cross-encoder entrenado en la agrupada (0,578 frente a 0,574) y supera a L2 en 6,2 puntos.** Soporte: 8 por clase del split train de cada fuente (en hatecheck, apartados del mismo fichero y fuera de la evaluación), 32 vecinos, temperatura 0,05, vectores de e5 con media de tokens (`linea_base_knn_e5`). No es zero-shot: usa 2 a 472 ejemplos por tarea.

## 3. McNemar pareado, caso a caso (prueba exacta bilateral; diferencia con IC95 pareado)

| comparación | JevBench (n=231) | batería agrupada (n=6651) | por tarea |
|---|---|---|---|
| L2 → cross-encoder fp32 | **+1,3 pts [−7,4; +9,5], p = 0,84** (45 solo L2, 48 solo CE): sin diferencia | **+5,8 pts [+4,3; +7,2], p < 0,0001** (960 solo L2, 1 343 solo CE) | gana en hatecheck (+10,2), massive-es (+4,3), massive-esc (+5,9) e inyecciones (+21,6), todas con p < 0,01; empata en jailbreak y tuits |
| L2 → cross-encoder int8 | +5,2 a favor de L2 [−3,0; +13,4], p = 0,25 | +2,0 a favor de L2, p = 0,005 | L2 mejor solo en tuits (+8,2) |
| L2 → kNN-e5 | no medido | kNN +6,2 pts [+4,6; +7,8], p < 0,0001 | kNN gana en hatecheck, massive-es e inyecciones; L2 en tuits |

Veredicto: en JevBench no hay diferencia entre L2 y el cross-encoder (regla de la revisión: «p > 0,05 se anota sin diferencia»); en la batería el cross-encoder gana con claridad. La revisión acertó en la dirección (el hito 4 sí es mejor en lo no visto) y el veredicto «no pasó» del hito se explica por la vara, no por el modelo.

## 4. Techo del LLM en JevBench (Qwen3-4B-Instruct-2507 Q4_K_M, la plantilla `von-qwen3-4b`)

**No concluido.** Se usó el prompt, la entrada y el intérprete de `server/decide/l3.go` (`llm_jevbench.py`) contra una réplica en kindling (4 vCPU). Se pararon con 67 de 231 casos (los primeros: intent, fact, extraction, tool_selection, 8 de long_policy…), por orden de Juan. Sobre esos 67:

- Lectura estricta de L3 (respuesta inválida = fallo): 0,343 [0,239–0,463]. De los 67, 41 fueron inválidos: 15 `timeout` (120–300 s, casos largos), 1 HTTP 4xx y el resto JSON parcial (el modelo devuelve p. ej. `{"probs":{"cancel_order":1.0}}` con un solo id, que `interpretarL3` rechaza).
- Lectura permisiva (probabilidades parciales renormalizadas, ids ausentes = 0): 0,716 [0,612–0,821], con 16 inválidos.
- Sobre esos mismos 67: L2 0,687 [0,582–0,806]; cross-encoder fp32 0,702 [0,597–0,806]; int8 0,567.

No sirve para el criterio (< 0,60 en los 231): faltan los casos difíciles y los largos. Lo que sí dice: con el L3 tal cual está, **más de la mitad de las respuestas son inválidas** y L3 escala sin contestar; eso es un problema de formato del prompt y de contexto del servidor, no de capacidad.

**Decisión sobre JevBench (provisional).** Con n = 231 el IC95 es ±6,4 y el McNemar no separa ni siquiera a L2 del cross-encoder; 108 de 231 casos pasan de 128 tokens y el banco está en inglés. Recomiendo tratar JevBench como prueba de humo y usar esta batería como vara. El disparador formal (< 0,60 del LLM) **no se ha verificado**; si el techo completo sale ≥ 0,60, JevBench seguiría siendo un criterio de paso válido pero ruidoso.

## 5. P2: ¿cabe el cross-encoder en 60 ms?

Exportado a ONNX (opset 17, `onnx_ce.py`) y a int8 dinámico (`quantize_dynamic`, pesos QInt8). Una petición de N opciones = **una pasada por opción, las N en un solo lote** de N secuencias de 128 tokens. ORT con 4 hilos (`taskset` a 4 hilos lógicos), 100 repeticiones tras 10 de calentamiento. **El PC tenía otra carga (load 4 a 8) durante la medida**, así que son cifras pesimistas.

| opciones | fp32 p50 / p99 (ms) | int8 p50 / p99 (ms) |
|---|---|---|
| 2 | 75,7 / 265,8 | 48,2 / 301,0 |
| 4 | 162,4 / 456,3 | 92,5 / 110,4 |
| 8 | 342,6 / 855,8 | 196,3 / 210,3 |
| 16 | 743,6 / 1816,0 | 414,8 / 468,5 |

Tamaño: 470 MB fp32, 118 MB int8. Diferencia máxima de puntuación frente a PyTorch: fp32 1,8e-6; int8 0,67 (en ids aleatorios).

- **Latencia: NO PASA** (p50 con 4 opciones: 92,5 ms, criterio ≤ 60 ms). Pasaría con 2 opciones (48 ms). Con la máquina libre la cifra bajaría, pero el coste es lineal en las opciones, así que 4 opciones quedan en el borde en el mejor caso.
- **Pérdida del int8: NO PASA.** −7,7 puntos en la batería agrupada (0,574 → 0,497, IC95 pareado [6,7; 8,8], p < 0,0001) y −6,5 en JevBench; hatecheck cae 11,8. Criterio: ≤ 1 punto. El int8 dinámico estropea la cabeza cruzada; haría falta cuantización estática, dejar en fp32 las capas finales o entrenar con conciencia de cuantización.
- El criterio de la revisión dice que esto no invalida P3 (una sola secuencia). Medida indicativa de P3, con pesos sin entrenar y el PC cargado (latencia no depende de los pesos): L2x int8 con la secuencia entera de 128 tokens, p50 ≈ 48 / 50 ms con 2 / 4 opciones.

## 6. Pendiente de P0/P2 y qué necesita

| qué | dónde | tiempo estimado | entradas |
|---|---|---|---|
| Techo del LLM en los 231 casos (retomar con `--timeout 600`; salta los ids hechos) | GPU g5 o CPU de ≥ 8 vCPU | ~25 min en GPU, 1–2 h en CPU de 8 vCPU | `jevbench_filas.jsonl` (231 filas del esquema, sale de `cargar_jevbench` sobre el clon `fstandhartinger/jevbench`); modelo `qwen3-4b-instruct-2507` Q4_K_M (o Qwen/Qwen3-4B-Instruct-2507 en HF); parcial `l3_jevbench_parcial.jsonl` (67 filas); `server` con contexto ≥ 8 k para los casos largos |
| Cross-encoder en PyTorch fp32 (solo contraste con ONNX fp32) | CPU | ~45–60 min | checkpoint `alt-pruebas/ck/mejor` (470 MB) |
| Latencia con la máquina libre (fp32 e int8) | CPU de 4 hilos, sin otra carga | ~10 min | `onnx/ce_{fp32,int8}.onnx` (se regeneran con `python -m model.alt.onnx_ce exportar`) |
| Informe | cualquiera (solo numpy) | segundos | `python -m model.alt.p0_medir informe …` con las predicciones |

**Dónde están las entradas:** el checkpoint del hito 4 (`mejor`, sha256 `70df6ae0…53e`) **solo está en el PC de pruebas** (`~/orquestador/alt-pruebas/ck/mejor`, y su `mejor-e5-small.tgz` de 387 MB); no lo busqué en S3 porque estaba en local. M1 (`datos/m1.pt`, sha256 `4336e971…aa9c`) tampoco consta en S3 con una clave que yo conozca (el lote anterior lo bajaba de `s3://chispa-pruebas-cuenta-aws/ajuste-80ab8ef/ckpt/mejor.pt`; no verificado). Las predicciones de este informe están en el PC (`~/orquestador/fable/arq/res/pred`) y una copia en el scratchpad de la sesión; no se versionan. e5: `intfloat/multilingual-e5-small` @ 614241f622f53c4eeff9890bdc4f31cfecc418b3.

## 7. Comandos exactos

    # PC de pruebas: PYTHONPATH = código + sdk/python + bibliotecas; HF_HOME con e5; HF_HUB_OFFLINE=1
    python -m model.alt.bateria_no_vista --hub HUB --tsjev datos/tsjev --salida arq/bateria
    python -m model.alt.onnx_ce exportar --ckpt alt-pruebas/ck/mejor --salida arq/onnx
    python -m model.alt.p0_medir pred --modelo l2|knn|ce-fp32-onnx|ce-int8-onnx|ce --bateria arq/bateria --jevbench jevbench_filas.jsonl --salida arq/res/pred --hilos 8 [--m1 m1.pt] [--ckpt CK] [--onnx F]
    taskset -c 0-3 python -m model.alt.onnx_ce latencia --onnx arq/onnx/ce_int8.onnx --hilos 4 --n 100 --salida lat_int8.json
    python -m model.alt.p0_medir informe --pred arq/res/pred --bateria arq/bateria --jevbench jevbench_filas.jsonl --salida informe.json --pares l2:ce-fp32-onnx l2:ce-int8-onnx ce-fp32-onnx:ce-int8-onnx l2:knn
    # LLM, en el anfitrión de kindling (réplica de la plantilla von-qwen3-4b, sin red):
    python3 llm_jevbench.py --filas jev_filas.jsonl --url http://IP:8000 --salida l3_jevbench.jsonl --timeout 120

Pruebas: `python3 -m unittest discover -s model/alt -t .` (50, todas OK en el PC).

## 8. P3 y techo del LLM: preparados, sin lanzar (código 0be529b)

Criterio actualizado por Juan: la vara de P3 es la batería de §1 (agrupada y por tarea) y `hatecheck-es` se informa aparte como la única tarea de verdad no vista (§0); se mantienen los criterios de latencia y AUROC de la revisión. El paquete lo calcula solo: `python -m model.alt.p0_medir informe-l2x` escribe `informe-l2x.{json,md}` con exactitud (IC95) y macro-F1 de L2x, L2, cruzado del hito 4 y kNN-e5 por tarea y agrupados, el bloque de `hatecheck-es`, McNemar de L2x frente a L2 y al cruzado (agrupado y por tarea) y los criterios:
- C1 macro-F1 medio de las 6 tareas ≥ L2 + 12 pts; C2 ≥ cruzado − 1 pt; C3 exactitud agrupada ≥ cruzado − 1 pt;
- C4 AUROC de la probabilidad máxima (aciertos frente a fallos) ≥ 0,78 (no hay cabeza de escalada aprendida);
- C5 latencia p50 ≤ 60 ms en el i7 (int8, 4 hilos, ≤ 8 opciones): PENDIENTE, se mide con `bench_l2x_i7.py` sobre `onnx/l2x_int8.onnx`;
- refutado si queda por debajo del cruzado (−1 pt) en ≥ 3 de las 6 tareas.
Las referencias (L2, cruzado, kNN) van en `baseline-top1.json` dentro de `bateria-no-vista.tgz` (id → top-1 y probabilidad máxima; sin etiquetas de Jev como datos de entrenamiento).

### 8.1 Instancia L2x (g5.xlarge SPOT, ≤ 6 h)

- Código y entrada: `codigo.tgz` (sha256 `f1a9f41f3f87e2c6…`, 464 262 B, de `empaquetar_l2x.sh 0be529b`, con `sdk/python`, sin los informes) y `bateria-no-vista.tgz` (sha256 `d9887ba5f84c13a8…`, 701 909 B).
- user-data: `model/alt/aws/lanzados/ud-l2x-0be529b.sh` (11 835 B, sha256 `387d4e4d1ba9d1656d867291940b1c425b09ec2ce1fe1076dcbe0c195a08c359`).
- Claves (bucket `chispa-pruebas-cuenta-aws`, todas por nombre exacto): **lee** `l2x-0be529b/{codigo.tgz,bateria-no-vista.tgz}` (subidos), `meta-20261027/meta/tsjev/{train,dev}-*` (13 ficheros), `meta-20261027/meta/synzs/train-00000..00007.jsonl.gz` y `synzs/dev-00000.jsonl.gz`, `blandas/e2deac55256e3314/blandas.jsonl.gz`; **escribe** `l2x-0be529b/{log.txt,listo,resultados.tgz,ckpt/mejor-l2x.tgz,onnx/l2x_int8.onnx,onnx/exportacion.json}`; **espera** `l2x-0be529b/verificado` hasta 30 min. Comprobadas con `head-object`: las 29 claves de lectura existen y las 11 de escritura están libres.
- Cierre: tras subir `listo`, la instancia sondea cada 30 s la clave `verificado` (≤ 30 min) y luego apaga; `shutdown -h +360` es el tope duro (el vigilante cierra y sube 42 min antes) y la instancia se termina sola.
- Coste máximo: 6 h × 0,51 USD/h spot ≈ 3,1 USD; con `MaxPrice=0,60` ≤ 3,6 USD; EBS de 150 GB < 0,2 USD. No puede caer a bajo demanda.
- Comando (`describe-images`: dispositivo raíz `/dev/xvda`, AMI disponible; `describe-security-groups`: `sg-ejemplo` es el grupo `default` de `vpc-ejemplo`; `describe-subnets`: `subnet-ejemplo` es us-east-1f de la misma VPC, con IP pública; `g5.xlarge` ofrecida en us-east-1f). Con `--dry-run` la respuesta fue `DryRunOperation: Request would have succeeded, but DryRun flag is set.`

      aws ec2 run-instances --region us-east-1 --image-id ami-ejemplo --instance-type g5.xlarge --count 1 \
        --subnet-id subnet-ejemplo --security-group-ids sg-ejemplo --iam-instance-profile Name=rol-ec2-pruebas \
        --instance-initiated-shutdown-behavior terminate \
        --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=0.60,SpotInstanceType=one-time}' \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --user-data file://model/alt/aws/lanzados/ud-l2x-0be529b.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-l2x-0be529b}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-l2x-0be529b}]'

  Cuando `listo` aparezca en S3: bajar `resultados.tgz`, verificarlo y `aws s3 cp <fichero con cualquier contenido> s3://chispa-pruebas-cuenta-aws/l2x-0be529b/verificado`.

### 8.2 Instancia LLM (g5.xlarge BAJO DEMANDA, ≤ 60 min)

Instancia aparte, bajo demanda: dura menos de una hora (spot no ahorra casi nada y una interrupción obligaría a repetir la instalación de vLLM), y mezclarla con el entrenamiento arriesgaría la ronda de 6 h con contexto de 16 k y otro runtime en la misma A10G.

- **Arreglo de las respuestas inválidas.** El primer intento (modelo libre, `response_format: json_object`) dio 41 inválidas de 67 (61 %): 16 sin respuesta (15 `timeout`, 1 HTTP 4xx) y 25 con un JSON parcial como `{"probs":{"weather":1.0}}`, que el intérprete de L3 rechaza. Ahora el cliente (`llm_jevbench.py --formato`) manda un esquema JSON con **todos los ids obligatorios** (número entre 0 y 1, sin ids ajenos), contexto 16 k, timeout 600 s y **un reintento** con la suma de la respuesta anterior como retroalimentación si no suma 1 ± 0,05.
- **Prueba en el i7** (CPU, misma plantilla: Qwen3-4B-Instruct-2507 Q4_K_M, `unsloth/Qwen3-4B-Instruct-2507-GGUF` @ a06e946bb6b655725eafa393f4a9745d460374c9, sha256 `3605803b…c67e597`, servidor `llama-cpp-python` 0.3.36 con el esquema como gramática, contexto 4 096), 12 casos cortos (10 de los inválidos de antes y 2 válidos):

  | | inválidas (lectura estricta) |
  |---|---|
  | libre, mismo servidor del i7 | 10 de 12 |
  | restringida, primer intento | 4 de 12 (todas por sumar 1,1) |
  | restringida con el reintento con retroalimentación | **1 de 12** |
  | restringida, lectura permisiva (renormalizar) | 0 de 12 |

  Es una muestra pequeña y elegida entre las inválidas: demuestra que el esquema elimina los JSON parciales y que las sumas mal hechas quedan en lo que corrige el reintento, no mide la tasa final. **Sin verificar:** vLLM 0.33.0 con `response_format: json_schema` en la A10G (el i7 solo pudo probar la gramática de llama.cpp). El script lo comprueba antes de gastar nada: autoprueba de 3 casos y FALLO si menos de 2 son válidas.
- Runtime y modelo fijados: vLLM 0.33.0 (la versión de la ronda de destilación del 3 de noviembre en esta AMI), `Qwen/Qwen3-4B-Instruct-2507` @ cdbee75f17c01a7cc42f958dc650907174af0554 (Apache-2.0), bf16, `--max-model-len 16384`. Es bf16, no el Q4_K_M de L3: el techo sale algo más alto que el del servidor de producción.
- Se recalculan los 231 casos con el mismo método (no se saltan los 67 del primer intento: 41 eran inválidos por el formato libre y mezclar métodos falsearía el techo). Las 26 válidas del primer intento se usan solo para contar cuántas inválidas pasan a válidas. Salida: `llm_techo.json` con exactitud estricta y permisiva (IC95 de 2 000 remuestreos), tasa de inválidas, errores, la referencia de L2 y del cruzado en los mismos casos y el criterio `< 0,60`.
- Paquete: `llm-0be529b/llm.tgz` (sha256 `12afc145157cf132…`), `jev_filas.jsonl` (231 casos, sha256 `f06caab0564b6c2d…`), `l3_jevbench_parcial.jsonl` (67 filas), `baseline-top1.json`; user-data `model/alt/aws/lanzados/ud-llm-0be529b.sh` (7 135 B, sha256 `6d3c7deb1ce1fddeb9da7d56778b98b31f8909bfd5353e523a8cf384a452031b`).
- Claves: **lee** `llm-0be529b/{llm.tgz,jev_filas.jsonl,l3_jevbench_parcial.jsonl,baseline-top1.json}` (subidos y comprobados); **escribe** `llm-0be529b/{log.txt,listo,resultados.tgz}`; **espera** `llm-0be529b/verificado` hasta 8 min. Modelo desde Hugging Face (no S3).
- Coste máximo: `shutdown -h +60` → 1,006 USD (+ EBS de 100 GB < 0,02 USD). El vigilante cierra y sube 14 min antes del tope.
- Comando (`--dry-run`: `DryRunOperation: Request would have succeeded`):

      aws ec2 run-instances --region us-east-1 --image-id ami-ejemplo --instance-type g5.xlarge --count 1 \
        --subnet-id subnet-ejemplo --security-group-ids sg-ejemplo --iam-instance-profile Name=rol-ec2-pruebas \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --user-data file://model/alt/aws/lanzados/ud-llm-0be529b.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-0be529b}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-0be529b}]'

### 8.3 Qué demostró la prueba en seco

En el i7 (`maquina-i7`, directorio nuevo, bajo `systemd-run --scope -p MemoryMax=3500M -p AllowedCPUs=0-5 nice`, CPU, `/root/chispa-audio/venv` con torch 2.5.1 y transformers 4.46.3; sin tocar VM ni CT) se desempaquetaron los **tarballs exactos** que están en S3 (sha256 de `codigo.tgz` `f1a9f41f3f87e2c6…` y de `llm.tgz` `12afc145157cf132…`), se sustituyó `__CHISPA_S3__` y se corrieron los dos user-data con `CHISPA_LOCAL=1`:
- l2x: `[codigo] OK imports` (con `chispa` de `sdk/python`), `[entrenar] OK` (2 pasos con e5-small real), `[evaluar] OK`, `[informe] OK 4 criterios` (produce `informe-l2x.md` con hatecheck-es aparte, McNemar y C1–C5), `[onnx] OK` (fp32 472 MB, int8 118 MB) y subida simulada de `ckpt/mejor-l2x.tgz`, `onnx/l2x_int8.onnx`, `onnx/exportacion.json`, `resultados.tgz` y `log.txt`.
- llm: entradas por clave exacta, `[techo] OK` (231 casos, 12 con respuesta de juguete) y subida simulada.
Un primer intento falló por el entorno del i7 (la caché de e5 no tenía `refs/main`); se arregló con una caché con enlaces en `/root/p3-prueba/hfcache`, sin tocar el script.
**No probado:** el entrenamiento real en la A10G (rendimiento y 3 épocas), `pip install vllm==0.33.0` y la salida estructurada de vLLM, la descarga de HF desde la instancia, y las versiones de PyTorch y transformers de la AMI (las del i7 son otras). La exportación a ONNX en la AMI es opcional: si falla queda un AVISO y se rehace desde el checkpoint.

## 9. Resultado del techo del LLM (EC2, 2026-11-13)

Primera ronda, inst-12 con el código 0be529b: vLLM 0.33.0 no arrancó. Durante el profile_run, el muestreador de FlashInfer se compila con JIT y pide nvcc, y la DLAMI no lo trae. Costó ≈ 0,32 USD.
Arreglo en 5eb239c: `VLLM_USE_FLASHINFER_SAMPLER=0`, y la espera se corta si vLLM muere.
Segunda ronda, inst-17 con el prefijo llm-5eb239c: 672 s en total y los 231 casos en 206 s. Agregado en `llm-techo-5eb239c.json`.

| | Exactitud (IC95) |
|---|---|
| Qwen3-4B-Instruct-2507 bf16, salida restringida, lectura estricta | **65,8 %** [59,7, 71,9] |
| Lo mismo, lectura permisiva | 66,2 % [60,2, 72,3] |
| L2 (referencia) | 49,8 % [43,3, 56,3] |
| CE fp32 ONNX (referencia) | 51,1 % [45,0, 57,6] |

- Inválidas en lectura estricta: 2 de 231 (0,9 %), frente al 61 % de la v1 libre. De los 67 casos comunes, había 41 inválidos y ahora queda 1.
- Reintentos: 223 casos con 1 intento y 8 con 2. Latencia por caso en la A10G: p50 795 ms, p95 1 725 ms.
- El techo es ≥ 0,60 en las dos lecturas, así que JevBench sigue siendo criterio de paso para los codificadores. El LLM queda unos 15 puntos por encima de L2 y del CE.

## 10. Resultado de P3: L2x refutado (2026-11-13)

**Entrenamiento:** en EC2, inst-23 con 0be529b, en una g5.xlarge bajo demanda porque la cuenta no tiene cuota de spot G. Tres épocas: exactitud en dev 0,606, 0,605 y 0,633. Costó ≈ 2,2 USD.
En esa ronda fallaron la evaluación y el informe: el `bateria-no-vista.tgz` hecho en el Mac llevaba entradas AppleDouble `._*.jsonl`, que GNU tar extrae.

**Arreglo** (9ce491c): `cargar_bateria` ignora los ficheros que empiezan por «.», `l2x.sh` borra `._*`, transformers queda fijado a 5.22.2 (la última en PyPI a la hora del entrenamiento) y hay un modo «solo evaluar».

**Evaluación:** la del checkpoint de 0be529b se hizo en el i7 por CPU (torch 2.5.1, transformers 4.46.3, fp32, 6 hilos): 6 882 filas en 1 445 s. Informe completo en `informe-l2x.md` y `informe-l2x.json`.

| Agrupada (6 651 filas) | Exactitud (IC95) | Macro-F1 |
|---|---|---|
| L2x | 0,529 [0,517, 0,541] | 0,495 |
| L2 | 0,516 [0,504, 0,528] | 0,448 |
| Cruzado del hito 4 | 0,574 [0,562, 0,586] | 0,596 |
| kNN-e5 | 0,578 [0,566, 0,590] | 0,573 |

Criterios:
- **C1–C4 no pasan.**
  - C1, macro-F1 frente a L2 + 12: 0,495 contra 0,568.
  - C2, macro-F1 frente al cruzado − 1: 0,495 contra 0,586.
  - C3, exactitud agrupada frente al cruzado − 1: 0,529 contra 0,564.
  - C4, AUROC de la confianza: 0,68 contra 0,78.
- **Refutado:** queda por debajo del cruzado en 5 de 6 tareas (hatecheck-es, inyecciones, jailbreak, massive-esc y tuits-es). Solo gana en massive-es, por +3,2 puntos con p = 0,015.
- Frente a L2, la agrupada mejora +1,3 puntos [−0,1, +2,6] con p = 0,06, que no es significativo.
- hatecheck-es (la tarea no vista de verdad): 0,537, frente a 0,514 de L2 y 0,616 del cruzado.
- JevBench: 0,506, prácticamente igual que L2 y el cruzado y muy por debajo del techo del LLM (0,658).

**ONNX int8:** la diferencia de 0,33 de `exportacion.json` sale de comparar logits crudos en una sola fila de juguete, así que no mide decisiones.
Sobre 866 filas reales (150 por tarea, onnxruntime 1.20.1), el top-1 del int8 coincide con el de torch fp32 en el **85,3 %** (75 % en massive-es) y la diferencia máxima de probabilidad llega a 0,95. La exactitud media se mantiene (0,554 frente a 0,555), pero las decisiones cambian caso a caso. Este int8 dinámico no sirve tal cual: haría falta calibrar la cuantización o dejar en fp32 las capas de atención.

**C5, latencia en el i7** (int8, 4 hilos, CPUs 0-3, con el i7 ya cargado a 7–8 de load average), p50 / p99:

| Opciones | p50 | p99 |
|---|---|---|
| 2 | 21,8 ms | 32,9 ms |
| 4 | 26,9 ms | 41,9 ms |
| 8 | 32,6 ms | 55,5 ms |
| 16 | 44,7 ms | 69,7 ms |

C5 (≤ 60 ms con hasta 8 opciones) **pasa**, pero no rescata a L2x, que pierde en exactitud.

## 11. Paso 1: fusión kNN-e5 + L2 y escalado por acuerdo (2026-11-13, coste 0 USD)

**Veredicto frente al criterio congelado: REFUTADO** (diseño 1, log-lineal; la variante lineal, la única permitida, también). La fusión sí sube la exactitud
(agrupada 0.610 frente a 0,580 de kNN, ≥ 0,60), pero el AUROC del escalado en hatecheck-es es 0.616, por debajo de 0,65, y el agrupado 0.729 no llega a 0,78.
Además hatecheck-es queda 3.4 puntos por debajo del mejor de kNN y L2 (el criterio de PASA tolera 1).

Código: `model/alt/fusion_knn_l2.py` (test sin torch: `model/alt/tests/test_fusion_knn_l2.py`). Cifras: `fusion-knn-l2-loglineal.json` (diseño 1) y `fusion-knn-l2-lineal.json` (variante).

### 11.1 Qué se hizo
- Vectores de e5 (multilingual-e5-small congelado, media de tokens, «query: », 256 tokens) de todos los textos train y test, una sola vez; kNN reconstruido en numpy con 32 vecinos y temperatura 0,05, con k ejemplos por clase muestreados del split train (10 semillas por k).
- **El soporte de la batería tiene como mucho 8 ejemplos por clase**, así que k = 8, k = 16 y «todo el soporte» son el mismo kNN (IC entre semillas = 0). Los k reales son 1, 2, 4 y 8.
- Reproducción: con todo el soporte la exactitud agrupada es 0.580 (la guardada de P0, ≈ 0,578). Coinciden las 5 tareas salvo hatecheck-es (0,6985 frente a 0,6925; 94 de 2 000 top-1 difieren, probablemente desempates y redondeos de los vectores entre máquinas; causa no verificada).
- L2 y cruzado (`ce-fp32-onnx`) son las predicciones guardadas de P0; los ids cubren todo el test de las 6 tareas (la carga falla si no).
- Fusión log-lineal: p ∝ exp(w·log max(p_knn, 1e-3) + (1−w)·log max(p_l2, 1e-3)). w en rejilla 0..1 (paso 0,05), elegido por la exactitud media por tarea en las otras 5; k elegido igual (empate: el de más soporte). Salió «todo» en las seis tareas.
- Escalado: regresión logística ridge (λ=1, rasgos estandarizados, cada tarea pesa igual) con acuerdo top-1 kNN/L2, prob. máxima, margen, entropía normalizada, similitud al vecino más cercano y similitud al centroide de la clase predicha. Se ajusta en las otras 5 tareas y se aplica a la que queda. AUROC agrupado = las 6 predicciones fuera de tarea juntas.

### 11.2 Curva de k (exactitud con IC95 t entre 10 semillas; w y fusión por semilla, dejando una tarea fuera)

| k por clase | kNN agrupada | fusión agrupada | kNN / fusión hatecheck-es | kNN / fusión inyecciones | kNN / fusión jailbreak | kNN / fusión massive-es | kNN / fusión massive-esc | kNN / fusión tuits-es |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.425 ± 0.015 | 0.544 ± 0.016 | 0.546 / 0.543 | 0.677 / 0.627 | 0.521 / 0.748 | 0.325 / 0.447 | 0.339 / 0.578 | 0.394 / 0.552 |
| 2 | 0.472 ± 0.017 | 0.559 ± 0.015 | 0.562 / 0.561 | 0.713 / 0.658 | 0.571 / 0.759 | 0.395 / 0.471 | 0.441 / 0.590 | 0.398 / 0.551 |
| 4 | 0.521 ± 0.009 | 0.583 ± 0.011 | 0.632 / 0.614 | 0.720 / 0.692 | 0.564 / 0.722 | 0.450 / 0.496 | 0.499 / 0.601 | 0.417 / 0.562 |
| 8 | 0.580 ± 0.000 | 0.610 ± 0.000 | 0.699 / 0.664 | 0.767 / 0.793 | 0.565 / 0.642 | 0.500 / 0.531 | 0.570 / 0.629 | 0.482 / 0.569 |
| 16 | 0.580 ± 0.000 | 0.610 ± 0.000 | 0.699 / 0.664 | 0.767 / 0.793 | 0.565 / 0.642 | 0.500 / 0.531 | 0.570 / 0.629 | 0.482 / 0.569 |
| todo el soporte | 0.580 ± 0.000 | 0.610 ± 0.000 | 0.699 / 0.664 | 0.767 / 0.793 | 0.565 / 0.642 | 0.500 / 0.531 | 0.570 / 0.629 | 0.482 / 0.569 |

Mejor k de kNN solo: 8 (= todo el soporte), 0.580 agrupada. La fusión también es mejor con todo el soporte.

### 11.3 Por tarea, con todo el soporte (IC95 bootstrap sobre casos, 2 000 remuestreos, semilla 0; w = peso del kNN en la fusión)

| tarea | n | kNN | L2 | cruzado | fusión | w | fusión − mejor(kNN, L2) |
|---|---|---|---|---|---|---|---|
| hatecheck-es | 2000 | 0.699 [0.678, 0.719] | 0.514 [0.492, 0.535] | 0.616 [0.594, 0.636] | 0.664 [0.643, 0.684] | 0.55 | -0.034 |
| inyecciones | 116 | 0.767 [0.690, 0.845] | 0.491 [0.397, 0.578] | 0.707 [0.621, 0.793] | 0.793 [0.724, 0.862] | 0.55 | +0.026 |
| jailbreak | 400 | 0.565 [0.517, 0.613] | 0.652 [0.608, 0.700] | 0.665 [0.618, 0.708] | 0.642 [0.598, 0.690] | 0.8 | -0.010 |
| massive-es | 1503 | 0.500 [0.474, 0.526] | 0.411 [0.386, 0.436] | 0.454 [0.429, 0.478] | 0.531 [0.506, 0.556] | 0.55 | +0.031 |
| massive-esc | 1438 | 0.570 [0.543, 0.595] | 0.563 [0.538, 0.588] | 0.622 [0.596, 0.647] | 0.629 [0.606, 0.653] | 0.55 | +0.060 |
| tuits-es | 1194 | 0.482 [0.453, 0.511] | 0.554 [0.527, 0.580] | 0.554 [0.526, 0.583] | 0.569 [0.541, 0.596] | 0.55 | +0.015 |
| agrupada | 6651 | 0.580 [0.569, 0.592] | 0.516 [0.504, 0.528] | 0.574 [0.562, 0.586] | 0.610 [0.600, 0.622] | - | - |

### 11.4 AUROC del escalado frente a la probabilidad máxima de la fusión (dejando la tarea fuera)

| | escalado | prob. máxima |
|---|---|---|
| agrupado | 0.729 | 0.693 |
| hatecheck-es | 0.616 | 0.591 |
| inyecciones | 0.711 | 0.746 |
| jailbreak | 0.831 | 0.766 |
| massive-es | 0.825 | 0.783 |
| massive-esc | 0.836 | 0.809 |
| tuits-es | 0.611 | 0.609 |

El escalado mejora a la probabilidad máxima en 5 de 6 tareas (no en inyecciones), pero hatecheck-es (2 clases, kNN y L2 ruidosos) no se separa.

### 11.5 McNemar pareado exacto bilateral (solo_fusion = casos que acierta la fusión y no el otro; diferencia con IC95 pareado)

**fusión frente a kNN**

| tarea | solo fusión | solo otro | p | diferencia [IC95] |
|---|---|---|---|---|
| hatecheck-es | 292 | 360 | 0.008642 | -0.034 [-0.059, -0.009] |
| inyecciones | 17 | 14 | 0.7201 | +0.026 [-0.060, +0.121] |
| jailbreak | 31 | 0 | 0 | +0.077 [+0.052, +0.105] |
| massive-es | 194 | 148 | 0.01484 | +0.031 [+0.007, +0.055] |
| massive-esc | 200 | 114 | 1e-06 | +0.060 [+0.036, +0.084] |
| tuits-es | 264 | 160 | 1e-06 | +0.087 [+0.054, +0.120] |
| agrupada | 998 | 796 | 2e-06 | +0.030 [+0.018, +0.043] |

**fusión frente al cruzado**

| tarea | solo fusión | solo otro | p | diferencia [IC95] |
|---|---|---|---|---|
| hatecheck-es | 356 | 258 | 8.8e-05 | +0.049 [+0.024, +0.072] |
| inyecciones | 24 | 14 | 0.1433 | +0.086 [-0.017, +0.181] |
| jailbreak | 72 | 81 | 0.5179 | -0.022 [-0.080, +0.038] |
| massive-es | 314 | 198 | 0 | +0.077 [+0.049, +0.108] |
| massive-esc | 182 | 171 | 0.5946 | +0.008 [-0.019, +0.032] |
| tuits-es | 195 | 178 | 0.4074 | +0.014 [-0.018, +0.045] |
| agrupada | 1143 | 900 | 0 | +0.036 [+0.024, +0.050] |

### 11.6 Variante (iteración 2): mezcla lineal de probabilidades p = w·p_knn + (1−w)·p_l2

Misma selección de w y k, mismo escalado. Agrupada 0.612; peor tarea frente al mejor de kNN/L2 -0.034 (hatecheck-es); AUROC del escalado 0.717 agrupado y 0.619 en hatecheck-es (prob. máxima: 0.664 y 0.595). También REFUTADO. Se cierra el tope de 2 iteraciones.

### 11.7 Veredicto

| condición | valor | umbral |
|---|---|---|
| exactitud agrupada de la fusión | 0.610 | ≥ 0,60 (cumple; y supera a kNN 0,580) |
| peor tarea frente al mejor de kNN y L2 | -0.034 (hatecheck-es) | ≥ −0,010 (no cumple) |
| AUROC del escalado, agrupado | 0.729 | ≥ 0,78 (no cumple) |
| AUROC del escalado, hatecheck-es | 0.616 | ≥ 0,70 para PASA; < 0,65 es REFUTADO (se activa) |

REFUTADO por la regla del AUROC en hatecheck-es. Lectura: kNN y L2 sí fallan en sitios distintos (la fusión gana +3,0 puntos agrupados y en 5 de 6 tareas
no pierde frente al mejor), pero el acuerdo, el margen y la distancia al vecino no separan aciertos de fallos mucho mejor que la probabilidad máxima
(AUROC agrupado +0.036), y en hatecheck-es no lo hacen en absoluto. La fusión le cuesta a hatecheck-es 3,4 puntos frente al kNN solo, que ahí ya es
el mejor modelo (0,699); el peso w se ajusta con tareas muy distintas.

### 11.8 Comandos exactos y rutas

En el PC de pruebas están las predicciones de P0 (`res/pred/{l2,knn,ce-fp32-onnx}/<tarea>.jsonl`); se copiaron a `maquina-i7`. Todo lo grande queda fuera del repo, en
`/mnt/nfs-lab/chispa-audio/fusion-knn-l2/` (`vec/<tarea>.npz` = vectores de e5 en float32; `pred/` = predicciones copiadas de P0; `repo/` = copia del código).
```
# maquina-i7 (venv /root/chispa-audio/venv), cada comando bajo: systemd-run --scope -q -p MemoryMax=3500M -p AllowedCPUs=0-5 nice
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/root/p3-prueba/hfcache PYTHONPATH=$D/repo python -m model.alt.fusion_knn_l2 vectores \
    --bateria /mnt/nfs-lab/chispa-audio/p3-eval/full/bat --salida $D/vec        # ≈ 4 min
PYTHONPATH=$D/repo python -m model.alt.fusion_knn_l2 medir --bateria .../full/bat --pred $D/pred --vec $D/vec \
    --salida $D/fusion-loglineal.json --semillas 10 --fusion loglineal                # ≈ 25 s (la variante: --fusion lineal)
python -m unittest model.alt.tests.test_fusion_knn_l2                                  # sin torch
```
Los JSON de este directorio solo llevan cifras (ningún texto de la batería).

