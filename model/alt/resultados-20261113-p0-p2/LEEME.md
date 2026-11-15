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


## 12. Paso 2: techo del LLM en la batería y el LLM como etiquetador (preparado, sin lanzar)

Aprobado por Juan con tope de 1,5 USD dentro de la partida de 35 y sin push. Código: `model/alt/llm_bateria.py` (muestra, fondo, medición; test `model/alt/tests/test_llm_bateria.py`)
y `model/alt/aws/llm_bateria.sh` (user-data). El cliente es `llm_jevbench.py` sin cambios: el método de llm-5eb239c (esquema con todos los ids obligatorios, contexto 16 k,
temperatura 0, un reintento con la suma anterior como retroalimentación).

**Criterio congelado.** PASA si la exactitud agrupada del LLM bf16 es ≥ 0,65, si en hatecheck-es saca ≥ 5 puntos sobre kNN (el mayor del guardado y el recalculado) y si el Q4_K_M
pierde ≤ 2 puntos frente al bf16 en las mismas filas. REFUTADO si la agrupada queda por debajo de la fusión del paso 1 en las mismas filas (0,610 en la batería completa).
También se mide el AUROC del escalado del LLM: la probabilidad máxima de la respuesta (una respuesta inválida cuenta como confianza 0) y la validez sola, por tarea y sobre todo en hatecheck-es.

### 12.1 Muestra (`llm_bateria muestra`, semilla 0)

Estratificada por etiqueta: 250 filas de test por tarea, hatecheck-es completa (2 000) e inyecciones completa (116). Total 3 116 filas.
La agrupada se pondera por el tamaño de cada tarea en la batería completa (2 000, 116, 400, 1 503, 1 438, 1 194), así que es comparable con el 0,610 de la fusión.
kNN, L2, el cruzado y la fusión se recalculan en las mismas filas (comparación pareada con McNemar e IC95 bootstrap dentro de cada tarea).
`filas-q4.jsonl` lleva las mismas 3 116 filas para el Q4: primero una submuestra anidada de 430 (hatecheck-es 150, massive-es 40, el resto 60) y luego las demás, intercaladas por tarea.
Si el Q4 se corta por tiempo, se mide solo en las filas que tienen respuesta.

### 12.2 Q4_K_M: en la g5, no en el i7

En el i7 (`maquina-i7`, i7-8700T con 6 hilos, llama.cpp b12138 para CPU, el GGUF de L3 con sha256 `3605803b…c67e597`, 2 ranuras) el Q4 tarda unos 65 s por fila en hatecheck-es,
unos 30 s en inyecciones y jailbreak y unos 6 min en massive-es. La prefill va a ~30 tokens/s y la generación a ~5 tokens/s. La submuestra de 430 filas pasaría de 8 h, así que no cabe.
En esa prueba 10 de 12 respuestas fueron válidas (las 2 inválidas, en massive-es y massive-esc, tras el reintento).
Se mide en la misma g5 con el binario oficial `llama-b12138-bin-ubuntu-cuda-12.8-x64.tar.gz` (sha256 `8aa6fd55…`) más su `cudart` (sha256 `1402566b…`), los dos comprobados en la instancia.
El GGUF es `unsloth/Qwen3-4B-Instruct-2507-GGUF` @ a06e946b; el sha256 se comprueba contra el de L3. La prueba usa 16 ranuras y 6 144 tokens por ranura.

### 12.3 Fondo sin etiqueta para el paso 3 (`llm_bateria fondo`)

Son textos de los splits train y dev de las mismas fuentes y revisiones. Ningún texto del fondo está en el test de ninguna tarea (texto normalizado). En hatecheck-es se descarta además
toda plantilla que tenga algún caso en el test. Las etiquetas de la fuente se quedan fuera de la instancia (`fondo-fuente.jsonl`, en el i7) y solo sirven para medir el acierto del etiquetador.

| tarea | candidatas | fuera por texto de test | fuera por plantilla de test | en el fondo |
|---|---|---|---|---|
| hatecheck-es | 3 655 | 2 000 | 1 475 | **180** |
| inyecciones | 546 | 0 | 0 | 546 |
| jailbreak | 1 598 | 12 | 0 | 1 572 |
| massive-es | 13 547 | 197 | 0 | 2 000 (de 12 965) |
| massive-esc | 13 547 | 197 | 0 | 2 000 (de 12 965) |
| tuits-es | 2 163 | 324 | 0 | 1 839 |

hatecheck-es no tiene fondo propio: quedan 180 casos de plantillas sin ningún caso en el test. Para el paso 3 haría falta otra fuente en español sin etiqueta.
Total: 8 137 textos, intercalados por tarea para que un corte por tiempo deje parte de cada tarea.

### 12.4 Instancia (g5.xlarge BAJO DEMANDA, ≤ 85 min, ≤ 1,43 USD)

Fases en orden de prioridad. Cada una tiene su tope y lo cortado queda escrito (el cliente es reanudable):

1. **A. bf16, vLLM 0.33.0:** las 3 116 filas, con 32 hilos, después de una autoprueba de 6 casos (uno por tarea, incluido massive-es con sus 60 opciones; FALLO si hay menos de 2/3 válidas).
2. **B. Q4_K_M, llama.cpp CUDA:** se para vLLM y se mide `filas-q4.jsonl` durante 22 min como mucho.
3. **C. Fondo:** vLLM otra vez, hasta que queden 15 min.

El vigilante cierra a los 71 min y sube los resultados; `shutdown -h +85` es el tope duro.
- Paquete en `llm-bateria-4994f3e/` (subido y vuelto a bajar con el mismo sha256):

  | fichero | sha256 |
  |---|---|
  | `llm.tgz` | `9cae2eeacf61b687…` (solo `llm_jevbench.py`, sin `._*`) |
  | `filas.jsonl` | `7ea97957…` |
  | `filas-q4.jsonl` | `757fcae0…` |
  | `fondo.jsonl` | `5bb146ef…` |

  user-data: `model/alt/aws/lanzados/ud-llm-bateria-4994f3e.sh` (13 853 B, sha256 `34ae1ab2862c744d0193bfae5895f7515e0f5cf24957e7c77ce62d8fa0828088`).
- Claves:
  - **Lee:** esos 4 ficheros.
  - **Escribe:** `log.txt`, `listo` y `resultados.tgz`.
  - **Espera:** `verificado`, hasta 8 min.

  Las 8 estaban libres (`head-object`). Los modelos vienen de Hugging Face y el binario de llama.cpp de GitHub, con el sha256 comprobado.
- **Prueba en seco en el i7** (`CHISPA_LOCAL=1`, una fila por fase, contra llama-server en CPU, con el user-data y los tarballs exactos): pasaron las entradas por clave exacta, la autoprueba, A, B, C, el resumen y la subida simulada, en 66 s.
  Después, `llm_bateria medir` sobre esas salidas recalcula las referencias en las filas de la muestra con la ponderación de la batería:

  | referencia | agrupada (muestra ponderada) | batería completa |
  |---|---|---|
  | kNN | 0,578 | 0,578 |
  | fusión | 0,606 | 0,610 |
  | cruzado | 0,572 | 0,574 |

  **No probado:** vLLM en esta ronda (sí en llm-5eb239c), el binario CUDA de llama.cpp en la A10G (que no le falte ninguna biblioteca) y el reinicio de vLLM tras el Q4. Si el Q4 falla, A y C siguen y queda un FALLO en el log.
- Orden (`--dry-run`: `DryRunOperation: Request would have succeeded`):

      aws ec2 run-instances --region us-east-1 --image-id ami-ejemplo --instance-type g5.xlarge --count 1 \
        --subnet-id subnet-ejemplo --security-group-ids sg-ejemplo --iam-instance-profile Name=rol-ec2-pruebas \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --user-data file://model/alt/aws/lanzados/ud-llm-bateria-4994f3e.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-bateria-4994f3e}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-bateria-4994f3e}]'

  Cuando aparezca `listo`, se baja `resultados.tgz` y se mide en el i7 con
  `python -m model.alt.llm_bateria medir --muestra $D/muestra --bf16 res/l3_bateria.jsonl --q4 res/l3_q4.jsonl --bateria …/p3-eval/full/bat --pred …/fusion-knn-l2/pred --vec …/fusion-knn-l2/vec`
  y con `fondo-resumen`. Después se sube `verificado`.
  Datos fuera del repo: `maquina-i7:/mnt/nfs-lab/chispa-audio/llm-bateria/` (`muestra/`, `fondo/` con `fondo-fuente.jsonl`, `hub/`, `llamacpp/`, `seco/`).

### 12.5 Resultado (EC2 inst-08, 2026-11-13): INCOMPLETO, falta el Q4; sin Q4 no pasaría (agrupada 0,6497 frente a 0,65)

Duró 3 468 s hasta `listo`; `verificado` se subió al momento y la instancia se terminó sola. Costó ≈ 1,0 USD (≈ 60 min a 1,006 USD/h, con el EBS de 100 GB aparte).
Cifras en `llm-bateria-4994f3e.json` (medición) y `llm-bateria-fondo-4994f3e.json` (fondo). Ninguno de los dos lleva textos.

| fase | resultado |
|---|---|
| A. bf16 (vLLM) | 3 116 de 3 116 filas en 350 s, 3 025 válidas (las 91 inválidas: 45 en massive-es y 46 en massive-esc) |
| B. Q4_K_M (llama.cpp CUDA) | **FALLO**: el binario oficial `ubuntu-cuda-12.8` pide `GLIBC_2.38` y la DLAMI AL2023 trae la 2.34. No se midió nada del Q4 |
| C. fondo | 8 137 de 8 137 textos en 2 501 s, 7 372 válidos |

**Criterio:**

| criterio | resultado | ¿se cumple? |
|---|---|---|
| agrupada ≥ 0,65 | **0,6497** [0,627, 0,673]; permisiva 0,662 | no, por 0,0003; el IC cubre el umbral |
| hatecheck-es ≥ kNN + 5 pts | 0,824 frente a 0,6985 (kNN recalculado) = **+12,6 pts** | sí |
| Q4 pierde ≤ 2 pts | sin medir | — |
| refutado si < fusión | 0,650 frente a 0,606 (fusión en las mismas filas) = +4,4 pts [+1,5, +7,1] | no está refutado |

#### Por tarea

Lectura estricta, IC95 bootstrap; las referencias son de las mismas filas.

| tarea | n | LLM bf16 | kNN | L2 | cruzado | fusión | LLM − fusión [IC95] | inválidas | AUROC pmax | AUROC validez |
|---|---|---|---|---|---|---|---|---|---|---|
| hatecheck-es | 2000 | **0,824** [0,807, 0,841] | 0,693 / 0,699 | 0,514 | 0,616 | 0,665 | +0,160 [+0,137, +0,183] | 0 | **0,517** | 0,5 |
| inyecciones | 116 | 0,569 [0,474, 0,655] | 0,767 | 0,491 | 0,707 | 0,793 | −0,224 [−0,328, −0,129] | 0 | 0,588 | 0,5 |
| jailbreak | 250 | **0,924** [0,892, 0,956] | 0,556 | 0,656 | 0,656 | 0,628 | +0,296 [+0,224, +0,368] | 0 | 0,672 | 0,5 |
| massive-es | 250 | 0,444 [0,384, 0,504] | 0,480 | 0,396 | 0,464 | 0,520 | −0,076 [−0,152, −0,004] | 45 | 0,732 | 0,662 |
| massive-esc | 250 | 0,564 [0,500, 0,624] | 0,580 | 0,548 | 0,628 | 0,616 | −0,052 [−0,116, +0,008] | 46 | 0,833 | 0,711 |
| tuits-es | 250 | 0,636 [0,576, 0,692] | 0,496 | 0,560 | 0,528 | 0,580 | +0,056 [−0,028, +0,140] | 0 | 0,681 | 0,5 |
| agrupada (ponderada) | 3116 | **0,650** [0,627, 0,673] | 0,578 | 0,511 | 0,572 | 0,606 | +0,044 [+0,015, +0,071] | 91 | 0,616 (mezcla) | 0,560 |

#### Lectura

- **El LLM gana mucho donde hay que entender el texto** (hatecheck-es, jailbreak, tuits-es) y pierde donde hay muchas opciones o un sesgo de la fuente:
  - En massive-es y massive-esc el 18 % de las respuestas no son válidas: con 18 a 60 ids obligatorios la suma se va de 1 ± 0,05 incluso tras el reintento. En lectura permisiva massive-esc sube a 0,608.
  - En inyecciones (deepset) se queda en 0,569: el etiquetado de deepset no coincide con lo que el LLM entiende por inyección.
- **Escalado:** el LLM casi siempre responde con probabilidad 0,9–1, así que la probabilidad máxima no separa aciertos de fallos.
  - hatecheck-es da AUROC 0,517, peor que el 0,616 de la fusión.
  - La validez solo informa en massive (0,66–0,71).
  - Con este formato el LLM **no da una señal de escalado útil**; haría falta otra (logprobs del token de la opción, autoconsistencia o acuerdo con kNN).
- **Fondo para el paso 3**, acierto frente a la etiqueta de la fuente en las respuestas válidas:

  | tarea | acierto | respuestas válidas con pmax ≥ 0,9 |
  |---|---|---|
  | jailbreak | 0,928 | 1 572 de 1 572 |
  | inyecciones | 0,718 | 545 de 546 |
  | massive-esc | 0,713 | 1 234 de 1 614 |
  | tuits-es | 0,637 | 728 de 1 839 |
  | hatecheck-es | 0,633 (180 textos) | 168 de 180 |
  | massive-es | 0,563 | 1 309 de 1 621 |

  En hatecheck-es el fondo acierta mucho menos que en el test (0,824): son plantillas distintas.

#### Q4 pendiente: qué haría falta

El binario CUDA de llama.cpp no carga en AL2023 por glibc. Hay dos salidas sin compilar:
- **(a)** llama.cpp en el contenedor oficial `ghcr.io/ggml-org/llama.cpp:server-cuda` con `--gpus all` (la DLAMI trae Docker y nvidia-container-toolkit);
- **(b)** el GGUF en vLLM 0.33.0 (experimental; no es el runtime de producción).

Una ronda solo de Q4 con el user-data de esta y las fases A y C desactivadas costaría ≈ 0,4 USD. La de esta ronda fue la primera falla.

### 12.6 Ronda solo de Q4: llama.cpp en el contenedor oficial CUDA (preparada, sin lanzar)

Juan eligió la opción (a), con tope de 0,6 USD. User-data: `model/alt/aws/llm_q4.sh` → `lanzados/ud-llm-q4-8deacf2.sh`
(7 796 B, sha256 `4bdc197bc0855d56184989b8da6ee9db69aa61c357c6b96148c79eaeb3afd727`).

- **Imagen fijada por digest:** `ghcr.io/ggml-org/llama.cpp@sha256:e110d1ae4a3572398950e7b64b5abd585a3eab678ae726484f743a12140bdc40`. Es el manifiesto linux/amd64 de
  `server-cuda12` (índice `sha256:cabbb32a…`), 2,62 GB comprimida, `org.opencontainers.image.version` **b12375**, commit 9d25e00, CUDA 12.8.1, creada el 2026-11-14.
  No hay imagen con versión para b12138: `server-cuda-b12138` y `server-cuda12-b12138` dan 404 en ghcr.io, y las etiquetas con versión de CUDA acaban en b11206.
  Por eso el runtime es b12375, no el b12138 de la prueba en el i7. El GGUF es el mismo; solo cambia la versión de llama.cpp.
- **Autoprueba antes del GGUF:** que existan `docker` y `docker info`, `docker pull` por digest, `docker run --rm --gpus all IMAGEN --version` y `--list-devices`, que debe listar `CUDA0`
  (comprobado en el código de b12375: `common_print_available_devices`). Si algo falla, FALLO y cierre en ≈ 3 min.
- **Lo mismo que la ronda bf16:**
  - el GGUF `unsloth/…-GGUF` @ a06e946b, por URL fija de HF y sha256 `3605803b…`;
  - `filas-q4.jsonl` (sha256 `757fcae0…`);
  - `llm.tgz` (sha256 `9cae2eea…`, el mismo cliente);
  - formato `json_schema` y un reintento;
  - `-np 16 -c 98304`, `-fa on`, `--jinja`, 16 hilos de cliente.

  Antes de las 3 116 filas hay una autoprueba de una fila por tarea.
- **Tiempo y coste:** `shutdown -h +33`; el cliente de las 3 116 filas no tiene tope propio: lo corta el vigilante cuando quedan 11 min (a los 22 min); el resto se mide en las filas con respuesta. Espera `verificado` hasta 6 min.
  Coste máximo ≈ 0,57 USD (con el arranque), más el EBS de 100 GB.
- **Claves en `llm-q4-8deacf2/`:**
  - **Lee:** `llm.tgz` y `filas-q4.jsonl`. Subidas y vueltas a bajar con el mismo sha256.
  - **Escribe:** `log.txt`, `listo` y `resultados.tgz`.
  - **Espera:** `verificado`.

  Las 6 estaban libres.
- **Prueba en seco en el i7** (modo local, una fila, llama-server b12138 en CPU, el ud exacto): pasaron las entradas, la autoprueba, Q4 y la subida simulada, en 36 s. `medir --q4` produce las inválidas y la
  diferencia por tarea y el McNemar agrupado. **No probado:** docker y el toolkit de NVIDIA en la DLAMI (no hay docker en el i7); por eso la autoprueba cierra en cuanto falla.
- **Criterio:** el Q4 pierde ≤ 2 puntos frente al bf16 en las mismas filas (agrupada ponderada, IC95 por pares y McNemar), y se miran las inválidas por tarea, sobre todo en massive.
- **Orden** (con `--dry-run` responde `DryRunOperation`):

      aws ec2 run-instances --region us-east-1 --image-id ami-ejemplo --instance-type g5.xlarge --count 1 \
        --subnet-id subnet-ejemplo --security-group-ids sg-ejemplo --iam-instance-profile Name=rol-ec2-pruebas \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --user-data file://model/alt/aws/lanzados/ud-llm-q4-8deacf2.sh \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-q4-8deacf2}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-q4-8deacf2}]'

### 12.7 Resultado de la ronda de Q4 (EC2 inst-13, 2026-11-15): el Q4 pierde 3,05 puntos y no cumple ≤ 2

Duró 999 s hasta `listo` y costó ≈ 0,31 USD (≈ 18 min). La autoprueba de docker y la GPU pasó: docker 25.0.14, la imagen en 50 s, `CUDA0: NVIDIA A10G`.
El runtime es **llama.cpp b12375** (commit 9d25e00), no el b12138 de la prueba en el i7.
Las 3 116 filas se hicieron en 824 s; 3 027 respuestas son válidas. Cifras en `llm-bateria-q4-8deacf2.json`.

| tarea | n | bf16 | Q4_K_M | Q4 − bf16 [IC95 por pares] | McNemar (solo Q4 / solo bf16, p) | inválidas bf16 → Q4 |
|---|---|---|---|---|---|---|
| hatecheck-es | 2000 | 0,824 | 0,8165 | −0,0075 [−0,014, −0,0005] | 19 / 34, p = 0,053 | 0 → 0 |
| inyecciones | 116 | 0,569 | 0,569 | 0 | 0 / 0 | 0 → 0 |
| jailbreak | 250 | 0,924 | 0,904 | −0,020 [−0,040, −0,004] | 0 / 5, p = 0,063 | 0 → 0 |
| massive-es | 250 | 0,444 | **0,344** | **−0,100** [−0,148, −0,052] | 8 / 33, p = 1e-4 | **45 → 54** |
| massive-esc | 250 | 0,564 | 0,580 | +0,016 [−0,028, +0,060] | 18 / 14, p = 0,60 | 46 → 35 |
| tuits-es | 250 | 0,636 | 0,592 | −0,044 [−0,084, −0,004] | 8 / 19, p = 0,052 | 0 → 0 |
| agrupada ponderada | 3116 | 0,6497 | **0,6192** | **−0,0305** [−0,0472, −0,0141] | 53 / 105, p = 4e-5 | 91 → 89 |

- **El criterio del Q4 no se cumple:** pierde 3,05 puntos y el IC entero queda por debajo de −1,4. La mayor parte viene de massive-es (−10 puntos con 60 opciones, y 9 inválidas más), con tuits-es y jailbreak detrás.
  En hatecheck-es el Q4 pierde menos de 1 punto y sigue 11,8 puntos por encima de kNN (0,8165 frente a 0,6985).
- El top-1 de Q4 y bf16 coincide en el 92,5 % de las filas.
- **Veredicto del paso 2 frente al criterio congelado: NO PASA, sin refutar.**

  | criterio | resultado |
  |---|---|
  | agrupada bf16 ≥ 0,65 | 0,6497: falla por 0,0003 |
  | hatecheck-es ≥ kNN + 5 puntos | +12,6 puntos: se cumple |
  | Q4 pierde ≤ 2 puntos | pierde 3,05: falla |
  | refutado si queda bajo la fusión | 0,650 frente a 0,606 de la fusión: no está refutado |

- **Lectura para la arquitectura:** el LLM en producción (Q4) es muy bueno en hatecheck-es y jailbreak, pero no sirve como clasificador de muchas opciones (massive). Tampoco da una señal de escalado útil (AUROC de pmax 0,517 en hatecheck-es).
  Encaja más como etiquetador o como juez de casos de pocas opciones que como cascada general.


## 13. Rondas R1, R2 y R3 hacia uso general (preparadas, sin lanzar; código 1208cd9)

Objetivo: que Chispa sirva para tareas que no ha visto (zero-shot y few-shot). Rama `llm-rondas-r123`, sin push. Gasto antes de estas rondas ≈ 4,1 de 35 USD.

### 13.1 R1, paso 3: kNN-e5 con soporte etiquetado por el LLM (c7i.2xlarge, ≤ 40 min, ≤ 0,4 USD)

**Criterio congelado antes de medir** (`model/alt/knn_soporte_llm.py`, `CRIT`; el user-data aborta si el código trae otro):
sha256 `5e0c38a679aed8f064cbc10e5df271075d2c5a0b22c8ad08ce8a99508aebafb9` de
`{"agrupada_min": 0.55, "hatecheck_min": 0.6, "perdida_max": 0.03, "referencia": "oro/todo", "refutado_si_agrupada_le": 0.516, "variante": "llm/todo"}`.
- Variante medida: kNN (32 vecinos, temperatura 0,05, como el paso 1) con soporte = los textos del fondo con respuesta válida de Qwen3-4B bf16 (7 372 de 8 137), etiqueta = top-1 del LLM, todo el soporte.
- Referencia: kNN con soporte de oro (split train de la batería, ≤ 8 por clase), todo el soporte, con los vectores de la misma máquina.
- **PASA** si pierde ≤ 3 puntos agrupados frente a la referencia, la agrupada da ≥ 0,55 y hatecheck-es ≥ 0,60. **REFUTADO** si la agrupada queda ≤ 0,516 (nivel de L2). Si no, NO PASA.
- Agrupada = aciertos / casos en todo el test de las 6 tareas (6 651 filas), la misma medida que el 0,580 del paso 1. Diferencia con IC95 pareado (bootstrap por tarea) y McNemar.

Curvas (descriptivas, no deciden): oro con k = 1, 2, 4, 8 por clase; LLM con k = 1, 2, 4, 8, 16, 32, 64, 128 por clase y todo (10 semillas); LLM solo con p ≥ 0,9;
oro ∪ LLM; y el mismo fondo con la etiqueta de su fuente (k = 8, 32, 128 y todo), que separa el efecto del tamaño del soporte del ruido del LLM. Cada punto lleva el
soporte medio por clase que hubo de verdad (hatecheck-es solo tiene 180 textos de fondo; massive-es, unos 33 por clase).

- Instancia: c7i.2xlarge bajo demanda (0,357 USD/h), la misma DLAMI (arranca sin GPU), 100 GB gp3, terminate al apagar, `shutdown -h +40`, espera de `verificado` 6 min.
- Pasos: transformers 5.22.2 → entradas por nombre exacto y recuentos → criterio (sha) → e5-small @ 614241f6 → vectores de batería y fondo → `knn_soporte_llm medir`.
- Entradas en `s3://chispa-pruebas-cuenta-aws/knn-llm-1208cd9/` (sha256): codigo.tgz `f04d66d5…`, bateria.tgz `b1b2ca54…`, vec-paso1.tgz `19d903fc…`,
  fondo.jsonl `5bb146ef…`, l3_fondo.jsonl `927e84d2…`, fondo-fuente.jsonl `9a4f7fd2…`. User-data: `lanzados/ud-knn-llm-1208cd9.sh`.
- **Prueba en seco en el i7** (modo local, el ud sin sustituir, 300 textos de fondo y los vectores del paso 1): entradas, criterio, vectores (54 s) y medir (46 s)
  pasaron en 158 s. El kNN de oro reproduce 0,5801. Las cifras del LLM con 300 textos no significan nada (unos 50 por tarea) y no se usan.

### 13.2 R2, paso 5: Qwen3-1.7B y Qwen3-0.6B en la batería (g5.xlarge, ≤ 50 min, ≤ 0,9 USD)

**Criterio congelado** (`model/alt/llm_bateria.py`, `CRIT5`): sha256 `767064e1fd6b805cafcd4aafc881cdee3de9f596b90e3905f11df7567555a971` de
`{"pasa_agrupada_min": 0.62, "pasa_hatecheck_min": 0.75, "refutado_agrupada_bajo": 0.58}`. **PASA** (sirve como L3 barato) si la agrupada ponderada (la del paso 2)
da ≥ 0,62 y hatecheck-es ≥ 0,75. **REFUTADO** si la agrupada da < 0,58. Si no, NO PASA. Se mide por modelo y por pares con el 4B bf16 en las mismas 3 116 filas.

- Modelos: `Qwen/Qwen3-1.7B` @ 70d244cc y `Qwen/Qwen3-0.6B` @ c1899de2, Apache-2.0, bf16, vLLM 0.33.0, contexto 16 k, en ese orden, con un tope de 16 min por modelo.
- Mismo arnés que el paso 2: `llm_jevbench.py`, el esquema con todos los ids, un reintento, 32 hilos y la misma `filas.jsonl` (sha256 `7ea97957…`).
- **Única diferencia:** los dos son híbridos y el cliente manda `chat_template_kwargs: {"enable_thinking": false}` (`--sin-pensar`). Es el modo sin pensamiento de su plantilla y el prompt no cambia. Sin esa opción, la petición es idéntica a la del paso 2.
- Latencia:
  - p50 y p95 de la muestra entera con 32 hilos (`ms` de cada fila);
  - p50 y p95 con UN hilo en `lat.jsonl` (10 filas por tarea), que es la cifra de L3.
- Estimación en el i7 para el modelo que pase: hasta ahora solo hay una medida, Q4 de 4B en CPU (unos 65 s por fila de hatecheck y unos 6 min por fila de massive-es, §12.2). Escalar por parámetros (1,7/4) es una cota gruesa. Si pasa, conviene medir 20 filas en el i7 con el GGUF Q4 del 1.7B.
- Medición fuera: `python -m model.alt.llm_bateria pequeno --muestra DIR --ref l3_bateria.jsonl --modelo qwen3-1.7b=l3-qwen3-1.7b.jsonl --lat qwen3-1.7b=lat-qwen3-1.7b.jsonl …`.
  Sobre la salida del 4B contra sí mismo da 0,6497 y diferencia 0, lo que confirma la medida del paso 2.
- Entradas en `s3://chispa-pruebas-cuenta-aws/llm-pequeno-1208cd9/`: llm.tgz `e22e3395…`, filas.jsonl `7ea97957…`. User-data: `lanzados/ud-llm-pequeno-1208cd9.sh`.
- **Prueba en seco en el i7**: el ud sin sustituir, contra un servidor OpenAI falso que devuelve probabilidades uniformes. Pasaron los dos modelos (autoprueba, latencia, muestra y resumen). Las 12 peticiones llevaron `enable_thinking: false`.

### 13.3 R3: tareas nuevas en español (sin AWS)

La propuesta está en `model/alt/manifiesto-tareas-nuevas.json` y la revisión es de la sesión madre:
- **APTO**: BSC-LT/LexBOE (tema legal, 12 clases, test 5 846, CC BY 4.0 y reutilización del BOE).
- **APTO con reserva**: MultiEURLEX es, nivel 1 (21 clases, multietiqueta, CC BY 4.0 y EUR-Lex).
- **DUDOSO**: fake news en español (572 filas, CC BY 4.0).

Las demás se descartaron por -SA, -NC, términos de Twitter o licencia sin verificar. Quedan huecos: no hay candidato apto para NLI, paráfrasis, emoción ni atención al cliente.

### 13.4 Orden de lanzamiento (las dos caben a la vez: la c7i no cuenta en la cuota G de 8 vCPU)
```
aws ec2 run-instances --region us-east-1 --image-id ami-ejemplo --instance-type c7i.2xlarge --count 1 \
  --subnet-id subnet-ejemplo --security-group-ids sg-ejemplo --iam-instance-profile Name=rol-ec2-pruebas \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://model/alt/aws/lanzados/ud-knn-llm-1208cd9.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-knn-llm-1208cd9}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-knn-llm-1208cd9}]'
aws ec2 run-instances --region us-east-1 --image-id ami-ejemplo --instance-type g5.xlarge --count 1 \
  --subnet-id subnet-ejemplo --security-group-ids sg-ejemplo --iam-instance-profile Name=rol-ec2-pruebas \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://model/alt/aws/lanzados/ud-llm-pequeno-1208cd9.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-pequeno-1208cd9}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-llm-pequeno-1208cd9}]'
```

### 13.5 Resultado de R1 (EC2 inst-26, c7i.2xlarge, 2026-11-15): PASA

La instancia corrió 7 min de trabajo (427 s) y unos 10 min en total contando la espera de `verificado`, por ≈ 0,06 USD. Las cifras están en `knn-llm-1208cd9.json`, que no lleva textos. Se comprobó el criterio contra el sha256 congelado `5e0c38a6…`.

**El kNN con soporte etiquetado por el LLM gana al de oro por 6,0 puntos agrupados** [IC95 +4,7, +7,3]: 0,6381 frente a 0,5783, en las 6 651 filas. hatecheck-es da 0,730.
Se cumplen los tres criterios: no pierde frente al oro (gana), la agrupada da ≥ 0,55 y hatecheck-es ≥ 0,60. No está refutado.

| k por clase (soporte) | oro | LLM | fuente (mismo fondo, etiqueta de la fuente) |
|---|---|---|---|
| 1 | 0,423 | 0,411 | |
| 2 | 0,468 | 0,441 | |
| 4 | 0,517 | 0,481 | |
| 8 | 0,578 | 0,515 | 0,575 |
| 16 | | 0,559 | |
| 32 | | 0,583 | 0,647 |
| 64 | | 0,601 | |
| 128 | | 0,617 | 0,676 |
| todo | 0,578 | **0,638** | 0,690 |

Además: el LLM con p ≥ 0,9 da 0,619, y oro ∪ LLM da **0,656**.

**Más soporte del LLM sube la curva sin aplanarse hasta «todo»:** +12 puntos entre 8 y todo. Para el mismo k, el soporte del LLM va 3–6 puntos por debajo del de oro.
El techo con etiquetas limpias en el mismo fondo es 0,690: el ruido del LLM cuesta 5,2 puntos [−6,5, −4,0]. Filtrar por p ≥ 0,9 no ayuda, porque quita soporte.

Por tarea (LLM / oro, todo el soporte):

| tarea | LLM | oro | diferencia |
|---|---|---|---|
| hatecheck-es | 0,730 | 0,6925 | +3,8 |
| inyecciones | 0,560 | 0,767 | −20,7 |
| jailbreak | 0,9625 | 0,565 | +39,8 |
| massive-es | 0,472 | 0,500 | −2,9 (n. s.) |
| massive-esc | 0,644 | 0,570 | +7,4 |
| tuits-es | 0,585 | 0,482 | +10,4 |

Lectura y reservas:
- **Inyecciones empeora.** El LLM etiqueta el fondo con acuerdo 0,72 con la fuente y el oro tiene etiquetas exactas.
- **massive-es (57 clases) no gana.** El acuerdo es 0,56, y 379 de 2 000 textos no tienen respuesta válida.
- **El salto de jailbreak** dice sobre todo que 8 ejemplos de oro eran muy poco soporte.
- **hatecheck-es** solo tiene 180 textos de fondo, de plantillas que no salen en el test. Con la etiqueta de la fuente baja a 0,583; el LLM etiqueta mejor que la fuente para lo que pide el test, o su sesgo coincide con el del test (causa no verificada).
- Los vectores de esta máquina difieren de los del paso 1 hasta en 0,057 (hatecheck-es) y 0,011 (jailbreak); las demás tareas son idénticas. Puede deberse a otra versión de transformers (5.22.2 frente a 4.46.3) o de tokenizador, sin verificar. Por eso el oro recalculado da 0,5783 frente a 0,5801. La comparación del criterio usa vectores de una sola máquina.

### 13.6 Resultado de R2 (EC2 inst-19, g5.xlarge, 2026-11-15): los dos REFUTADOS

La instancia corrió 22 min de trabajo (1 330 s) y unos 25 min en total con la espera, por ≈ 0,42 USD. Las cifras están en `llm-pequeno-1208cd9.json` (sin textos), medidas con `llm_bateria pequeno` contra `CRIT5` (sha256 `767064e1…`). Las 3 116 filas tienen respuesta en los dos modelos.

| | Qwen3-4B bf16 (paso 2) | Qwen3-1.7B | Qwen3-0.6B |
|---|---|---|---|
| agrupada ponderada | 0,6497 | **0,333** [0,314, 0,351] | **0,207** [0,193, 0,221] |
| frente al 4B (por pares) | | −31,7 [−34,3, −29,1] | −44,3 [−46,8, −41,8] |
| hatecheck-es | 0,824 | 0,478 | 0,274 |
| inyecciones | 0,569 | 0,474 | 0,483 |
| jailbreak | 0,924 | 0,644 | 0,624 |
| massive-es (inválidas de 250) | 0,444 (45) | 0,056 (224) | 0,012 (231) |
| massive-esc (inválidas de 250) | 0,564 (46) | 0,184 (176) | 0,032 (239) |
| tuits-es | 0,636 | 0,500 | 0,384 |
| latencia un hilo p50 / p95 | | 283 / 11 025 ms | 123 / 5 721 ms |
| latencia 32 hilos p50 / p95 | 760 / 18 104 ms | 476 / 27 773 ms | 241 / 22 970 ms |
| acuerdo top-1 con el 4B | | 0,509 | 0,311 |

Los dos quedan muy por debajo de 0,58: **REFUTADOS como L3 barato** con este arnés. No hay modelo que pase, así que no hay estimación de CPU en el i7.
Según el plan, tampoco se prepara la destilación 4B → 1,7B.

Comprobado que no es un fallo del arnés:
- hay 0 respuestas uniformes en el 1.7B;
- el 0.6B dice `false` en las 2 000 filas de hatecheck-es (el 73 % es `true`);
- el 1.7B dice `false` en el 69 %;
- en massive las respuestas inválidas son vectores de ceros o todos iguales sobre las 60 opciones.

El mismo cliente da 0,6497 con el 4B. La latencia de un hilo es baja en las tareas binarias (90–280 ms), pero massive (60 opciones en JSON) dispara el p95.

Reserva: el arnés pide probabilidades en JSON, que es exigente para un modelo pequeño. Otro formato, como logprobs sobre la etiqueta, podría dar más. Sería otro método, no una repetición, y habría que congelarlo aparte.

### 13.7 R3 convertido en el i7 (código 912a77b): dos tareas, MultiEURLEX no viable

**Uso: solo para medir, nunca para entrenar.**

Las tres fuentes se bajaron a `/mnt/nfs-lab/chispa-audio/tareas-nuevas/hub` en sus revisiones fijadas. Se convirtieron con `python -m model.alt.tareas_nuevas`, que sigue las reglas de la batería no vista:
- soporte de 8 por clase sacado de train;
- test barajado con un tope de 2 000;
- texto cortado a 3 000 caracteres.

La batería nueva está en `tareas-nuevas/bat/` (sha256: lexboe.jsonl `2f2af9de…`, fakenews-es.jsonl `b28b5dae…`). El manifiesto, que solo lleva cifras, está en `tareas-nuevas-manifiesto-912a77b.json`.

Ninguna de las tres fuentes aparece en los manifiestos de entrenamiento: `manifiesto-entrenamiento.json`, `manifiesto-l2x.json` y `model/datos`. tsjev no está en el i7, así que no se filtró contra tsjev. El riesgo es bajo, porque son dominios distintos.

| tarea | test | soporte | clases | cortados a 3 000 car. | notas |
|---|---|---|---|---|---|
| lexboe | 2 000 | 96 (8 × 12) | 12 | 988 | muy desequilibrada: Funcionarios y Personal 946, Vivienda 7, Fuerzas de Seguridad 15. Los nombres reales de 3 clases no son los de la tarjeta (se cuentan en los datos) |
| fakenews-es | 572 | 0 (solo zero-shot) | 2 | 236 | 286 / 286; «true» = noticia falsa |
| eurlex-n1 | — | — | — | — | **no viable**: de 5 000 documentos de test, 61 tienen un único descriptor de nivel 1 y solo UNIÓN EUROPEA llega a 20 |

**Atribución** (obligatoria en todo informe que use estas tareas):
- LexBOE: «Fuente de los datos: Agencia Estatal Boletín Oficial del Estado (BOE), vía BSC-LT/LexBOE (CC BY 4.0). Datos modificados.»
- Fake news: «Posadas-Durán et al., FakeNewsCorpusSpanish (CC BY 4.0), vía mariagrandury/fake_news_corpus_spanish.»
- MultiEURLEX, aunque no se use: «© Unión Europea, EUR-Lex; descriptores EuroVoc © Unión Europea.»

**Reserva para el veredicto del paso 4:** de las tres tareas nuevas solo quedan dos, y una es legal y la otra solo de zero-shot. Medirán poco la generalidad.
No se encontró ninguna cuarta tarea apta que no sea legal. La única dudosa, horeca-spanish-reviews, es sintética de Gemini y la decide Juan.


## 14. Rondas R4, R5 y R6 y la integración (preparadas, sin lanzar; rama `llm-rondas-r456`)

### 14.1 R4: zero-shot del 4B en las tareas nuevas y fondo ampliado (g5.xlarge, ≤ 60 min, ≈ 1,0 USD; código 9f68027)

**Criterio congelado antes de medir** (`knn_soporte_llm.CRIT4`, `--criterio r4`). Es el sha256 `a273e9271013035e5725af0fbec6580d4dcd6c995da3b98465b083fa0c0950ce` de
`{"fondo": "fondo+fondo2", "pasa_agrupada_min": 0.66, "refutado_si_agrupada_le": 0.638, "variante": "llm/todo"}`.
- La medida es el kNN de R1 con todo el soporte válido del LLM: fondo de R1 más fondo2, sobre la batería original de 6 tareas.
- **PASA** si la agrupada da ≥ 0,66. **REFUTADO** si da ≤ 0,638, es decir, si no mejora R1. Si no, NO PASA.
- Se mide en R6, en la c7i, con los vectores de esa máquina.

Qué etiqueta el 4B (mismo arnés que el paso 2):
- (a) `filas-nuevas.jsonl`: todo el test de lexboe (2 000) y de fakenews-es (572). Da el zero-shot del LLM y es la entrada de R6.
- (b) `fondo2.jsonl`: 18 000 textos sin etiqueta (`llm_bateria fondo2`):
  - 6 000 de massive-es, de 10 467 disponibles;
  - 8 000 de massive-esc, de 10 456;
  - 4 000 de LexBOE, de 52 158.
  - Mismas exclusiones que el fondo: ningún texto de test ni de soporte, de la batería ni de las tareas nuevas, y ninguno ya usado en el fondo de esa tarea.
- Las demás tareas del fondo (hatecheck-es, inyecciones, jailbreak, tuits-es) ya estaban agotadas en R1. Solo massive y LexBOE tienen más texto con licencia aprobada.

**No se llega a 25–30 k.** En R1, un texto de massive-es costó 0,93 s de pared, el 74 % del tiempo del fondo, porque son 60 opciones en JSON. En 60 minutos caben el test nuevo y unos 8–10 k textos de fondo.
- El orden va por prioridad: primero el test nuevo; luego el bloque 1 del fondo2 (LexBOE intercalado con la mitad de massive-esc, 8 000 textos); luego el bloque 2 (massive-es intercalado con el resto de massive-esc).
- Lo etiquetado queda aunque se corte. Total esperado: unos 16–18 k textos con el fondo de R1.

Datos:
- Entradas en `s3://chispa-pruebas-cuenta-aws/llm-r4-9f68027/`: llm.tgz `c8c0cdfd…`, filas-nuevas.jsonl `9dab2c69…`, fondo2.jsonl `b788b574…`.
- User-data: `lanzados/ud-llm-r4-9f68027.sh` (sha256 `dfa4c847…`).

Prueba en seco en el i7, con el ud sin sustituir y el servidor falso: pasaron entradas, autoprueba, nuevas y fondo2. `run-instances --dry-run` da DryRunOperation.
Orden de lanzamiento: el de §13.4 para la g5, con `--user-data file://model/alt/aws/lanzados/ud-llm-r4-9f68027.sh` y Name `chispa-llm-r4-9f68027`.

### 14.2 Integración (`integracion-uso-general` @ 4799323, sin push y sin tocar main)

Se creó desde `llm-rondas-r123`:
- Se fusionaron audio-experimento y servidor-p1-m1, sin conflictos.
- arquitectura-p0-p2, servidor-p1-latencia, fusion-knn-l2 y llm-bateria ya estaban dentro.

Verificación:
- `make test`, `make smoke`, `go vet ./...` y `go build ./...` pasan en arena, server, runtime, sdk/go y format.
- Pasan 45 tests de model/alt sin torch.
- `delega scan` da limpio.

Fallos que ya estaban antes de la integración, comprobados en las ramas de origen:
- `demo/wasm/LICENSE.go` rompe `go vet` y `go build` de demo/wasm.
- Varios tests de model/datos, cascada, l1, liston y l2 fallan en sus aserciones.
- Los tests que necesitan torch o pyarrow, incluido model/audio, no se pueden ejecutar en el Mac.

**Deuda anotada, no se arregla en esta tanda** (orden de la sesión madre):
- `demo/wasm/LICENSE.go` rompe `go vet` y `go build` de demo/wasm;
- hay aserciones que ya fallaban en model/datos (test_fuentes, test_libres, test_reglas), cascada (test_capas, test_informe), l1, liston y l2 (test_corpus_poda, test_aumento, test_w2);
- los tests de torch y pyarrow sin ejecutar en el Mac, entre ellos model/audio.

### 14.3 R5: el paso 5 con logprobs en vez de JSON (g5.xlarge, ≤ 52 min, ≈ 0,87 USD; código 0bcbd78)

**Criterio congelado** (`llm_bateria.CRIT5L`, `pequeno --criterio r5`): sha256 `17c1ea124ed0a2f9f1221f339b964af3a25dffca4f2bee509f4e9a3a6d5c1576`. Mismos umbrales que CRIT5:
- **PASA** con agrupada ≥ 0,62 y hatecheck-es ≥ 0,75;
- **REFUTADO** con agrupada < 0,58;
- el sha es propio porque el método es nuevo.
Se mide en las mismas 3 116 filas del paso 2 y, por pares, frente al 4B en JSON (0,6497).

Método (`model/alt/llm_logprobs.py`, que usa el prompt de `llm_jevbench`):
- Cada opción lleva una etiqueta de un carácter (A–Z, a–z, 0–9; hasta 62). Se pide `max_tokens=1` con `logprobs` y `top_logprobs=20`.
- Las etiquetas que no salen en el top se puntúan con `prompt_logprobs`, terminando la conversación en la etiqueta (`continue_final_message`). Es exacto para las 4 primeras; el resto recibe un piso igual al mínimo conocido menos 1.
- probs = softmax sobre las opciones. Una fila es inválida solo si no se puntúa ninguna opción.
- Los híbridos llevan `--sin-pensar`; el 4B-Instruct-2507 no.

La g5 corre tres modelos en este orden: Qwen3-1.7B @ 70d244cc, Qwen3-0.6B @ c1899de2 y Qwen3-4B-Instruct-2507 @ cdbee75f. Es vLLM 0.33.0 con `--max-logprobs 20` y un tope de 12 min por modelo; los pesos se bajan en segundo plano.
El 4B es la referencia del método: si en logprobs se queda muy por debajo de 0,6497, el fallo es del método y no de los modelos pequeños.

Riesgos sin verificar contra un vLLM real:
- el formato del campo `prompt_logprobs` en la respuesta de chat de 0.33.0. Si no es el esperado, el relleno cae al piso y se anota en `error`; la fila sigue siendo válida.
- que un carácter ASCII suelto sea un único token en Qwen.
- que el 4B, el último, quepa en el tope.

Datos:
- Entradas en `s3://chispa-pruebas-cuenta-aws/llm-logprobs-0bcbd78/`: llm.tgz `af1c3c9a…` (llm_logprobs.py y llm_jevbench.py), filas.jsonl `7ea97957…`.
- User-data: `lanzados/ud-llm-logprobs-0bcbd78.sh` (sha256 `d51132db…`).
Prueba en seco: el subagente que lo escribió corrió el ud con LOCAL=1 contra un servidor falso de logprobs; pasaron los tres modelos y salieron resumen.json y resultados.tgz. pytest da 10 pruebas en verde y `run-instances --dry-run` da DryRunOperation.

### 14.4 R6: simulación del paso 4 en una c7i, más la medida de R4, las versiones y la latencia (≤ 55 min, ≈ 0,33 USD; código 805d1e0)

**Criterio congelado** (`model/alt/prequential.py`, `CRIT_R6`): sha256 `288d1260f7bb5868218ec0c676e390710a42c642b2166aa2942fbea2acd8f064` de
`{"alfa": 0.5, "nuevas": ["hatecheck-es", "lexboe", "fakenews-es"], "perdida_max_original": 0.02, "ref_nuevas": "oro_estatico", "ref_originales": "oro_llm_estatico", "refutado_si_dif_le": 0.0, "regla": "peso", "retraso": 50, "sube_min": 0.05, "temp": 0.05, "variante": "preq_oro_fondo", "vecinos": 32}`.
- **PASA** si se cumplen las dos:
  - Las nuevas agrupadas (aciertos / casos) con `preq_oro_fondo` suben ≥ 5 puntos sobre el kNN estático de oro.
  - Ninguna tarea de la batería pierde más de 2 puntos frente a oro ∪ fondo-LLM estático (el 0,656 de R1).
- **REFUTADO** si las nuevas no superan al estático de oro. En otro caso, NO PASA.

Simulación:
- Cada tarea es un flujo de las filas de test con etiqueta del LLM: hatecheck-es completa, unas 250 del resto, lexboe 2 000 y fakenews-es 572. Las etiquetas salen de l3_bateria del paso 2 y l3_nuevas de R4.
- Cada fila se predice con kNN sobre el banco. La etiqueta del LLM de la fila t entra al banco tras predecir la fila t+50.
- El banco empieza con 8 ejemplos de oro por clase; en fakenews-es empieza vacío.
- **El oro manda:** en el voto, un vecino de oro pesa 1 y uno del LLM pesa α = 0,5. La variante con umbral (s ≥ 0,90: solo oro) es descriptiva y no entra en el criterio.
- Variantes: oro estático, oro ∪ fondo-LLM estático, prequential sobre oro y prequential sobre oro ∪ fondo. Se usan 10 semillas de orden.

Reservas, dichas antes de medir:
- En fakenews-es el estático de oro no tiene banco: predice una constante, así que su 50 % es el de clases equilibradas. Lo que sube ahí mide tener banco frente a no tenerlo. Se da el desglose por tarea y el acierto directo del LLM.
- hatecheck-es cuenta a la vez como nueva y como original.
- Con unas 250 filas por tarea, una pérdida puntual de 2 puntos está dentro del ruido. Se da el IC pareado por tarea, pero el criterio es puntual.

Otros pasos de la misma máquina, por prioridad:
1. Vectores de e5-small con transformers 5.22.2 de la batería, las nuevas, el fondo y fondo2.
2. R4 (CRIT4).
3. R6.
4. Vectores de la batería con transformers 4.46.3 y tokenizers 0.20.3, con el torch de la AMI y con torch 2.5.1+cpu, frente al paso 1 (i7: torch 2.5.1+cpu y transformers 4.46.3). Cierra la diferencia de R1.
5. Latencia de L0–L2 int8 con 1 vCPU: `docker run --cpus=1 --memory=1800m`, sin Lambda ni IAM; `model/alt/aws/latencia_1vcpu/`, imagen `python:3.12-slim@sha256:5e2bdb1b…`, ORT 1.31.1, 400 peticiones y 5 arranques en frío.
   - Hace falta `modelo-int8.chispa` (perfil movil-int8, 249 716 640 bytes, sha256 `995dd6a7…`), que sale de `model/bench/movil/aws/modelos.sh` en un prefijo `movil-AAAAMMDD`. Esa fecha no está en el repo ni en la bóveda.
   - Sin el prefijo, la latencia queda en `omitidos`.
   - L0 sale omitido de todos modos: el perfil movil-int8 trae el L0 de kindling v2, no L0TF.
   - Las cifras son del cómputo (los mismos grafos), no del servidor Go.

Prueba en seco en el i7 (el ud sin sustituir, 400 textos de fondo, etiquetas falsas de R4): entradas, criterios, vectores (645 s), R4 (48 s) y R6 (16 s) pasaron. Los veredictos de esa prueba no significan nada.

Entradas: en `s3://chispa-pruebas-cuenta-aws/knn-r6-805d1e0/` ya están todas menos `l3_fondo2.jsonl` y `l3_nuevas.jsonl`, que se suben al verificar R4. User-data: `lanzados/ud-knn-r6-805d1e0.sh` (sha256 `8276b475…`). `--dry-run` da DryRunOperation.
**Se lanza solo después de que yo suba las dos entradas de R4**: sin ellas el ud se para en el paso 2.

### 14.5 Resultado de R5 (EC2 inst-01, g5.xlarge, 2026-11-15): pequeños REFUTADOS; el 4B por logprobs da 0,729

La instancia corrió 25 min de trabajo (1 511 s), unos 28 min en total contando la espera, por ≈ 0,47 USD. Las cifras están en `llm-logprobs-0bcbd78.json`, que no lleva textos ni `raw`, medidas con `llm_bateria pequeno --criterio r5` (CRIT5L, sha256 `17c1ea12…`).
Respondieron las 3 116 filas de los tres modelos y no hubo ninguna inválida. El relleno por `prompt_logprobs` funcionó: no se anotó ni un error. Las 250 filas de massive-es llevan, además, el piso en alguna opción.

| | 4B en JSON (paso 2) | **4B por logprobs** | 1.7B por logprobs | 0.6B por logprobs |
|---|---|---|---|---|
| agrupada ponderada | 0,6497 | **0,729** [0,708, 0,751] | 0,394 [0,374, 0,413] | 0,185 [0,171, 0,199] |
| frente al 4B en JSON (por pares) | | **+8,0** [+5,8, +10,1] | −25,6 | −46,5 |
| hatecheck-es | 0,824 | 0,831 | 0,726 (dice `true` en las 2 000) | 0,274 (dice `false` en todas) |
| inyecciones | 0,569 | **0,767** | 0,517 | 0,483 |
| jailbreak | 0,924 | 0,916 | 0,360 | 0,360 |
| massive-es | 0,444 (45 inválidas) | **0,656** (0) | 0,088 | 0,012 |
| massive-esc | 0,564 (46 inválidas) | **0,684** (0) | 0,268 | 0,044 |
| tuits-es | 0,636 | 0,640 | 0,372 | 0,336 |
| latencia un hilo p50 / p95 | | 33 / 1 402 ms | 20 / 758 ms | 14 / 434 ms |

Veredicto con CRIT5L:
- **Qwen3-1.7B y Qwen3-0.6B, REFUTADOS también por logprobs.** Ya no es el formato. El 1.7B no discrimina: responde una constante en hatecheck-es y un 36 % en jailbreak. Ningún Qwen3 pequeño sirve de L3 barato con este prompt.
- **El método está validado por el 4B y lo mejora mucho:** 0,729 frente a 0,6497. Pasaría el criterio de agrupada del paso 2 (≥ 0,65), que el JSON no alcanzó. Las ganancias vienen de las tareas con muchas opciones o con JSON inválido: massive-es +21, massive-esc +12 e inyecciones +20.
- **Latencia:** el p50 del 4B con un hilo es de 33 ms, porque genera 1 token en vez de un JSON. El p95 de 1,4 s viene de massive, donde hay que puntuar etiquetas fuera del top 20.

**Lectura para L3:** el cambio que más gana no es un modelo más pequeño, sino el formato. El mismo 4B, pidiéndole una etiqueta y leyendo logprobs, es más exacto y mucho más rápido. Hay que confirmarlo con otra muestra antes de adoptarlo: la muestra es la del paso 2, que ya se miró, aunque el método se congeló antes de ver estas cifras.
Las etiquetas del fondo de R1 y R4 están hechas en JSON; un fondo etiquetado por logprobs podría tener menos ruido (sin medir).

### 14.6 R7: confirmación del 4B por logprobs y reetiquetado del soporte (g5.xlarge, ≤ 50 min, ≈ 0,84 USD; código 4747eb1)

**Criterio congelado** (`llm_bateria.CRIT7`, `llm_bateria confirmar`): sha256 `90b2ad87440627d9780b4675ab7d129e272cb71f8a9878eba4f6f2e45d58ba12`.
- Se mide el 4B con logprobs (el mismo código que R5) en el test de la batería que no entró en la muestra del paso 2: 3 535 filas (jailbreak 150, massive-es 1 253, massive-esc 1 188, tuits-es 944).
- hatecheck-es e inyecciones no tienen filas fuera de la muestra, así que **decide solo la agrupada**, ponderada por el tamaño de cada tarea en la batería.
- **PASA** si la agrupada da ≥ 0,70. **REFUTADO** si da < 0,66.
- Se mide sobre las filas respondidas. El orden es aleatorio dentro de cada tarea y está intercalado, así que un corte deja una submuestra al azar de cada tarea.

Tiempo, medido en R5: con logprobs una fila de massive-es cuesta 1,17 s de pared y una de massive-esc 0,37 s, porque se puntúan las etiquetas fuera del top 20. Las demás cuestan menos de 0,04 s.
- La confirmación completa son ≈ 32 min. Lleva un tope de `CONF_MIN = 24` min, que deja ≈ 2 300–2 600 filas, para que quede tiempo de reetiquetar.
- Después se reetiqueta `reetiqueta.jsonl` (28 709 textos) por prioridad:
  1. el test de lexboe y fakenews-es (2 572, ≈ 9 min);
  2. el fondo de R1, primero las tareas baratas (4 137), luego massive-esc (2 000) y luego massive-es (2 000);
  3. fondo2.
- En 50 min no llega a fondo2: el reetiquetado cubrirá el test nuevo, las tareas baratas del fondo y parte de massive-esc.

Datos:
- `llm.tgz` lleva el mismo `llm_logprobs.py` y `llm_jevbench.py` que R5, con los mismos sha256 de fichero, aunque el tarball cambie.
- Entradas en `s3://…/llm-r7-4747eb1/`: llm.tgz `8df0eb7b…`, filas-confirma.jsonl `0b6b5eb1…`, reetiqueta.jsonl `fd7d00d0…`.
- User-data: `lanzados/ud-llm-r7-4747eb1.sh`.

Prueba en seco en el i7, con el ud sin sustituir y un servidor de logprobs falso: pasaron entradas, autoprueba, confirmación y reetiquetado. `--dry-run` da DryRunOperation.

### 14.7 Resultado de la parte LLM de R4 (EC2 inst-20, 2026-11-15); la medida del kNN (CRIT4) va en R6

La instancia corrió su tope de 60 min (≈ 1,0 USD), con `verificado` subido.

**(a) Test de las tareas nuevas:** 2 572 de 2 572 filas respondidas, pero solo 1 477 válidas. fakenews-es tiene 572/572 válidas y **lexboe 905/2 000**.
- En lexboe, el JSON del 4B no suma 1: pone 0,93 en dos clases a la vez.
- Zero-shot del 4B en JSON (`llm-r4-nuevas-9f68027.json`, IC95 bootstrap):

| tarea | estricta | permisiva |
|---|---|---|
| lexboe | **0,067** [0,056, 0,078] | 0,222 [0,204, 0,239] |
| fakenews-es | 0,559 [0,518, 0,600] | 0,559 |

- Incluso contando solo las respuestas válidas, lexboe da ≈ 0,15. El LLM no sirve en zero-shot para esa tarea con este prompt: 12 clases jurídicas, un 47 % de «Funcionarios y Personal» y el texto cortado a 3 000 caracteres.

**(b) fondo2:** se etiquetaron 5 440 de 18 000 textos en 26 min, todos del bloque 1: 2 719 de lexboe (1 259 válidos) y 2 721 de massive-esc (2 210 válidos). massive-es no llegó.
- El soporte del LLM queda en 7 372 + 3 469 = 10 841 textos válidos. En la batería original, el fondo2 solo añade massive-esc.
- Así que CRIT4 (≥ 0,66) depende casi solo de cuánto suba massive-esc.

R6 se mide tal como se congeló, con estas etiquetas en JSON. En lexboe, el flujo prequential solo añade al banco las 905 etiquetas válidas.

**R6 con las etiquetas de R4** (código 12ebb49):
- User-data: `lanzados/ud-knn-r6-12ebb49.sh` (sha256 `1df0d07f…`). Sustituye al de 805d1e0, que no tenía el modelo int8.
- Entradas, todas en `s3://…/knn-r6-12ebb49/`: l3_fondo2 `a100a537…`, l3_nuevas `52874409…`, codigo.tgz `7a0f2884…`, latencia.tgz `cc6aeb84…`.
- Latencia con `modelo-int8.chispa` de `movil-90e3263c/movil`; el sha256 se comprueba antes de medir.
- `--dry-run` da DryRunOperation.

### 14.8 Resultados de R6 en la c7i (EC2 inst-34, 2026-11-15): R4 NO PASA, R6 REFUTADO; versiones y latencia

La instancia trabajó 995 s y terminó a los ≈ 19 min, por ≈ 0,11 USD. Ficheros (solo cifras): `knn-r4-12ebb49.json`, `prequential-r6-12ebb49.json`, `versiones-r6-12ebb49.jsonl` y `latencia-1vcpu-12ebb49.json`.

**R4, kNN con el fondo ampliado (CRIT4 `a273e927…`): NO PASA.**
- La agrupada da 0,6389, frente a 0,6381 en R1. No llega a 0,66, pero tampoco cae a ≤ 0,638.
- El fondo2 solo creció en massive-esc: de 1 614 a 3 824 textos válidos. massive-esc pasó de 0,644 a 0,647; el resto no cambió.
- Con los mismos textos y la etiqueta de la fuente, el kNN da 0,712, frente a 0,690 en R1. Más soporte limpio sí sube la curva; el del LLM en JSON, apenas.

**R6, simulación prequential (CRIT_R6 `288d1260…`): REFUTADO.**
- Las nuevas agrupadas con `preq_oro_fondo` dan 0,471, frente a 0,632 del estático de oro (−16,0 puntos).
- Las originales no pierden frente a oro ∪ LLM estático. Con `preq_oro_fondo` dan 0,711, frente a 0,7105, y en ninguna tarea se pierden más de 2 puntos.

| tarea | oro estático | oro ∪ LLM estático | preq oro | preq oro ∪ fondo | preq oro (umbral, descriptivo) | preq oro ∪ fondo (umbral, descriptivo) | LLM directo |
|---|---|---|---|---|---|---|---|
| hatecheck-es | 0,6925 | 0,7345 | 0,725 | 0,731 | 0,693 | 0,697 | 0,824 |
| lexboe | 0,6085 | **0,2075** | 0,253 | **0,203** | 0,648 | 0,637 | 0,067 |
| fakenews-es | 0,500 | 0,500 | 0,500 | 0,500 | 0,500 | 0,500 | 0,559 |
| inyecciones | 0,767 | 0,569 | 0,745 | 0,573 | 0,695 | 0,585 | 0,569 |
| jailbreak | 0,556 | 0,960 | 0,704 | 0,964 | 0,737 | 0,962 | 0,924 |
| massive-es | 0,480 | 0,520 | 0,484 | 0,542 | 0,491 | 0,546 | 0,444 |
| massive-esc | 0,580 | 0,664 | 0,607 | 0,674 | 0,610 | 0,650 | 0,564 |
| tuits-es | 0,496 | 0,572 | 0,513 | 0,567 | 0,499 | 0,568 | 0,636 |
| **nuevas agrupadas** | 0,632 | 0,475 | 0,490 | **0,471** | **0,649** | 0,646 | |
| **originales agrupadas** | 0,6425 | 0,7105 | 0,678 | **0,711** | 0,658 | 0,688 | |

Lectura:
- **Lo refuta lexboe, y lexboe mide sobre todo el JSON** (reserva pedida por la sesión madre). Las etiquetas del 4B en JSON aciertan un 6,7 % en lexboe: el 45 % son inválidas y las válidas dan ≈ 0,15. El voto con α = 0,5 no frena ese ruido y hunde el kNN de 0,61 a 0,20.
- **La regla del umbral** (si el oro más cercano tiene similitud ≥ 0,90, se usa solo el oro) sí protege lexboe: da 0,648 sin fondo y 0,637 con él. Con ella las nuevas suben 1,4–1,7 puntos sobre el oro, lejos de los 5 que pedía el criterio, y las originales ceden 2,3 puntos frente al estático oro ∪ LLM. Es descriptiva, no estaba congelada, y no cambia el veredicto.
- **inyecciones** sigue cayendo con el LLM con cualquier regla de peso (0,767 → 0,57). Con umbral se queda en 0,695. «El oro manda» por peso no basta.
- **fakenews-es** se queda en 0,500 con todas las variantes. Las etiquetas del LLM (0,559) no le dan al kNN nada mejor que el azar.
- R7 y R6b repiten la medida con etiquetas por logprobs.

**Diferencia de vectores frente al paso 1: cerrada.**
- Con torch 2.5.1+cpu y transformers 4.46.3 (las versiones del paso 1), los vectores de esta c7i son **idénticos** a los del paso 1 en las 6 tareas (diferencia máxima 0,0). Frente a transformers 5.22.2 vuelven a salir 0,057 en hatecheck-es y 0,011 en jailbreak.
- La causa es la versión de la biblioteca, no la máquina.
- No se pudo separar torch de transformers: la variante con el torch de la AMI falló porque el venv no ve el torch de `/opt/pytorch`.
- Para comparar entre rondas, los vectores van fijados a la versión de transformers que se use.

**Latencia de L1, L2 y torres int8 con 1 vCPU y 1,8 GB** (`docker --cpus=1 --memory=1800m`, Xeon 8488C, `modelo-int8.chispa` con sha256 comprobado, 400 peticiones de tuits-es, 128 tokens, 6 opciones):

| | p50 | p95 | p99 |
|---|---|---|---|
| L1 | 1,5 ms | 4,5 ms | 6,9 ms |
| L2 | 6,6 ms | 9,1 ms | 9,7 ms |
| cadena L1 + L2 | **8,6 ms** | 13,0 ms | 15,6 ms |
| torres al vuelo | 27,6 ms | 32,5 ms | 33,8 ms |
| cadena zero-shot (L1 + L2 + torres) | 36,6 ms | 42,1 ms | 45,2 ms |

- **Frío:** desde que arranca el proceso hasta la primera respuesta pasan 3,5–3,8 s; con `docker run` entero, 5,0–5,5 s. La carga se reparte entre L1 (1,1–1,2 s) y L2/torres (0,7–0,9 s cada una).
- **Memoria:** el pico es de 1 423 MB de RSS y 1 481 MB en el cgroup, dentro de 1,8 GB pero con poco margen.
- L0 sale omitido: el perfil movil-int8 trae el L0 de kindling v2.
- Es el cómputo con los mismos grafos, no el servidor Go. El frío no incluye la descarga de la imagen que haría Lambda.
- La instancia tenía una carga de 2,4–3,7 por los pasos anteriores, aunque el contenedor iba fijado a un núcleo. Puede inflar algo las colas.

### 14.9 Resultado de R7 (EC2 inst-04, g5.xlarge, 2026-11-15): confirmación NO PASA por poco; logprobs arregla las tareas nuevas

La instancia trabajó 38 min (2 294 s), unos 40 min en total con la espera, por ≈ 0,68 USD.
Ficheros: `llm-r7-confirma-4747eb1.json` y `llm-r7-nuevas-4747eb1.json`, solo cifras.
No hubo ninguna respuesta inválida en ninguna fila.

**Confirmación fuera de la muestra (CRIT7 `90b2ad87…`): NO PASA.**
Respondieron 2 961 de 3 535 filas en el tope de 24 min, intercaladas al azar por tarea.

| tarea | n | 4B por logprobs, fuera de la muestra | 4B por logprobs, en la muestra (R5) |
|---|---|---|---|
| jailbreak | 150 | 0,907 [0,860, 0,947] | 0,916 |
| massive-es | 932 | 0,637 [0,607, 0,667] | 0,656 |
| massive-esc | 935 | 0,724 [0,695, 0,752] | 0,684 |
| tuits-es | 944 | 0,646 [0,617, 0,677] | 0,640 |
| **agrupada ponderada (estas 4 tareas)** | 2 961 | **0,691** [0,673, 0,707] | **0,684** (recalculada con las mismas 4 tareas y pesos) |

La agrupada queda entre 0,66 y 0,70: NO PASA, pero no está refutada.

**La ganancia se confirma.** Con las mismas cuatro tareas y los mismos pesos, el resultado fuera de la muestra (0,691) es igual o mejor que en la muestra (0,684). El 0,729 de R5 sale más alto porque incluye hatecheck-es (0,831, con peso 2 000) e inyecciones, que no tienen filas fuera de la muestra.

Nota: el umbral de 0,70 de CRIT7 se fijó sin descontar ese efecto. Es un aviso para el siguiente criterio; el veredicto no cambia.

**Reetiquetado (lo que llegó): 1 790 textos, todos del test nuevo.**
Son las 572 filas de fakenews-es y 1 218 de las 2 000 de lexboe. No llegó ningún texto del fondo de R1 ni de fondo2, porque la confirmación se comió el tiempo.

Mismas filas, 4B por logprobs frente a 4B en JSON (permisivo):

| tarea | n | logprobs | JSON permisivo |
|---|---|---|---|
| lexboe | 1 218 | **0,530** [0,503, 0,560] | 0,236 |
| fakenews-es | 572 | **0,682** [0,643, 0,720] | 0,559 |

**R6b tal como se congeló no informaría.** Su CRIT6B compara el kNN con el fondo reetiquetado por logprobs frente al de JSON. Como no se reetiquetó ni un texto del fondo, las dos variantes serían idénticas y la diferencia daría 0 por construcción. No conviene lanzarlo así.
Hace falta antes una ronda que reetiquete el fondo. Lo que tiene más ruido en JSON son lexboe, massive-esc e inyecciones; massive-es cuesta 1,17 s por texto.

### 14.10 R7b: reetiquetado por logprobs en dos g5 a la vez y alcance de R6b (preparado, sin lanzar; código 540c8c1)

El ud es el de R7 sin la confirmación (`llm_r7b.sh`), con el mismo cliente y el mismo modelo. Lleva un tope de 35 min por máquina (≈ 0,59 USD cada una).
Solo se reetiquetan textos que ya tienen etiqueta JSON, para que R6b compare etiquetas y no tamaños de soporte.

Colas, en orden de prioridad (estimación con los costes de R5 y R7):

| cola | contenido | estimación |
|---|---|---|
| A, 9 638 textos | resto del test de lexboe (782); fondo de R1 de hatecheck-es, inyecciones, jailbreak y tuits-es (4 137); fondo de R1 de massive-esc (2 000); lexboe de fondo2 con etiqueta de R4 (2 719) | ≈ 31 min; lexboe de fondo2 se cortará en parte |
| B, 4 721 textos | massive-esc de fondo2 con etiqueta de R4 (2 721); fondo de R1 de massive-es (2 000) | ≈ 17 min más massive-es a 1,17 s por texto; massive-es llegará a ≈ 500–600 |

Entradas:
- `s3://…/llm-r7b-a-540c8c1/`: reetiqueta.jsonl `1b08bac9…`.
- `s3://…/llm-r7b-b-540c8c1/`: reetiqueta.jsonl `9da77e8b…`.
- El llm.tgz es el mismo de R5 en contenido.
- User-data: `ud-llm-r7b-a-540c8c1.sh` (sha256 `ee33bfd8…`) y `ud-llm-r7b-b-540c8c1.sh` (sha256 `b08000a3…`).
- Prueba en seco en el i7 con el servidor falso: OK. Los `--dry-run` dan DryRunOperation.

**Alcance de R6b**, fijado antes de medir (`knn_soporte_llm.CRIT6B_ALCANCE`, sha256 `7e885b2a04e36f42b295de8d66416871cbfe1df0696be859b574ce70af4feeed`).
- CRIT6B sigue igual (sha `75922ecb…`: PASA con ≥ +2 puntos, REFUTADO con ≤ 0). Lo que se fija ahora es dónde se mide:
  - solo en las tareas de la batería original con ≥ 50 % de su soporte JSON válido reetiquetado por logprobs;
  - las logprobs solo sustituyen etiquetas de textos que ya tenían etiqueta JSON.
- Lo pidió la sesión madre después de ver que R7 no reetiquetó el fondo.
- El informe lista las tareas incluidas y la fracción reetiquetada de cada una. Por la estimación, massive-es quedará fuera.

### 14.11 Resultado de R7b (EC2 inst-16 e inst-32, 2026-11-15) y ud de R6b

Las dos colas etiquetaron 16 min cada una y no hubo ninguna respuesta inválida. Las dos recibieron `verificado`.

| cola | textos etiquetados | qué cubren |
|---|---|---|
| A | 6 887 de 9 638 | el test de lexboe que faltaba (782), todo el fondo de R1 de hatecheck-es, inyecciones, jailbreak y tuits-es (4 137) y 1 968 de los 2 000 de massive-esc del fondo de R1 |
| B | 2 728 de 4 721 | todo massive-esc de fondo2 que tenía etiqueta de R4 (2 721) y solo 7 de massive-es |

En lexboe de fondo2 no se reetiquetó ningún texto.

Junto con R7, `lp_reetiqueta.jsonl` reúne 11 405 etiquetas por logprobs:
- el test entero de lexboe (2 000) y de fakenews-es (572);
- el soporte de 5 tareas de la batería, casi completo.

**massive-es se queda prácticamente sin reetiquetar** (7 de 2 000). Por el alcance congelado (≥ 50 %), R6b la deja fuera.

**R6b (código 4cb8cc7)** se ejecuta en una c7i.2xlarge con un tope de 40 min (≈ 0,24 USD):
- User-data: `lanzados/ud-knn-r6b-4cb8cc7.sh` (sha256 `3774b223…`).
- Entradas en `s3://…/knn-r6b-4cb8cc7/`: lp_reetiqueta `7d22bae3…` (R7 + A + B), lp_bateria `16274abf…` (el 4B por logprobs de R5), codigo.tgz `9716c83b…`. El resto, igual que en R6.
- Criterios: CRIT6B `75922ecb…` y su alcance `7e885b2a…`. El prequential descriptivo reutiliza CRIT_R6 `288d1260…`.
- `--dry-run` da DryRunOperation.

### 14.12 Resultado de R6b (EC2 inst-09, c7i.2xlarge, 2026-11-15): NO PASA (+1,7 puntos, pedía +2)

La instancia trabajó 12 min (719 s) por ≈ 0,09 USD. Ficheros: `knn-r6b-4cb8cc7.json` y `prequential-r6b-4cb8cc7.json`, solo cifras.

**CRIT6B (`75922ecb…`, alcance `7e885b2a…`): NO PASA.**
- Medido sobre el soporte del kNN; la variante con etiquetas por logprobs se compara por pares con la de JSON.
- Las tareas del alcance son hatecheck-es, inyecciones, jailbreak, massive-esc y tuits-es, con un 99–100 % de su soporte reetiquetado. massive-es queda fuera: solo tiene un 0,4 % reetiquetado.
- Diferencia agrupada en el alcance: **+1,69 puntos** [+0,72, +2,60]. Es significativa, pero no llega a +2.
- En las 6 tareas, la diferencia es +1,34 [+0,59, +2,07] (McNemar 366/277, p = 5e-4). El kNN con todo el soporte del LLM pasa de 0,6389 a 0,6522.

| tarea | kNN, soporte en JSON | kNN, soporte por logprobs | diferencia [IC95] |
|---|---|---|---|
| inyecciones | 0,560 | **0,879** | +31,9 [+24,1, +40,5] |
| massive-esc | 0,647 | **0,748** | +10,0 [+8,1, +12,0] |
| hatecheck-es | 0,730 | 0,723 | −0,7 [−1,8, +0,4] |
| jailbreak | 0,9625 | 0,9225 | −4,0 [−6,0, −2,0] |
| tuits-es | 0,585 | 0,532 | −5,4 [−8,0, −2,8] |
| massive-es (fuera del alcance) | 0,472 | 0,473 | +0,1 |

**Lectura:**
- Con etiquetas por logprobs se arreglan los dos peores casos del soporte LLM: inyecciones pasa de −20,7 a +11,2 sobre el oro y massive-esc gana +10 puntos.
- Al mismo tiempo empeoran jailbreak y tuits-es, donde las etiquetas en JSON ya eran buenas. En tuits-es, el LLM por logprobs acierta 0,640 en el test, pero su soporte rinde peor en el kNN, sin causa verificada.
- El efecto neto es positivo y significativo, aunque pequeño.

**Prequential con etiquetas por logprobs** (descriptivo, CRIT_R6): REFUTADO otra vez. Las nuevas dan 0,521, frente a 0,632 del oro estático.
- El banco del flujo lleva etiquetas por logprobs, pero el soporte estático de lexboe sigue siendo el de fondo2 en JSON, que no llegó a reetiquetarse. lexboe queda en 0,318.
- Con la regla del umbral, lexboe da 0,629, las nuevas 0,643 y las originales 0,697.
- inyecciones con fondo LLM sube a 0,88, frente a 0,57 con JSON.
- El LLM directo por logprobs da lexboe 0,534 y fakenews 0,682: por encima de cualquier kNN en fakenews y por debajo del oro en lexboe.
