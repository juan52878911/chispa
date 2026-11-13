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

**Dónde están las entradas:** el checkpoint del hito 4 (`mejor`, sha256 `70df6ae0…53e`) **solo está en el PC de pruebas** (`~/orquestador/alt-pruebas/ck/mejor`, y su `mejor-e5-small.tgz` de 387 MB); no lo busqué en S3 porque estaba en local. M1 (`datos/m1.pt`, sha256 `4336e971…aa9c`) tampoco consta en S3 con una clave que yo conozca (el lote anterior lo bajaba de `s3://chispa-pruebas-879381269751/ajuste-80ab8ef/ckpt/mejor.pt`; no verificado). Las predicciones de este informe están en el PC (`~/orquestador/fable/arq/res/pred`) y una copia en el scratchpad de la sesión; no se versionan. e5: `intfloat/multilingual-e5-small` @ 614241f622f53c4eeff9890bdc4f31cfecc418b3.

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

## 8. P3 preparado (sin lanzar)

**Código:** `model/alt/l2x.py` (una secuencia `[CLS] query: pregunta [SEP] [OPC] desc… [SEP] texto [SEP]`, vector de cada opción = media de su tramo, cabeza sobre [opción, texto, opción⊙texto], softmax por pregunta; entrenamiento con ≤ 8 opciones barajadas con la correcta siempre dentro, torneo por trozos si hay más al evaluar), `entrenar.py --arquitectura l2x`, `l2x_onnx.py` (fp32 e int8), `bench_l2x_i7.py` (solo numpy + onnxruntime; p50/p99 con 4 hilos, 128 tokens, 2/4/8/16 opciones, cruzado y L2x), `manifiesto-l2x.json` (el del hito 4 más synzs, Apache-2.0). **No trae** las cabezas CORAL/multi, calibración, cabeza de escalada ni preselección con L1.

**Pruebas hechas (CPU, PC de pruebas):** 12 pruebas unitarias nuevas (entrada, presupuesto de tokens, torneo, sobreajuste de dos filas, guardado/carga, ONNX = PyTorch a 1e-3); humo con e5-small real (3 pasos, exportación a ONNX de 472 MB / 118 MB, predicción en torch y en ORT int8, banco); y **prueba en seco del paquete AWS desempaquetando el tarball en un directorio limpio** y corriendo el `l2x.sh` con `CHISPA_LOCAL=1` (sustituyendo `__CHISPA_S3__` como lo hará el lanzador): imports de `chispa` y `model.*` OK, 2 pasos de entrenamiento, evaluación, exportación y subida simulada de `ckpt/mejor-l2x.tgz` y `onnx/l2x_int8.onnx`. Esa prueba se hizo con el tarball de `16efc9b`; el actual (`e4025e7`) difiere solo en el cálculo del IC95 de `p0_medir` (no se usa al predecir), **sin repetir la prueba en seco** por el cambio de plan.

**Paquete:** `bash model/alt/aws/empaquetar_l2x.sh e4025e7 > codigo.tgz` (460 764 bytes, sha256 `2e9c2263eebec0ae…`; incluye `sdk/python`) y `bateria-no-vista.tgz` (515 331 bytes, sha256 `bec3ec60df67e5a7…`, las 6 tareas y el manifiesto). Ambos están en el scratchpad de la sesión, sin subir.

**Claves de S3** (todas por nombre exacto, sin `--recursive`; `CHISPA_S3 = s3://chispa-pruebas-879381269751/l2x-e4025e7`). Las leídas con `head-object` existen hoy (blandas: 16,2 MB; synzs/train-00000: 7,8 MB; tsjev/train-00000: 20,3 MB).
- Lee: `$CHISPA_S3/codigo.tgz`, `$CHISPA_S3/bateria-no-vista.tgz` (**subirlos antes**), `…/meta-20261027/meta/tsjev/{train,dev}-*.jsonl.gz` (los nombres salen de `model/meta/resultados/conversion-20261027.json`), `…/meta-20261027/meta/synzs/train-00000..00007.jsonl.gz` y `synzs/dev-00000.jsonl.gz`, `…/blandas/e2deac55256e3314/blandas.jsonl.gz`. Fuera de S3: Hugging Face (e5-small y 7 datasets del manifiesto) y el clon público de JevBench.
- Escribe: `$CHISPA_S3/{log.txt,listo,resultados.tgz,ckpt/mejor-l2x.tgz,onnx/l2x_int8.onnx,onnx/exportacion.json}`; espera `$CHISPA_S3/terminar` hasta 20 min.

**Coste máximo por ronda:** techo duro de 360 min (`shutdown -h +360`, con `terminate`). Spot g5.xlarge en us-east-1f ≈ 0,51 USD/h → ≤ 3,1 USD; con `MaxPrice=0,60` ≤ 3,6 USD; en el peor caso bajo demanda (1,006 USD/h) serían 6,0 USD, pero con MaxPrice no puede caer a bajo demanda. Más EBS de 150 GB: < 0,2 USD. Entrenamiento: 3 épocas, lote 32, bf16, hasta ~300 min; el rendimiento en la A10G **no está medido**, así que 3 épocas completas no están garantizadas (si se acaba el tiempo guarda la época en curso).

**Comando para Juan** (el perfil `chispa-pruebas-ec2` y la AMI salen de mis lecturas; la AMI y el subred los comprobé con `describe-*`, el nombre del dispositivo raíz `/dev/xvda` y el SG por defecto **no** están verificados contra una ejecución anterior de esta cuenta):

    S3=s3://chispa-pruebas-879381269751/l2x-e4025e7
    aws s3 cp codigo.tgz $S3/codigo.tgz && aws s3 cp bateria-no-vista.tgz $S3/bateria-no-vista.tgz
    sed "s|__CHISPA_S3__|$S3|" model/alt/aws/l2x.sh > user-data.sh
    aws ec2 run-instances --region us-east-1 --image-id ami-004851d2e6f24b4b4 --instance-type g5.xlarge --count 1 \
      --subnet-id subnet-0b152606d6e4aedd2 --security-group-ids sg-0caaea4ac99da8ca1 --iam-instance-profile Name=chispa-pruebas-ec2 \
      --instance-initiated-shutdown-behavior terminate \
      --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=0.60,SpotInstanceType=one-time}' \
      --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --user-data file://user-data.sh --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=chispa-l2x}]'

**Criterios de paso de P3 (de la revisión):** macro-F1 medio en massive, hatecheck e inyecciones ≥ L2 + 12 puntos y ≥ hito 4 − 1; p50 ≤ 60 ms en el i7 (int8, 4 hilos, ≤ 8 opciones); AUROC de escalar ≥ 0,78 fuera de pliegue. Se refuta si queda por debajo del hito 4 en ≥ 2 de las 3 tareas. Dos avisos míos: (a) massive e inyecciones están en el train de tsjev, así que esas tres tareas ya no miden zero-shot limpio; mejor añadir hatecheck-es y la media de la batería; (b) el int8 dinámico rompió 7,7 puntos al cross-encoder (§5), así que el int8 de L2x hay que medirlo antes de dar por buena la latencia (`p0_medir pred --modelo l2x-int8-onnx`); y la cabeza de escalada no está implementada, así que el AUROC solo se puede medir con la probabilidad máxima.
