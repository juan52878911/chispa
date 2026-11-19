# Batería v3, fase 2: el ud de la c7i

**Estado (W3b, sin red):** diseño, constructor, comprobación de licencias, medida de solapes, manifiesto inicial y el ud están escritos y probados en seco con datos inventados.
Nada se ha descargado ni lanzado; ninguna licencia de las 31 candidatas está verificada (todas «por verificar en fase 2», con la URL de su ficha en `bateria-v3-manifiesto.json`).
**Cambios de W3b:** (1) las huellas se hacen DENTRO del ud (ya no hay `huellas.tgz`); (2) lectores XML de `nsf-direccion` y `wikinoticias-es-categoria`; (3) bloque `informativo` para los 6 benchmarks públicos y 5 reemplazos de tarea real; (4) un solo ud con las 31 candidatas, tope 90 min.
Lo que no se puede comprobar de forma automática queda **fuera**, con su motivo; nada se verifica a mano en el Mac.

## Qué hace el ud (`model/datos/aws/bateria_v3.sh` + `bateria_v3_fases.sh`)
1. Baja `codigo.tgz` (de `bateria-v3-SHA/`), y de v2 `s3://chispa-pruebas-cuenta-aws/w1-fs8-a6c5293/dev.tgz` (6 jsonl) y `s3://chispa-pruebas-cuenta-aws/r10-apertura-29c441f-a/selladas.tgz` (8 jsonl; ya abiertas en §17.26: leerlas para huellas no las gasta),
   por nombre exacto (el rol no tiene ListBucket) y comprueba **antes de usarlas** sus sha256 (`ce0423dc…3ca9` y `a2bb334e…5a0`, fijos en el propio ud) y el del manifiesto.
2. Entorno: venv con `huggingface_hub`, `pyarrow` y `PyYAML` fijados; **comprueba imports, nº de tareas y alcance de las fuentes antes de bajar un dato**.
2b. **Huellas dentro de la c7i** (`bateria_v3_huellas.py`; salen solo hashes y sketches, los textos no salen de la instancia):
   - **v2**: extrae los dos `.tgz` (extracción segura: sin rutas absolutas, `..` ni enlaces; carpeta nueva) y falla si no hay 6 y 8 jsonl o si > 1 % de las filas no trae campo de texto. Una huella (`h` = sha256 del texto normalizado, `s` = sketch bottom-32 de 5-gramas) por texto → `huellas-v2.jsonl.gz`.
   - **Entrenamiento**: de `model/alt/manifiesto-entrenamiento.json` (va en `codigo.tgz`) se bajan las fuentes **públicas** (id de HF; con revisión si el manifiesto la trae, si no la cabecera del momento, que queda anotada): `toxicidad_protestas`, `toxicidad_constitucion`, `goemociones_es`, `spam_tanaos`, `soporte_ti`, `ciberacoso` (_1 y _2) y `tsjev`.
     Se hace la huella de **toda celda de texto ≥ 20 caracteres** de sus ficheros de datos (parquet, csv, tsv, json, jsonl): no hace falta saber qué columna entrenó, y al ser un superconjunto solo puede descartar de más, nunca de menos (ciberacoso se huella también con `@usuario`, como se entrenó).
     tsjev se baja entero (no solo las 365 tareas admitidas) con tope de 14 GB; las demás, 8 GB. El texto bajado se borra al terminar cada fuente. Tope global de 35 min (`CHISPA_V3_TOPE_HUELLAS_MIN`): lo que falte queda como «parcial».
   - **Solape no comprobado** (lo escribe `estado-huellas.json` y pasa al manifiesto, `solape_no_comprobado`, con su motivo): lo que no es un id público de HF, lo que falla al bajar o leer (sin ficheros, ilegible, gated, sin tiempo, pasa del tope), y las exclusiones del manifiesto de entrenamiento
     (SIB-200, Clandestino, PQRS de datos.gov.co, blandas de Qwen3.5, XNLI, clima, ALIA; y massive, clinc150, hatecheck e inyecciones, que son medición de v1). Comprobarlo es de Juan, en su máquina, solo con hashes (`bateria_v3_solapes huellas` + `construir --huellas`).
   - El log solo sube si no comparte **ninguna secuencia de 8 palabras** con un texto de v2 (todos) ni con una muestra de 5 000 textos por fuente del entrenamiento (además de la auditoría de las selladas de v3). Al terminar se borran los textos de v2 y la muestra.
3. `reservas-v2`: baja las tarjetas (en la revisión de v2) y las páginas de origen de promise, ropa, rtp y CFPB, las guarda con sha256 y pasa la tarjeta por la comprobación (`bateria-v3-reservas-v2.md`).
4. `construir`, por candidata: ficha (README.md o página) → **licencia automática** (`bateria_v3_licencias.py`: admite CC0, CC BY, MIT, Apache-2.0, ODC-BY, CDLA-Permissive y dominio público;
   deja fuera -SA, -NC, -ND, ausente, múltiple, contradictoria entre cabecera y cuerpo, y cualquier mención de Twitter/X o Reddit; las fuentes por URL solo pasan si su ficha trae el texto exigido)
   → descarga con la revisión de HF (o el sha del commit / el sha256 del fichero) y tope de tamaño → sha256 de cada fichero → esquema → solapes → partición → sellado.
   Guarda el texto de la licencia con su sha256 en `licencias/`.
5. Solapes (`bateria_v3_solapes.py`): hash del texto normalizado y sketch bottom-32 de n-gramas de 5 palabras contra las huellas de v2 y del entrenamiento y entre las tareas de v3.
   Hash igual o Jaccard ≥ 0,8 → se descarta la fila; 0,5–0,8 → se marca (`marca_solape`). El manifiesto trae los recuentos por fuente (`v2`, `entrenamiento`, `v3:<otra tarea>`) y `solapes.estado_huellas`.
5b. **Lectores nuevos** (escritos sin ver el esquema real; tolerantes): `xml` (NSF: un registro `<Award>` por concesión, `AbstractNarration` como texto con el HTML escapado limpio y `Organization/Directorate/LongName` como etiqueta; sin distinguir mayúsculas ni espacios de nombres; XML con `<!ENTITY` se omite)
   y `mediawiki` (volcado `.xml.bz2` de Wikinoticias: salta redirecciones y otros espacios de nombres, limpia el wikitexto —plantillas, referencias, archivos, enlaces, «Fuentes»— y etiqueta con la ÚNICA categoría temática de un conjunto cerrado que traiga el artículo). Si el esquema no encaja, la tarea sale `fuera` con las etiquetas XML vistas (validación de esquema = fase 2).
   Otras claves nuevas del lector: `separador`, `umbral` (yesno desde una nota numérica), `lista` (registros dentro de cada JSON de un zip) e índices de lista en las rutas (`Hazards.0.Name`).
5c. **Bloques**: cada tarea trae `bloque` = `cuentan` | `informativo`. Los 6 benchmarks públicos (mmmlu, bbh, gsm8k, math, strategyqa, pubmedqa; «benchmark público, probable contaminación de L3») se construyen y se sellan igual pero **no cuentan** en el agregado ni en las hipótesis selladas.
   `resumen.json` da por tipo `candidatas/construidas/cuentan/informativo/es/en` con `n_bajo` (< 8 que cuenten) y `por_idioma` (el agregado en español, con `n_bajo` si cuentan < 8).
6. Parte: soporte con k = 8 por clase (del `train` oficial si lo hay) y test ≤ 500 estratificado, con semilla 20261119; sin duplicados, sin textos con dos etiquetas, sin test que esté en el soporte;
   una tarea con < 40 filas de test o < 2 clases queda fuera. Texto ≤ 3 000 caracteres.
7. **Sella**: `sellada/<tarea>.jsonl` con permisos 0444 y su sha256 en el manifiesto; solo se lee con `abrir_sellada(ruta, motivo)`, que anota antes en `bateria-v3-aperturas.jsonl` (como v2) y se niega si el sha256 no coincide.
8. Comprueba que cada sellada coincide con el manifiesto (sha256, recuento, permisos) y **audita el log**: si algún texto de test aparece en él, `log.txt` no se sube.
9. Sube: `manifiesto.json`, `licencias.tgz` (+ `reservas-v2.json`), `datos-sellados.tgz` (soporte, sellada, manifiesto, licencias y el registro de la auditoría) y `resumen.json`, más `log.txt` (solo recuentos, estados y motivos) y `listo`; espera `verificado` y se apaga.
   El constructor solo imprime recuentos, estados de validación y motivos: nunca texto de test.

**Coste:** `shutdown -h +90`: 90 min × 0,1785 USD/h (el precio de c7i.xlarge que usa LEEME §19.3, us-east-1) = 0,268 USD + ~0,005 de volumen gp3 de 30 GB (30 × 0,08 USD/GB-mes ÷ 730 h × 1,5 h) = **~0,27 USD < 0,3**.
Lo que el tope cubre: ~10 min de entorno y reservas, ≤ 35 min de huellas (tsjev es lo grande), el resto para construir las 31 candidatas, 5 min de espera de `verificado`. El tope de 0,3 USD da como máximo ~97 min a este precio; a 0,20 USD/h serían 0,30 con 90 min: si el precio es ese, bajar a `CHISPA_MIN=80`
(0,267). Hoy cabe; si dejara de caber (precio o tsjev mucho mayor), habría que partir en dos ud (huellas + mitad de las candidatas cada uno, repitiendo las huellas: ~0,2 USD cada uno). Sin GPU, sin llamadas de pago.

## Comandos exactos
```bash
# 1. Empaquetar (no sube nada; el commit SHA debe contener model/datos/bateria_v3*, aws/bateria_v3*.sh y model/alt/manifiesto-entrenamiento.json). Ya no hay DIR_HUELLAS ni huellas.tgz
bash model/datos/aws/empaquetar_bateria_v3.sh SHA DIR_SALIDA
#    escribe DIR_SALIDA/codigo.tgz y model/datos/aws/lanzados/ud-bateria-v3-SHA.sh (sin placeholders; falla si el ud pasa de 15 500 bytes)

# 2. Subir codigo.tgz por nombre exacto (dev.tgz y selladas.tgz de v2 ya están subidos y sus sha256 van fijos en el ud)
aws s3 cp DIR_SALIDA/codigo.tgz s3://chispa-pruebas-cuenta-aws/bateria-v3-SHA/codigo.tgz

# 3. Lanzar (primero con --dry-run: debe dar DryRunOperation). Misma red, perfil y AMI que LEEME §13.4; volumen >= 30 GB. El rol del perfil debe poder leer los 3 objetos por nombre exacto
aws ec2 run-instances --region us-east-1 --image-id ami-ejemplo --instance-type c7i.xlarge --count 1 \
  --subnet-id subnet-ejemplo --security-group-ids sg-ejemplo --iam-instance-profile Name=rol-ec2-pruebas \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://model/datos/aws/lanzados/ud-bateria-v3-SHA.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-bateria-v3-SHA}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-bateria-v3-SHA}]'  # [--dry-run]

# 4. Esperar `listo` (nombre exacto; el rol no tiene ListBucket) y bajar lo que importa
aws s3 cp s3://chispa-pruebas-cuenta-aws/bateria-v3-SHA/listo -
for f in manifiesto.json resumen.json reservas-v2.json licencias.tgz datos-sellados.tgz log.txt; do aws s3 cp s3://chispa-pruebas-cuenta-aws/bateria-v3-SHA/$f DIR_RESULTADOS/$f; done
# 5. Cerrar: el ud espera `verificado` 5 min y se apaga
echo ok | aws s3 cp - s3://chispa-pruebas-cuenta-aws/bateria-v3-SHA/verificado
```
La subida y el lanzamiento los hace quien tenga la decisión de gasto (Juan): no se ha subido ni lanzado nada, ni se ha hecho `--dry-run`.

## Después de `listo` (lectura, no verificación a mano)
- `resumen.json` y `manifiesto.json`: cuántas tareas por tipo y estado (`construida`, `fuera` con `motivo_fuera`, `pendiente`), la licencia detectada, su sha256, la revisión fijada y los recuentos de solapes.
  Las revisiones que el ud fijó (`rev`) se copian al `bateria-v3-manifiesto.json` del repo para las siguientes corridas (con `rev` fijada el constructor la usa en vez de la cabecera).
- `manifiesto.json` trae `solape_no_comprobado` (con motivo) y `solapes.estado_huellas` (qué fuentes del entrenamiento se comprobaron, con su revisión y nº de huellas): lo no comprobado lo cierra Juan en su máquina, solo con hashes.
- Copiar `manifiesto.json` y `datos-sellados.tgz` es lo único que sale de la nube; el `.tgz` contiene texto de test: va a la carpeta de selladas, nunca al repo ni a un informe.
- `reservas-v2.json`: tarjeta y origen de promise, ropa, rtp y CFPB con sha256 (`bateria-v3-reservas-v2.md` lo interpreta).

## Decisiones que quedan para Juan (no son comprobaciones)
1. **Lanzar o no** (~0,27 USD).
2. **Reemplazos**: los 5 de `bateria-v3-candidatas.md` («reemplazo de») son de memoria y de confianza baja; tras los reemplazos cuentan 10 choice, 7 yesno y 8 score (en español solo 2 por tipo, 6 en total): yesno ya sale «n bajo» y cualquier fallo en fase 2 baja los otros; en español hay pocas fuentes con licencia permisiva que yo conozca. Si Juan conoce fuentes en español (tarea real, licencia admitida), se añaden a `bateria_v3_candidatas.py` y se repite con `--solo` (céntimos).
3. **Esquemas de memoria**: ids, columnas y URL de las candidatas son de memoria (los lectores XML, además, sin ver el esquema real). Las que fallen quedarán `fuera` con el motivo; se corrigen y se repite con `--solo`.
4. **Dev**: v3 sale sin partición dev (la regla del ud es subir solo lo sellado).
5. **Reservas de v2** (`bateria-v3-reservas-v2.md`).
6. **Solape no comprobado**: PQRS y las demás exclusiones del entrenamiento (ver 2b) se comprueban en la máquina de Juan, solo con hashes.

## Prueba en seco (lo que se hizo en fase 1; sin red ni AWS)
```bash
python -m pytest model/datos/tests -q        # incluye el constructor contra fixtures inventados, los lectores XML, las huellas y el ud en modo local
python -m model.datos.tests.fixtures_v3.generar /tmp/fx3        # espejo de HF, ficheros file://, dev.tgz/selladas.tgz y entrenamiento falsos
CHISPA_V3_ESPEJO=/tmp/fx3/espejo python -m model.datos.bateria_v3 construir --manifiesto /tmp/fx3/manifiesto.json --trabajo /tmp/fx3t --salida /tmp/fx3o --huellas /tmp/fx3/huellas
```
`test_bateria_v3_ud.py` empaqueta el HEAD (hay que commitear antes de probar) y corre el ud con `CHISPA_LOCAL=1` (carpeta en vez de S3, `CHISPA_V3_DEV`/`CHISPA_V3_SELLADAS` para los .tgz de v2, `CHISPA_V3_ENTRENAMIENTO` para un manifiesto de entrenamiento inventado, `CHISPA_V3_ESPEJO` en vez de la red).
