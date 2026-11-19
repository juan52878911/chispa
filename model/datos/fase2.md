# Batería v3, fase 2: el ud de la c7i

**Estado (W3c, segunda y última iteración, sin red):** la fase 2 real (c7i, ud `100de63`, 299 s) construyó 12 de 31 y dejó 19 fuera; resultados en `tests/fixtures_v3/fase2-100de63/`.
Esta iteración arregla lo arreglable, saca las licencias y descargas federales del registro oficial (catalog.data.gov), propone candidatas nuevas (español y sí/no primero) y prepara **un solo ud** que rehace la fase 2 entera desde cero (huellas incluidas; no se mezclan dos corridas).
El ud se prepara y **NO se lanza**. Si tras la corrida algún tipo queda bajo 8, se informa «n bajo» (abrir igual es decisión de Juan): no se rellena.
Lo que no se puede comprobar de forma automática queda **fuera**, con su motivo; nada se verifica a mano en el Mac.

## W3c: qué cambió respecto a la fase 2 real
**Se quedan fuera (ya no son candidatas; pasan a descartadas con el motivo real):** `patentes-cpc` (31,8 GB), `wikinoticias-es-categoria` (-SA), `argq-calidad` (-SA), `bbh-deportes` y `sts-es` (tarjeta sin licencia).
**Lectores (todos probados en `tests/test_bateria_v3_lectores.py` con las columnas que registró el `log.txt` y las fichas reales del fixture):**
- Nombres de columna **tolerantes** (mayúsculas, acentos, espacios y signos: «Label 1» = `label1`; «Título» = `titulo`) y alternativas `a||b`: arregla `biotech-evento` (Content/Label 1/Title) y `fake-news-es`.
- Delimitador: regla escrita; si la cabecera sale en UNA columna con tabulador, «|» o «;» dentro se relee con ese delimitador: arregla `unam-tesis-carrera` (`texto|titulo|carrera`, además con `separador="|"` explícito) y los tsv guardados como `.csv` de `wnli-es`.
- **JSON** (`JSONDecodeError … line 2 column 1` de `wnli-es` y `strategyqa-si-no`: eran objetos de varias líneas): se lee jsonl línea a línea y, si la primera línea no es un JSON completo, el fichero entero (lista, `{"data": [...]}` o objetos uno tras otro).
- **Ficheros con otro esquema** (`moral-stories-accion`: el repo trae varias configuraciones y se leyó una con solo ID, label y moral_action): cada fichero se comprueba por separado y se **salta** el que no trae las columnas (se cuenta en `ficheros_omitidos_por_esquema`); si no queda ninguno, el motivo trae nombres y **tipos** de columna de cada fichero (nunca texto).
- `fake-news-es`: la ficha real cita el corpus de Posadas-Durán/FakeDeS y las columnas son `CATEGORY, HEADLINE, TEXT, TOPICS, SOURCE, LINK, ID`: es el corpus que se quería (verdadero/falso); se lee `CATEGORY` (Fake→«si», True→«no», la pregunta sigue siendo «¿es falsa?») con `HEADLINE` + `TEXT`; si `CATEGORY` trae otros valores, el motivo los lista (son etiquetas, no texto).
- `wnli-es`: la tarjeta cita tres ficheros (train, dev, test sin etiquetas): se usan train y dev (`**/*train*`, `**/*dev*`, `**/*valid*`), sin partición oficial.
- **Columnas automáticas** (`auto`, para las candidatas nuevas de esquema desconocido; regla escrita en `_auto_columnas`): etiqueta = primera columna (orden del fichero) cuyo nombre casa con `etiqueta_regex`, que no es id/idioma/fecha/enlace y trae 2–30 valores cortos; texto = columnas de cadena con media >= 30 caracteres y >= la mitad de la mayor (máx. 2). Lo elegido queda en el manifiesto (`esquema_elegido`) y en `resumen.json` (`esquemas_auto`); si ninguna columna sirve, fuera con nombres y tipos. En los sí/no el mapa `simple/complejo`… es explícito: **no se adivina la polaridad** de un 0/1.
- `filtro` ya no distingue mayúsculas; los mensajes de chat (`{"role","content"}`) se aplanan a su contenido; `mapa_etiqueta` es tolerante a mayúsculas y, si no cubre nada, el motivo lista los valores sin mapa.
**Fuente `ckan` (registro oficial en catalog.data.gov, API CKAN):** `package_show` por nombre o `package_search` por texto (título con `titulo_regex`, recurso con `recurso_regex`; el primero por relevancia). Admite `license_id` ∈ {`us-pd`, `other-pd`, `odc-pddl` (dominio público), `cc-zero` (CC0), `cc-by`, `odc-by`, `mit`, `apache-2.0`}; fuera con motivo: `other-open`, `other-at`, `license-not-specified`, SA/NC/ND en el título o la URL de la licencia o en `rights`, `accessLevel` no público, origen Twitter/X o Reddit, ningún paquete o ningún recurso descargable.
La **descarga** sale de `resources[].url` del paquete (un directorio se resuelve con `indice`+`patron`); si el recurso es una página o una API y la tarea trae `descarga`, esa URL solo vale si es del **mismo dominio** que algún recurso del catálogo. No se fuerza el user-agent: si la agencia devuelve 403, queda fuera con ese motivo.
Se guarda en `licencias/<tarea>.txt` el registro reducido (license_id, título y URL de la licencia, organización, `metadata_modified`, recurso, descarga) con su sha256 (`licencia_sha256`), `ckan_license_id` en el manifiesto y una línea `CKAN … license_id=…` en el log; `resumen.json` trae `licencias_catalogo`.
Las ocho federales (`tormentas-tipo`, `nhtsa-componente`, `neiss-parte-cuerpo`, `nsf-direccion`, `cpsc-recall-peligro`, `vaers-hospitalizado`, `osha-hospitalizado`, `ctgov-fase`) usan esta fuente (los nombres de paquete se buscan por texto: de memoria, sin verificar); `msha-gravedad` y `fda-clase-retirada` ya pasaron por su ficha y no cambian.
**Huellas de tsjev:** ya no se salta por tamaño (134,5 GB): se baja, se huella y se **borra fichero a fichero** (el disco aguanta el mayor, ≤ 40 GB, no la suma), con cupo de 4 M de huellas y el tope de tiempo de las huellas (45 min). Lo que no dé tiempo cuenta como «sin tiempo» y la fuente queda **parcial** en `solape_no_comprobado`.
**Candidatas nuevas (11, de memoria, confianza baja salvo helpsteer3):** español `readability-es` (sí/no), `arena-preferencia-es` (sí/no), `helpsteer3-es` (score), `cares-capitulo`, `humset-sector-es`, `noticias-es-categoria` (choice); inglés `ethics-justicia` (sí/no), `news-category-en`, `worldbank-sector`, `grants-categoria` (choice), `fsis-clase` (score). No recuerdo más fuentes con licencia permisiva y tarea real que cumplan las reglas: son 11 y no 20; si algo falla queda fuera con su motivo.

**Recuento (37 candidatas: cuentan / informativo; es):** choice 14 / 1; yesno 10 / 2; score 8 / 2; en español 10 candidatas que cuentan.
Ya confirmado en la fase 2 real: choice 1, yesno 2, score 5 (es 1). Con los lectores arreglados (todas con tarjeta admitida): choice 3, yesno 5 (es 3 en total con unam, fake-news, wnli y oasst1 → 4).
**Techo si todo pasa** (cuentan): choice 14, yesno 10, score 8, español 10. **Riesgo real**: score necesita las tres nuevas/federales (`ctgov-fase`, `helpsteer3-es`, `fsis-clase`) para llegar a 8 y es la más probable de quedar «n bajo»; yesno y choice dependen de que las federales no den 403 y de las nuevas de memoria; español necesita 8 de 10. El informe «n bajo» sale solo en `resumen.json`; no se rellena.

## Qué hace el ud (`model/datos/aws/bateria_v3.sh` + `bateria_v3_fases.sh`)
1. Baja `codigo.tgz` (de `bateria-v3-SHA/`), y de v2 `s3://chispa-pruebas-879381269751/w1-fs8-a6c5293/dev.tgz` (6 jsonl) y `s3://chispa-pruebas-879381269751/r10-apertura-29c441f-a/selladas.tgz` (8 jsonl; ya abiertas en §17.26: leerlas para huellas no las gasta),
   por nombre exacto (el rol no tiene ListBucket) y comprueba **antes de usarlas** sus sha256 (`ce0423dc…3ca9` y `a2bb334e…5a0`, fijos en el propio ud) y el del manifiesto.
2. Entorno: venv con `huggingface_hub`, `pyarrow` y `PyYAML` fijados; **comprueba imports, nº de tareas y alcance de las fuentes antes de bajar un dato**.
2b. **Huellas dentro de la c7i** (`bateria_v3_huellas.py`; salen solo hashes y sketches, los textos no salen de la instancia):
   - **v2**: extrae los dos `.tgz` (extracción segura: sin rutas absolutas, `..` ni enlaces; carpeta nueva) y falla si no hay 6 y 8 jsonl o si > 1 % de las filas no trae campo de texto. Una huella (`h` = sha256 del texto normalizado, `s` = sketch bottom-32 de 5-gramas) por texto → `huellas-v2.jsonl.gz`.
   - **Entrenamiento**: de `model/alt/manifiesto-entrenamiento.json` (va en `codigo.tgz`) se bajan las fuentes **públicas** (id de HF; con revisión si el manifiesto la trae, si no la cabecera del momento, que queda anotada): `toxicidad_protestas`, `toxicidad_constitucion`, `goemociones_es`, `spam_tanaos`, `soporte_ti`, `ciberacoso` (_1 y _2) y `tsjev`.
     Se hace la huella de **toda celda de texto ≥ 20 caracteres** de sus ficheros de datos (parquet, csv, tsv, json, jsonl): no hace falta saber qué columna entrenó, y al ser un superconjunto solo puede descartar de más, nunca de menos (ciberacoso se huella también con `@usuario`, como se entrenó).
     tsjev (134,5 GB; entero, no solo las 365 tareas admitidas) se baja y huella fichero a fichero borrando cada uno (ver W3c); las demás, tope de 8 GB. El texto bajado se borra al terminar cada fuente. Tope global de 45 min (`CHISPA_V3_TOPE_HUELLAS_MIN`): lo que falte queda como «parcial».
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

**Coste:** `shutdown -h +90`: 90 min × 0,1785 USD/h (el precio de c7i.xlarge que usa LEEME §19.3, us-east-1) = 0,268 USD + ~0,013 de volumen gp3 de 80 GB (80 × 0,08 USD/GB-mes ÷ 730 h × 1,5 h) = **~0,281 USD < 0,3** (la corrida real de la fase 2 duró 299 s con 31 candidatas; el tope solo cubre a tsjev y a las descargas federales).
Lo que el tope cubre: ~10 min de entorno y reservas, ≤ 45 min de huellas (tsjev es lo grande y quedará parcial), el resto para construir las 37 candidatas, 5 min de espera de `verificado`. El tope de 0,3 USD da como máximo ~97 min a este precio; a 0,20 USD/h serían 0,30 con 90 min: si el precio es ese, bajar a `CHISPA_MIN=80`
(0,267). Hoy cabe; si dejara de caber (precio o tsjev mucho mayor), habría que partir en dos ud (huellas + mitad de las candidatas cada uno, repitiendo las huellas: ~0,2 USD cada uno). Sin GPU, sin llamadas de pago.
**tsjev completo no cabe en 0,3 USD con garantía:** son 134,5 GB y se huella toda celda de texto >= 20 caracteres con 4 vCPU (un solo proceso Python): aunque el disco de 80 GB aguanta (un fichero cada vez) y la descarga entra en unos minutos, la lectura y los hashes de todo no caben en los 45 min de huellas. Saldrá **parcial** y quedará en `solape_no_comprobado`.
Propuestas: (a) aceptar el parcial y cerrar lo que falta en la máquina de Juan, solo con hashes (`bateria_v3_solapes huellas` + `construir --huellas`); (b) un ud aparte solo de huellas de tsjev: en una c7i.4xlarge (16 vCPU, ~0,71 USD/h) cuesta más de 0,3 USD con una hora, así que no cabe en el tope; (c) bajar solo los ficheros de las 365 tareas admitidas (la mitad de las filas), pero hace falta el mapa tarea → fichero del repo, que no se conoce y que el ud podría inspeccionar en una corrida futura.

## Comandos exactos
```bash
# 1. Empaquetar (no sube nada; el commit SHA debe contener model/datos/bateria_v3*, aws/bateria_v3*.sh y model/alt/manifiesto-entrenamiento.json). Ya no hay DIR_HUELLAS ni huellas.tgz
bash model/datos/aws/empaquetar_bateria_v3.sh SHA DIR_SALIDA
#    escribe DIR_SALIDA/codigo.tgz y model/datos/aws/lanzados/ud-bateria-v3-SHA.sh (sin placeholders; falla si el ud pasa de 15 500 bytes) y deja al lado una copia, lanzados/codigo-bateria-v3-SHA.tgz: el tgz no sale igual byte a byte en cada empaquetado y el ud lleva el sha256 del suyo, así que se sube ESE

# 2. Subir codigo.tgz por nombre exacto (dev.tgz y selladas.tgz de v2 ya están subidos y sus sha256 van fijos en el ud)
aws s3 cp model/datos/aws/lanzados/codigo-bateria-v3-SHA.tgz s3://chispa-pruebas-879381269751/bateria-v3-SHA/codigo.tgz

# 3. Lanzar (primero con --dry-run: debe dar DryRunOperation). Misma red, perfil y AMI que LEEME §13.4; volumen de 80 GB. El rol del perfil debe poder leer los 3 objetos por nombre exacto
aws ec2 run-instances --region us-east-1 --image-id ami-004851d2e6f24b4b4 --instance-type c7i.xlarge --count 1 \
  --subnet-id subnet-0b152606d6e4aedd2 --security-group-ids sg-0caaea4ac99da8ca1 --iam-instance-profile Name=chispa-pruebas-ec2 \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://model/datos/aws/lanzados/ud-bateria-v3-SHA.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-bateria-v3-SHA}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-bateria-v3-SHA}]'  # [--dry-run]

# 4. Esperar `listo` (nombre exacto; el rol no tiene ListBucket) y bajar lo que importa
aws s3 cp s3://chispa-pruebas-879381269751/bateria-v3-SHA/listo -
for f in manifiesto.json resumen.json reservas-v2.json licencias.tgz datos-sellados.tgz log.txt; do aws s3 cp s3://chispa-pruebas-879381269751/bateria-v3-SHA/$f DIR_RESULTADOS/$f; done
# 5. Cerrar: el ud espera `verificado` 5 min y se apaga
echo ok | aws s3 cp - s3://chispa-pruebas-879381269751/bateria-v3-SHA/verificado
```
La subida y el lanzamiento los hace quien tenga la decisión de gasto (Juan): no se ha subido ni lanzado nada, ni se ha hecho `--dry-run`.

## Después de `listo` (lectura, no verificación a mano)
- `resumen.json` y `manifiesto.json`: cuántas tareas por tipo y estado (`construida`, `fuera` con `motivo_fuera`, `pendiente`), la licencia detectada, su sha256, la revisión fijada y los recuentos de solapes.
  Las revisiones que el ud fijó (`rev`) se copian al `bateria-v3-manifiesto.json` del repo para las siguientes corridas (con `rev` fijada el constructor la usa en vez de la cabecera).
- `manifiesto.json` trae `solape_no_comprobado` (con motivo) y `solapes.estado_huellas` (qué fuentes del entrenamiento se comprobaron, con su revisión y nº de huellas): lo no comprobado lo cierra Juan en su máquina, solo con hashes.
- Copiar `manifiesto.json` y `datos-sellados.tgz` es lo único que sale de la nube; el `.tgz` contiene texto de test: va a la carpeta de selladas, nunca al repo ni a un informe.
- `reservas-v2.json`: tarjeta y origen de promise, ropa, rtp y CFPB con sha256 (`bateria-v3-reservas-v2.md` lo interpreta).

## Decisiones que quedan para Juan (no son comprobaciones)
1. **Lanzar o no** (~0,28 USD).
2. **«n bajo»**: es la última iteración; si tras la corrida un tipo (o el español) queda con menos de 8 que cuenten, se informa «n bajo» y abrir igual o no es decisión de Juan; no se rellena con benchmarks públicos ni se hace una tercera ronda de candidatas de memoria.
3. **Esquemas de memoria**: ids, columnas, licencias y nombres de paquete del catálogo de las candidatas nuevas y federales son de memoria. Las que fallen quedan `fuera` con el motivo (las de `auto` con nombres y tipos de columna).
4. **Dev**: v3 sale sin partición dev (la regla del ud es subir solo lo sellado).
5. **Reservas de v2** (`bateria-v3-reservas-v2.md`).
6. **Solape no comprobado**: PQRS y las demás exclusiones del entrenamiento (ver 2b) se comprueban en la máquina de Juan, solo con hashes.

## Prueba en seco (sin red ni AWS)
```bash
python -m pytest model/datos/tests -q        # incluye el constructor contra fixtures inventados, los lectores XML, las huellas y el ud en modo local
python -m model.datos.tests.fixtures_v3.generar /tmp/fx3        # espejo de HF, ficheros file://, dev.tgz/selladas.tgz y entrenamiento falsos
CHISPA_V3_ESPEJO=/tmp/fx3/espejo python -m model.datos.bateria_v3 construir --manifiesto /tmp/fx3/manifiesto.json --trabajo /tmp/fx3t --salida /tmp/fx3o --huellas /tmp/fx3/huellas
```
`test_bateria_v3_ud.py` empaqueta el HEAD (hay que commitear antes de probar) y corre el ud con `CHISPA_LOCAL=1` (carpeta en vez de S3, `CHISPA_V3_DEV`/`CHISPA_V3_SELLADAS` para los .tgz de v2, `CHISPA_V3_ENTRENAMIENTO` para un manifiesto de entrenamiento inventado, `CHISPA_V3_ESPEJO` en vez de la red).
