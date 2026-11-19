# Batería v3, fase 2: el ud de la c7i

**Estado (fin de la fase 1, sin red):** diseño, constructor, comprobación de licencias, medida de solapes, manifiesto inicial y el ud están escritos y probados en seco con datos inventados.
Nada se ha descargado ni lanzado; ninguna licencia de las 24 candidatas está verificada (todas «por verificar en fase 2», con la URL de su ficha en `bateria-v3-manifiesto.json`).
Lo que no se puede comprobar de forma automática queda **fuera**, con su motivo; nada se verifica a mano en el Mac.

## Qué hace el ud (`model/datos/aws/bateria_v3.sh` + `bateria_v3_fases.sh`)
1. Baja `codigo.tgz` y `huellas.tgz` por nombre exacto y comprueba sus sha256 y el del manifiesto.
2. Entorno: venv con `huggingface_hub`, `pyarrow` y `PyYAML` fijados; **comprueba imports, nº de tareas y alcance de las fuentes antes de bajar un dato**.
3. `reservas-v2`: baja las tarjetas (en la revisión de v2) y las páginas de origen de promise, ropa, rtp y CFPB, las guarda con sha256 y pasa la tarjeta por la comprobación (`bateria-v3-reservas-v2.md`).
4. `construir`, por candidata: ficha (README.md o página) → **licencia automática** (`bateria_v3_licencias.py`: admite CC0, CC BY, MIT, Apache-2.0, ODC-BY, CDLA-Permissive y dominio público;
   deja fuera -SA, -NC, -ND, ausente, múltiple, contradictoria entre cabecera y cuerpo, y cualquier mención de Twitter/X o Reddit; las fuentes por URL solo pasan si su ficha trae el texto exigido)
   → descarga con la revisión de HF (o el sha del commit / el sha256 del fichero) y tope de tamaño → sha256 de cada fichero → esquema → solapes → partición → sellado.
   Guarda el texto de la licencia con su sha256 en `licencias/`.
5. Solapes (`bateria_v3_solapes.py`): hash del texto normalizado y sketch bottom-32 de n-gramas de 5 palabras contra las huellas de v2 y del entrenamiento y entre las tareas de v3.
   Hash igual o Jaccard ≥ 0,8 → se descarta la fila; 0,5–0,8 → se marca (`marca_solape`). El manifiesto trae los recuentos por fuente.
6. Parte: soporte con k = 8 por clase (del `train` oficial si lo hay) y test ≤ 500 estratificado, con semilla 20261119; sin duplicados, sin textos con dos etiquetas, sin test que esté en el soporte;
   una tarea con < 40 filas de test o < 2 clases queda fuera. Texto ≤ 3 000 caracteres.
7. **Sella**: `sellada/<tarea>.jsonl` con permisos 0444 y su sha256 en el manifiesto; solo se lee con `abrir_sellada(ruta, motivo)`, que anota antes en `bateria-v3-aperturas.jsonl` (como v2) y se niega si el sha256 no coincide.
8. Comprueba que cada sellada coincide con el manifiesto (sha256, recuento, permisos) y **audita el log**: si algún texto de test aparece en él, `log.txt` no se sube.
9. Sube: `manifiesto.json`, `licencias.tgz` (+ `reservas-v2.json`), `datos-sellados.tgz` (soporte, sellada, manifiesto, licencias y el registro de la auditoría) y `resumen.json`, más `log.txt` (solo recuentos, estados y motivos) y `listo`; espera `verificado` y se apaga.
   El constructor solo imprime recuentos, estados de validación y motivos: nunca texto de test.

**Coste:** `shutdown -h +75`: 75 min × 0,1785 USD/h (el precio de c7i.xlarge que usa LEEME §19.3, us-east-1) = 0,223 USD + ~0,01 de volumen = **~0,23 USD < 0,3**. Si la región o el precio cambian, recalcular: N = 0,3 / precio_h × 60 − margen
(a 0,20 USD/h: 75 min = 0,25 USD). Sin GPU, sin llamadas de pago.

## Comandos exactos
```bash
# 0. Huellas de v2 y del entrenamiento, en la máquina donde viven los datos (salen SOLO hashes y sketches; nada de texto)
python -m model.datos.bateria_v3_solapes huellas --entrada DIR_BATERIA_V2/dev DIR_BATERIA_V2/selladas --salida huellas/huellas-v2.jsonl.gz --fuente v2
python -m model.datos.bateria_v3_solapes huellas --entrada DIR_ENTRENAMIENTO_JSONL_GZ --salida huellas/huellas-entrenamiento.jsonl.gz --fuente entrenamiento
#    (opcional, mismo comando: la batería v1 y las tareas nuevas, como huellas-v1.jsonl.gz y huellas-tareas-nuevas.jsonl.gz)
#    Cada comando imprime filas_sin_campo_de_texto y falla si pasan del 1 %. v2 selladas ya se abrieron (§17.26): leerlas para sacar huellas no las gasta.

# 1. Empaquetar (no sube nada; el commit SHA debe contener model/datos/bateria_v3* y aws/bateria_v3*.sh)
bash model/datos/aws/empaquetar_bateria_v3.sh SHA huellas DIR_SALIDA
#    escribe DIR_SALIDA/{codigo.tgz,huellas.tgz} y model/datos/aws/lanzados/ud-bateria-v3-SHA.sh; falla si las huellas traen texto o el ud pasa de 15 500 bytes

# 2. Subir los dos tgz por nombre exacto
aws s3 cp DIR_SALIDA/codigo.tgz  s3://chispa-pruebas-879381269751/bateria-v3-SHA/codigo.tgz
aws s3 cp DIR_SALIDA/huellas.tgz s3://chispa-pruebas-879381269751/bateria-v3-SHA/huellas.tgz

# 3. Lanzar (primero con --dry-run: debe dar DryRunOperation). Misma red, perfil y AMI que LEEME §13.4; volumen >= 30 GB
aws ec2 run-instances --region us-east-1 --image-id ami-004851d2e6f24b4b4 --instance-type c7i.xlarge --count 1 \
  --subnet-id subnet-0b152606d6e4aedd2 --security-group-ids sg-0caaea4ac99da8ca1 --iam-instance-profile Name=chispa-pruebas-ec2 \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://model/datos/aws/lanzados/ud-bateria-v3-SHA.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-bateria-v3-SHA}]' 'ResourceType=volume,Tags=[{Key=proyecto,Value=chispa},{Key=Name,Value=chispa-bateria-v3-SHA}]'  # [--dry-run]

# 4. Esperar `listo` (nombre exacto; el rol no tiene ListBucket) y bajar lo que importa
aws s3 cp s3://chispa-pruebas-879381269751/bateria-v3-SHA/listo -
for f in manifiesto.json resumen.json reservas-v2.json licencias.tgz datos-sellados.tgz log.txt; do aws s3 cp s3://chispa-pruebas-879381269751/bateria-v3-SHA/$f DIR_RESULTADOS/$f; done
# 5. Cerrar: el ud espera `verificado` 5 min y se apaga
echo ok | aws s3 cp - s3://chispa-pruebas-879381269751/bateria-v3-SHA/verificado
```
La subida y el lanzamiento los hace quien tenga la decisión de gasto (Juan): en fase 1 no se ha subido ni lanzado nada, ni se ha hecho `--dry-run`.

## Después de `listo` (lectura, no verificación a mano)
- `resumen.json` y `manifiesto.json`: cuántas tareas por tipo y estado (`construida`, `fuera` con `motivo_fuera`, `pendiente`), la licencia detectada, su sha256, la revisión fijada y los recuentos de solapes.
  Las revisiones que el ud fijó (`rev`) se copian al `bateria-v3-manifiesto.json` del repo para las siguientes corridas (con `rev` fijada el constructor la usa en vez de la cabecera).
- Copiar `manifiesto.json` y `datos-sellados.tgz` es lo único que sale de la nube; el `.tgz` contiene texto de test: va a la carpeta de selladas, nunca al repo ni a un informe.
- `reservas-v2.json`: tarjeta y origen de promise, ropa, rtp y CFPB con sha256 (`bateria-v3-reservas-v2.md` lo interpreta).

## Decisiones que quedan para Juan (no son comprobaciones)
1. **Lanzar o no** (0,23 USD) y con qué huellas (¿solo v2 y entrenamiento, o también v1 y tareas nuevas?).
2. **Lectores pendientes**: `nsf-direccion` (XML) y `wikinoticias-es-categoria` (volcado MediaWiki) no tienen lector: el ud las deja `pendiente`. Escribirlos exige ver el esquema real.
   Sin ellas, choice queda con 6 y la parte en español de v3 es mínima.
3. **Esquemas de memoria**: ids, columnas y URL de las 24 candidatas son de memoria. Las que fallen quedarán `fuera` con el motivo (no existe, sin ficheros, columnas que faltan, 404): se corrigen en `bateria_v3_candidatas.py` y se repite con `--solo` (sale por unos céntimos).
4. **Dev**: v3 sale sin partición dev (la regla del ud es subir solo lo sellado). Si se quiere un dev para ajustar, hay que decidirlo antes de sellar, con la misma semilla y solo con metadatos.
5. **Contaminación**: mmmlu, bbh, gsm8k, math, strategyqa y pubmedqa son benchmarks públicos: probables en el preentrenamiento de los LLM; el manifiesto lo marca en `riesgos`.
6. **Reservas de v2** (`bateria-v3-reservas-v2.md`).

## Prueba en seco (lo que se hizo en fase 1; sin red ni AWS)
```bash
python -m pytest model/datos/tests -q        # incluye el constructor contra fixtures inventados y el ud en modo local
python -m model.datos.tests.fixtures_v3.generar /tmp/fx3        # espejo de HF, ficheros file:// y huellas inventados
CHISPA_V3_ESPEJO=/tmp/fx3/espejo python -m model.datos.bateria_v3 construir --manifiesto /tmp/fx3/manifiesto.json --trabajo /tmp/fx3t --salida /tmp/fx3o --huellas /tmp/fx3/huellas
```
`test_bateria_v3_ud.py` empaqueta el HEAD y corre el ud con `CHISPA_LOCAL=1` (carpeta en vez de S3, `CHISPA_V3_ESPEJO` en vez de la red).
