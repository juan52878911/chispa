# Latencia de L0-L2 de Chispa con 1 vCPU y ~1,8 GB (sustituto de Lambda)

Mide cuánto tardan L0, L1 y L2 (int8) y la cadena L0 → L1 → L2 dentro de `docker run --cpus=1 --memory=1800m --memory-swap=1800m`.
Es un **sustituto**: Lambda real queda fuera. Corre en una EC2 con docker (c7i.2xlarge, DLAMI AL2023); no crea nada en AWS.

    bash correr.sh -d ./artefactos [-t textos.jsonl] [-n 400] [-r 5]     # deja salida-AAAAMMDD-HHMMSS/latencia-1vcpu.json

## Qué es cada capa y qué sirve hoy
Servidor y SDK son Go; este banco es Python (ORT + numpy + tokenizers) sobre **los mismos grafos y pesos**, así que mide el coste del cómputo,
no el del servidor Go ni el de HTTP. Fuente: `docs/04-formato-chispa.md`, `model/l2/README.md`, `model/l1/README.md`, `format/chispa3/l0tf.go`.

| Capa | Qué es | Dónde vive en el `.chispa` v3 | Coste esperado (doc/medidas previas, otra CPU) |
|---|---|---|---|
| L0 | TF-IDF de caracteres (`char_wb` 2-5, sublineal) + regresión logística; sin ONNX | sección `L0TF` (8); la `L0` (2) es el formato lineal de kindling v2 | ≈ 0,2 ms p99 (c6a, `docs/07`) |
| L1 | embedding estático int8 (Model2Vec `minishlab/potion-multilingual-128M`, 500 353 × 256, 155 MB) + coseno contra las opciones | sección `L1` (3), formato `.chispa-l1` | 25 µs p50 (`Embed`, 16 tokens); L1 por HTTP: sin medir |
| L2 | `intfloat/multilingual-e5-small` congelado + ranuras (K=16) entrenadas por nosotros, ONNX int8 (ids → S), MaxSim contra las torres de `ESQUEMAS` | sección `L2` (4) | 20 ms p99 a 64 tokens (AVX2, 1 hilo, fase 2 T2.6); móvil int8: cascada 28 ms p50 |
| torres al vuelo | el mismo e5 + `Torre` (MLP) en ONNX int8, para opciones de un esquema que no está en `ESQUEMAS` (zero-shot) | sección `TORR` (9) | sin medir en servidor |

Lo que **no** es L0-L2 y no se mide aquí: el cross-encoder (`model/alt/onnx_ce.py`) y L2x (`model/alt/l2x_onnx.py`) de P2/P3, que son alternativas de `model/alt`
(P2: int8 con 4 hilos, 4 opciones, p50 92,5 ms: no pasa; L2x ≈ 48-50 ms con 128 tokens). Se miden con `model/alt/bench_l2x_i7.py`.

## Entradas exactas (directorio `-d`, montado solo lectura en `/artefactos`)
| Fichero | Para qué | De dónde sale |
|---|---|---|
| `modelo-int8.chispa` | L1, L2 y TORR (y L0 si trae `L0TF`) | **Solo existe como producto de `model/bench/movil/aws/modelos.sh`** (perfil `movil-int8`, checkpoint `s3://chispa-pruebas-cuenta-aws/ajuste-80ab8ef/ckpt/mejor.pt`, 4,1 MB, entrenado por nosotros). Debía quedar en `$CHISPA_S3/movil/modelo-int8.chispa` (prefijo `movil-AAAAMMDD`; **la fecha no está en el repo**). Verificación: 249 716 640 bytes, sha256 `995dd6a7276a7026ce12ad3e04a5324714f62dc05a1c20b11141a580c30db6f4` (`model/bench/movil/resultados-20261102-210131/manifiesto.json`). Con `CHISPA_S3` el script lo baja por clave exacta. |
| `tokenizer-e5.json` | tokenizar para L2 y torres (ids de e5; el ONNX poda el vocabulario por dentro) | `tokenizer.json` de `intfloat/multilingual-e5-small` (HF, MIT según su ficha; el inventario del repo lo marca «sin verificar»). Con `CHISPA_E5_REV` (40 hex) el script lo baja de `…/resolve/<rev>/`. |
| `tokenizer-potion.json` | tokenizar para L1 | `tokenizer.json` de `minishlab/potion-multilingual-128M` (HF, MIT según su ficha; «sin verificar» en `docs/licencias-inventario.md`). Con `CHISPA_POTION_REV`. |
| `textos.jsonl` (opcional, `-t`) | textos reales (campo `texto`) | el que se pase; no hay datos en el repo. Sin él, textos sintéticos (lognormal, mediana 18 palabras, 3-120) |

Ninguna revisión de HF está fijada en el repo: hay que fijarlas al preparar la instancia (el JSON guarda el sha256 de cada fichero en `artefactos.sha256`).
No hace falta `e5.chispa-tok` (es el formato del tokenizador Go) ni la torre suelta `torre-int8.onnx` (es la misma que `TORR`).

**Qué se omite y por qué.** `modelo-int8.chispa` del perfil móvil trae `L0` de kindling v2, que este banco no lee, y no `L0TF`: **L0 sale `omitido`** salvo que se
dé con `--l0tf` un `.chispa` hecho por `chispa train --conjunta` (trae `L0TF`) o la sección suelta. Mientras, cuente L0 como ≈ 0,2 ms. Cualquier capa sin fichero o sin
tokenizador queda en `omitidos` con el motivo y el resto se mide. Los vectores de las opciones y de las torres son sintéticos (no se leen `ESQUEMAS` ni `CALIB`): el coste
es el real, la decisión no.

## Qué mide
- **Frío** (`-r` arranques, 5 por defecto; cada uno es un `docker run` nuevo con la caché de páginas vaciada si hay permiso): `tiempo_hasta_primera_respuesta_desde_arranque_proceso_ms`
  (intérprete + imports + carga de las capas + 1.ª petición completa), `carga_ms` por capa, `primera_peticion_ms` y `segunda_peticion_ms` por capa, y `docker_run_entero_ms` (host: crear el
  contenedor + todo lo anterior). Sin vaciar la caché es optimista.
- **Caliente** (`-n` ≥ 300 peticiones distintas tras 20 de calentamiento, 1 hilo intra-op, lote 1, sin relleno): p50/p95/p99/media/máx por `L0`, `L1`, `L2`, `torres`, `cadena`
  (L0+L1+L2 las tres: **peor caso**; la cascada real para en la primera que acepta, así que L0 aceptando cuesta solo L0) y `cadena_zero_shot` (L1+L2+torres de `-k` opciones nuevas).
  Las cifras por capa salen de las mismas peticiones que la cadena. `L2_por_tokens` separa L2 por longitud (≤16, ≤32, ≤64, >64 tokens; tope `-m`, 128 por defecto como `MAX_TOK`; el checkpoint G.8 usa 256).
- **Memoria**: `rss_pico_mb` (`ru_maxrss` del proceso) y `cgroup_al_final` (`memory.peak`, `memory.max`, `cpu.max`) para confirmar que el límite se aplicó.
- Entorno: CPU, carga de la instancia antes y después, versiones, digest de la imagen base, flags exactos.

## Cómo leerlo frente a Lambda
- **CPU:** Lambda da 1 vCPU completa a ~1769 MB; `--cpus=1` es una cuota CFS (100 ms de periodo), parecida en espíritu pero no idéntica. La c7i (Sapphire Rapids, AVX-512/VNNI) probablemente
  es más rápida que el hardware de Lambda por vCPU: tome las cifras como cota **optimista** de cómputo. El contenedor ve todos los núcleos de la instancia, pero ORT y tokenizers van con 1 hilo.
- **Memoria:** `-m 1800m` son 1800 MiB (≈ 1,89 GB); Lambda con 1769 MB tiene un poco menos y su runtime consume parte. Compare `rss_pico_mb` con 1769 y deje margen (≥ 15 %).
  Si `caliente_rc` es 137, el contenedor murió por memoria y eso ya es la respuesta.
- **Arranque en frío:** Lambda suma a lo medido aquí la descarga/descompresión de la imagen o el zip (155 MB de L1 + ~100 MB de ONNX + e5), que aquí no están (el `.chispa` ya es local) y que
  no se reproducen. Use `desde_arranque_proceso_ms` como el *init* del código y súmele el arranque de la plataforma por su cuenta. Con el modelo en `/var/task` el primer acceso a L1 (mmap de 155 MB)
  paga fallos de página: por eso hay que vaciar la caché de páginas.
- **Quién manda en la latencia:** L2 (e5 completo por consulta) y, en zero-shot, las torres (un lote de `-k` secuencias por esquema nuevo). L0 y L1 son órdenes de magnitud menores.
  Para un presupuesto de 60 ms mire `L2.p99_ms` y `cadena.p99_ms`; para esquemas nuevos, `cadena_zero_shot.p99_ms`, que no cabe en un presupuesto como ése si las torres no están en `ESQUEMAS`.
- **Ruido:** la instancia hace otras cosas. Mire `carga_instancia` y `contenedor.cpuset`; con `CHISPA_CPUSET=<núcleo>` el contenedor queda en un núcleo físico concreto (déjelo libre de lo demás).
  Una corrida con carga alta no vale: repita. Una sola instancia y una sola corrida no dan IC; los 5 fríos dan al menos el rango.
- **Lo que no mide:** HTTP/JSON del servidor Go, concurrencia, caché de torres, L2c/CONJ, L3, ni la calidad (los pesos son los reales, las puntuaciones se descartan).

## Notas de preparación
- `onnxruntime` va fijado a 1.31.1 (el del repo, `modelos.sh`); `numpy` 2.5.4 y `tokenizers` 0.23.3 son las versiones con las que se probó el banco (en un Mac con ORT 1.32.0 y en un
  contenedor `python:3.12-slim` arm64 con ORT 1.31.1, con un `.chispa` sintético). Cámbielas con `CHISPA_*_VERSION`. La imagen base `python:3.12-slim` **no va fijada por digest**: ponga
  `CHISPA_IMAGEN=python:3.12-slim@sha256:…` con el digest que elija (el script anota el que resuelva). El contenedor corre sin red y con el sistema de ficheros de solo lectura.
- Pruebas del banco: `python -m unittest model.alt.tests.test_latencia_1vcpu` (necesita numpy, onnx, onnxruntime y tokenizers; construye un `.chispa` diminuto en un directorio temporal).
