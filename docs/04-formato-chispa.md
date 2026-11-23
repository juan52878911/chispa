# El formato `.chispa` v3

Un solo fichero lleva L0, L1, L2, los esquemas compilados y la calibración. Se valida entero antes de usar nada.
Especificación y validaciones: [`format/chispa3/README.md`](../format/chispa3/README.md). Cargador: `chispa3.Abrir`.

## Estructura
Little endian.

- Cabecera de 32 bytes: magia, versión 3, número de secciones, tamaño total y CRC-32C.
- Tabla de secciones: tipo, offset, longitud y sha256 de cada una.
- Datos: las secciones, alineadas a 16 bytes.

Hasta diez secciones (una por tipo), sin tipos repetidos ni desconocidos. El escritor ordena por tipo, así que la salida es determinista.
El sha256 de una sección se comprueba una vez, al pedir sus bytes. Abrir no lee los datos.

## Secciones
Una sección es opcional. La cascada arma las capas con las que encuentra.

| Tipo | Sección | Qué lleva | Quién la usa |
|---|---|---|---|
| 1 | `META` | JSON: versión, tareas, licencias y `dorado` (sha256 hex o vacío) | trazabilidad |
| 2 | `L0` | el `.chispa` v2 de kindling, tal cual (Chispa lineal) | L0 |
| 3 | `L1` | el `.chispa-l1` de `runtime/l1` (embedding estático), tal cual | L1 |
| 4 | `L2` | ONNX int8 del codificador y las ranuras (blob) | L2 y L2c |
| 5 | `ESQUEMAS` | torres de opción int8, una por hash `HashEsquema(pregunta, opciones)` | L2 |
| 6 | `CALIB` | JSON: `tipos` (T, a, b, θ por tipo de pregunta), `tau_escalate`, y opcionales `tau_l2c`, `tipos_l2` y `tau_escalate_l2` | τ y calibración |
| 7 | `CONJ` | cabeza conjunta L0 + L2 por esquema: pesos, T, β y prototipos int8 por clase | L2c |
| 8 | `L0TF` | L0 TF-IDF de caracteres con regresión logística; vocabulario ordenado, coeficientes f32 o f16 | L0, y entrada de L2c |
| 9 | `TORR` | ONNX de la torre de esquemas (e5 de pasaje + cabeza de torre) con dim, max_len y prefijo | L2 al vuelo |
| 10 | `SOPL1` | por pregunta y opción: k (ejemplos) y media int8 de sus embeddings de L1 | prototipo encogido en L1 y L2c (v2: con la norma de cada ranura de L2c; v3: y el n del soporte de CONJ) |
| 11 | `L0MULTI` | varios L0 en un fichero, uno por pregunta few-shot: cada uno es el contenido de una sección `L0` o `L0TF`, con su nombre | L0 (todos), y entrada de la unión de `mkmodelo -unir` |

Los números salen del enum de [`chispa3.go`](../format/chispa3/chispa3.go).

### Notas por sección
- **META**: `dorado` guarda la huella del conjunto dorado con que se evaluó, si la hay.
- **META.`texto_torre`** (opcional): versión del texto con que L2 calcula al vuelo la torre de las opciones de un esquema nuevo (sección `TORR`). `"v1"` (valor por defecto si el campo falta, así que los `.chispa` anteriores no cambian) es `passage: pregunta · id · descripción`; `"v2_sin_id"` omite el id cuando la opción tiene descripción: `passage: pregunta · descripción`. Cualquier otro valor es un error de carga (`META.texto_torre "…" desconocido`), no un v1 silencioso. Regla completa en [08-esquemas-nuevos.md](08-esquemas-nuevos.md#texto-de-torre). Lo leen `format/chispa3` (`MetaTexto`; el runtime usa `Fichero.TextoTorreEfectivo()`, que parte de `TORR`), el servidor, `chispamovil` y, a través de este, los SDK de Android e iOS.
- **Lectura tolerante en el runtime.** El servidor y `chispamovil` leen de META *solo* `texto_torre` (los demás campos no les sirven) y *solo* si el `.chispa` trae `TORR`: ignoran campos desconocidos y no validan `licencias` ni `dorado`, así que un `.chispa` zero-shot con META «libre» sigue cargando. Un `texto_torre` desconocido con `TORR` es un error de carga claro; uno **distinto del de `TORR`** (p. ej. `META.texto_torre: "v2_sin_id"` con TORR v1, lo que escribían R18 y `chispa train` de ZV) **no** impide cargar: el modelo carga sin torre al vuelo (ver [08](08-esquemas-nuevos.md#texto-de-torre)). La META estricta (`licencias` mapa de strings, `dorado` sha256 hex o vacío) es la que escribe y valida el entrenador: `formato.meta` (Python) y el entrenador de Go rechazan lo que no la cumple.
- **`chispa train` y `texto_torre`.** Con `--l2`/`--ckpt`, el `.chispa` final copia `texto_torre` de la META del `l2.chispa` (ausente = v1; el campo solo se escribe si no es v1). Con `--torre`, el ONNX declara la versión en la metadata `chispa.torre.texto_torre` (ausente = v1) y `train` **falla** si no coincide con la del `l2.chispa` (lo mismo con la torre autodetectada junto al checkpoint). `--texto-torre {v1,v2_sin_id}` debe coincidir con las fuentes que haya; sin `--l2`/`--ckpt` escribe lo que diga (por defecto, la del ONNX o v1). El resumen de capas de `train` muestra la versión (`texto de torre v2_sin_id`).
- **L0**: elegido por el conjunto de etiquetas igual al de ids de las opciones. Un modelo por esquema.
- **L1**: `choice`, `noul`, `score` y `multi`. Los parámetros de calibración están en `CALIB`.
- **L2**: el ONNX acepta los ids del tokenizador de e5. El servidor necesita `-tok` ([06-despliegue.md](06-despliegue.md)).
- **ESQUEMAS**: el hash se calcula sobre la pregunta y las opciones. Cambiar la pregunta cambia el hash. Versión 1: `HashEsquema(pregunta, ids)` (32 bytes; no ve descripción ni `kind`). Versión 2: `HashEsquema2` (16 bytes, ver «Clave de esquema v2»); la versión va en la cabecera.
- **CALIB**: ver «Claves de CALIB». La clave `scoreK` guarda a y b de la cabeza ordinal; los incrementos van en `scoreK#j`.
- **CONJ**: el hash es `HashEsquema(pregunta, ids ordenados)` (v1) o `HashEsquema2(pregunta, kind, opciones)` (v2). La pregunta debe ser la misma que al entrenar. Solo `choice`.
- **L0TF**: si está, el servidor usa este L0 y no el de la sección `L0`.
- **L0MULTI**: `u32 versión(1) · u32 N` y N × (`u32 tipo` (2 = `L0` o 8 = `L0TF`) · `u16 len` · nombre UTF-8 (opcional, para mensajes de error) · `u64 len` · datos tal cual de esa sección); little endian, sin relleno ([`l0multi.go`](../format/chispa3/l0multi.go)). Cada modelo lleva la versión interna de su tipo (`L0TFVersion`…). Se escribe cuando son **dos o más** L0; con uno solo se sigue escribiendo `L0` o `L0TF` como siempre. El servidor y el móvil añaden a L0 el de `L0TF` (o `L0`) y todos los de `L0MULTI`, y la elección por pregunta es la de siempre (por el conjunto de ids de las opciones). Dos modelos con el mismo conjunto de etiquetas son un error al unir y al cargar. **No hay CALIB ni τ por modelo**: la calibración es la de la sección `CALIB` del fichero (igual que con `"modelos"`). Golden: `format/chispa3/testdata/l0multi.chispa`.
- **TORR**: `u32 versión(1) · u32 dim · u32 max_len · u32 len_prefijo · prefijo UTF-8 · ONNX` (little endian; [`torr.go`](../format/chispa3/torr.go)). **TORR v2** lleva además la versión del texto de torre: `u32 versión(2) · u32 dim · u32 max_len · u32 len_prefijo · u32 texto · prefijo UTF-8 · ONNX`, con `texto` 2 = `v2_sin_id`. TORR v1 significa siempre texto v1; una torre con texto v1 se escribe como TORR v1 (los `.chispa` de siempre no cambian de bytes). Se lee la 1 y la 2; otra versión o un `texto` desconocido es un error de carga, y un runtime anterior rechaza la v2 (`TORR v2: versión desconocida`) y se queda sin torre al vuelo, en vez de calcular v1 en silencio. Si `META.texto_torre` existe y no coincide con TORR, `Fichero.TextoTorreEfectivo` devuelve `ErrTextoTorreInconsistente` y el runtime **degrada** en vez de rechazar el fichero: carga sin torre al vuelo, con aviso (`mt.Avisos`/`Avisos()` y log), `torre_al_vuelo_motivo: "texto_torre_inconsistente"` en `/healthz` y `L0`/`L1`/`L2` con `ESQUEMAS` en pie (META sin el campo no se contrasta). Reconstruya el fichero con `chispa train` para recuperar las torres al vuelo. Quien escribe TORR es solo `chispa train --torre`: v2 si el texto del fichero es `v2_sin_id`. `max_len` es el truncado del pasaje que aplica el runtime (por defecto 128; tope duro 512, `chispa3.MaxLenTope`; por encima, error al escribir o abrir). El del mensaje no va en ninguna sección: es la clave `chispa.consulta.max_len` de `metadata_props` del ONNX de la sección `L2` (misma convención que `chispa.torre.*`; sin ella, 128; tope 512 con error claro; [`maxlen.go`](../format/chispa3/maxlen.go), que lo lee en Go puro sin ORT). `chispa train --l2 <fichero>` no la escribe: la pone el exportador del modelo. El servidor y el núcleo móvil la leen al abrir (`MaxLenConsulta`/`MaxLenPasaje`); Kotlin y Swift no recortan por su cuenta. El escritor Python (`formato.torr`) da los mismos bytes: `format/chispa3/testdata/torre.chispa` lo escribe Python y lo lee un test de Go.
- **SOPL1**: `u32 versión(1, 2 o 3) · u32 dim · u32 n` y, por pregunta (hash = `ClaveConj`, como CONJ, ordenado), `u32 C · C×(u16 len · id) · C×u32 k · C×f32 escala · C×dim×int8 media` y, **desde la versión 2**, `u32 K · C×K×f32 normas` y, **solo en la versión 3**, `u32 M · M×u32 n_l2` (M = 0 o C) ([`sopl1.go`](../format/chispa3/sopl1.go)). La media de los embeddings de L1 de los ejemplos de cada opción va en int8 con escala por opción (decisión: 4× menos que f32 y error de cuantización ≤ 0,4 %). El runtime mezcla `p = normaliza((1−w)·p_desc + w·media)` con `w = k/(k+λ)`. **Equivalencia con la fórmula de G.8** (`model/l2/formula-shrinkage.md`): κ ≡ nuestra λ; el peso de la descripción es `λ_c = κ/(κ+n_c) = 1−w` (`chispa3.LambdaClase`, `PesoSoporte`), la media de los ejemplos va **sin normalizar** antes de mezclar (la norma se aplica después) y κ = 0 da solo ejemplos. En L1 es un vector por opción (`Mezclar`). En L2c se mezclan los prototipos del simproto, no la torre: `Q_j = λ_c·h + (1−λ_c)·M_j` con el mismo `h` (torre de la descripción) en las K ranuras, `P_j = Q_j/‖Q_j‖` por ranura y `simproto = mean_s max_j s·P_j` (`SimprotoMezclado`, `ProtosMezclados`); `sim_opción` no cambia. Una opción sin ejemplos conserva el simproto de CONJ. **Normas por ranura (versión 2).** CONJ guarda las ranuras de soporte ya normalizadas, pero `modelo.py` mezcla la media **sin normalizar** (`M_j = (1/n)·Σ S_i,j`, norma ≤ 1). La versión 2 añade a cada entrada `u32 K · C×K×f32 normas` (`normas[c·K+j] = ‖M_c,j‖`, en [0, 1]; K = 0 = entrada sin normas); el runtime reconstruye `M_j = proto_j·escala_j·norma_j` (`EntradaSopl1.NormasDe`, `ProtosMezclados`) y la fórmula coincide con `modelo.py` para todo n. El escritor emite la versión 2 solo si alguna entrada trae normas (si no, los bytes son los de la versión 1 de siempre); el lector acepta las dos. Un `.chispa` con SOPL1 v1, o con v2 pero con K distinto del de CONJ, usa la media normalizada: con n = 1 coincide con `modelo.py` y con n ≥ 2 la media pesa algo más que allí. Las normas salen de `chispa train --conjunta --shrinkage-lambda …` (`conjunta.normas_ranuras`: ‖Σ S‖/n con las mismas filas de soporte que `prototipos`); L1 no las usa. **n_l2 (versión 3).** El soporte de CONJ (los `SOPORTE` por clase) puede ser un subconjunto de los ejemplos que ve L1 (k), y `modelo.py` pesa con el n del primero: `λ_c = κ/(κ+n)`. La versión 3 añade a cada entrada, tras las normas, `u32 M · M×u32 n_l2` (M = 0 = sin dato, o M = C con el n de cada clase; `n_l2[c] = 0` también = sin dato). L2c usa `n_l2` si está (`EntradaSopl1.NL2De`, en `SimprotoMezclado`: servidor y `chispamovil`) y si no el k de siempre; **L1 sigue con k**. Se escribe desde el mismo recuento de filas de soporte que las normas (`conjunta.n_soporte`; `chispa train --conjunta --shrinkage-lambda …`). Va como versión 3 y no como ampliación de la 2 porque las entradas no llevan longitud: un lector de la 2 rechazaría los bytes de más. El escritor emite la 3 solo si alguna entrada trae `n_l2` (si no, los bytes son los de la 1 o la 2 de siempre); el lector acepta las tres. Caso a mano (`TestShrinkageL2cUsaNL2`): k = 5, n_l2 = 3, κ = 3 ⇒ λ_c = 0,5 como en `modelo.py` (con k saldría 3/8).

**Semántica de λ** (L1 y L2c, igual): ausente/`nil` = **apagado** (el comportamiento de antes del soporte); λ = 0 → `λ_c = 0`, solo ejemplos; λ = +∞ → `λ_c = 1`, solo descripción (en L2c antes +∞ apagaba; ya no). La config es `l1_shrinkage_lambda` y `l2_shrinkage_lambda` (JSON no tiene ∞: un valor enorme, p. ej. 1e300, hace de ∞); en el núcleo móvil `AbrirConSoporte(ruta, λL1, λL2)` con `SoporteApagado` (−1; negativo o NaN también) como apagado, y en Kotlin/Swift `lambdaSoporteL1/L2` nulos = apagado. `chispa train --l1 … --ejemplos … --shrinkage-lambda λ` la escribe (sin la opción no se escribe nada). `formato.sopl1` (Python) da los mismos bytes: `format/chispa3/testdata/soporte.chispa` (v1) `soporte-v2.chispa` (v2, con normas, caso a mano n = 3) y `soporte-v3.chispa` (v3, con n_l2).

## Espacio de nombres de L2 en CALIB
`tipos` y `tau_escalate` son de L1; L2 y L2c leen, si existen, sus propios campos:
- `tipos_l2` (opcional): mismo formato que `tipos` (T, a, b, θ). Es lo que lee L2 (`choice`, `noul`, `scoreK`, `scoreK#j`, `multi_l2@…`) y, con `CONJ`, también L2c. Si falta, L2 lee `tipos`, como antes.
- `tau_escalate_l2` (opcional, en [0,1]): el τ de L2. Si falta, L2 usa `tau_escalate`. Con `CONJ`, `tau_l2c` sigue mandando sobre él.
- L1 lee siempre `tipos` y `tau_escalate`; ningún campo de los de L2 se usa en L1 en el runtime (servidor `decide.ArmarDesde`, `chispamovil` con `TauEscalateL2()`, Kotlin/Swift `Motor.tauEscalateL2()`).
- `tau_l2_esquema_nuevo` y `l1_lambda_calib` no cambian.
- Un fichero sin estos campos se lee y decide exactamente igual que antes. Si `tipos_l2` existe (aunque esté vacío) L2 no mira `tipos`: sus claves `multi_l2@…` deben estar en `tipos_l2`.
- `chispa train --l2 x.chispa` y `--ckpt` copian de la CALIB de `x.chispa`: `tipos` → `tipos_l2` (con `scoreK`, `scoreK#j` y `multi_l2@…`), `tau_escalate` → `tau_escalate_l2` y `tau_l2c` → `tau_l2c`; la CALIB de L1 del fichero final no se toca. Si `x.chispa` no trae CALIB, no se escribe nada de L2. Con `--conjunta`, `tau_l2c` es el calibrado para la nueva cabeza y no el de `x.chispa`. Las claves `multi_l2@…` que `chispa train` calibre con ejemplos van a `tipos_l2` si el fichero lo trae.
- **Un runtime anterior rechaza el fichero.** El lector de CALIB decodifica con `DisallowUnknownFields` (`format/chispa3/secciones.go`): un servidor, SDK o app móvil sin estos campos falla al cargar con `chispa3: contenido de sección inválido: json: unknown field "tipos_l2"` (igual con `tau_escalate_l2`); no cae a `tipos` ni falla en silencio. Versión mínima de servidor, SDK y `chispamovil` para abrirlos: commit `cb11a09` o posterior. No hay marca de versión: no hagas rollback del runtime, ni distribuyas a apps móviles ya publicadas, un fichero que traiga `tipos_l2` o `tau_escalate_l2`. Un fichero sin esos campos sigue abriéndose en cualquier runtime.

## Claves de CALIB
Las de `multi` (ver [03-api.md](03-api.md#respuesta-200)):
- `multi@<hex>`: (a, b, θ) de todas las opciones de un esquema; `<hex>` es el `HashEsquema2` (pregunta, kind y opciones con descripción) en hex. `multi@<hex>#<id>`: los de una opción.
- `multi_l2@<hex>` y `multi_l2@<hex>#<id>`: lo mismo para L2 (MaxSim); se leen de `tipos_l2` si existe y, si no, de `tipos`. **Sin ellas L2 no opina en `multi`.** `chispa train` no las calibra (no tiene el ONNX de L2); las escribe un exportador de modelos o `multi.escribir_calib` con `maxsim_l2` por la API, y van a `tipos_l2`. Un `chispa train` posterior con `--l2` copia a `tipos_l2` las que ya traiga la CALIB de `x.chispa`.
- `multi` y `multi#<id>` (sin `@`): ficheros anteriores; solo se leen si el fichero no tiene ninguna clave con `@`.
- `#<id>` en general: sufijo de parámetros por opción.

## Origen del τ
- **CALIB manda siempre.** El τ de L1 es `tau_escalate`. El de L2 es `tau_escalate_l2` si existe, si no `tau_escalate`; con `CONJ`, `tau_l2c`. Da igual que el servidor arranque con `-config` o sin él (`decide.ArmarDesde`).
- **La config solo puede subirlo.** Solo cuenta si su JSON trae `capas`. Un τ de `capas` mayor que el de CALIB lo sube y sale en el log («τ de L1: la config lo sube de … a …»). Uno menor se ignora con aviso. Lo mismo vale para `tau_l2_esquema_nuevo`.
- **Antes había un fallo.** Con `-config` (incluso `{}`) mandaban los τ fijos de `ConfigPorDefecto` (0,80/0,80/0,70/0,60) sin avisar. La misma batería escalaba entre el 83 y el 100 % sin config y un 37 % con `{}`. Test: `TestTauUnicoConYSinConfig` (`server/cmd/chispa-server`).
- **Sin CALIB nada cambia.** Siguen valiendo los τ de la config, o los de por defecto. L0 y L3 no tienen τ en CALIB y siguen en la config.

## CALIB sin `tau_escalate`
Se rechaza al cargar con `chispa3.ErrCalibSinTau` («CALIB sin tau_escalate … reexporte el modelo»). Antes se leía como τ = 0 y L1 aceptaba cualquier respuesta. Un `"tau_escalate": 0` explícito sigue siendo válido. `formato.calib` de Python se niega a escribir una CALIB sin él.

## Calibración por capa (`calibracion`)
Clave opcional de CALIB. Lleva una calibración de la confianza top-1 por capa (`L0`, `L1`, `L2`, `L3`). La cascada la aplica antes de comparar con el τ de la capa. Las `probs` de la respuesta salen sin tocar.

```json
"calibracion": {
  "L0": {"metodo": "platt", "a": 1.3, "b": -0.2},
  "L1": {"metodo": "isotonica", "x": [0.2, 0.6, 1.0], "y": [0.0, 0.5, 0.9]}
}
```

- **`platt`**: c' = σ(a·logit(c) + b), con c recortada a [1e-6, 1−1e-6]. Exige `a > 0` (no puede invertir el orden) y `b` finito.
- **`isotonica`**: interpolación lineal de (x, y). `x` es estrictamente creciente en [0,1] e `y` no decreciente en [0,1], las dos de la misma longitud ≥ 2. Fuera de [x₀, xₙ] vale y₀ o yₙ.
- Una capa sin opinión (c = 0) sigue en 0. Una capa desconocida, un método desconocido o unos parámetros fuera de rango dan `ErrContenido` al cargar.
- **Hoy no la produce ningún exportador**: los valores los generará `model/`. El lector no inventa ninguno.
- **Qué capas cuentan como calibradas**: las que tienen entrada en `calibracion`, y L2 si CALIB trae `tipos_l2`. Con los ficheros de hoy, L0, L1 y L3 no lo están. Contestan igual por encima de su τ, pero la respuesta lleva `"calibrada": false` (ver [03-api.md](03-api.md)).
- **Un runtime anterior rechaza el fichero** con `unknown field "calibracion"`, igual que con `tipos_l2`. No distribuyas un fichero con `calibracion` a apps móviles ya publicadas.

## Señales en la respuesta
Campos opcionales nuevos en cada resultado de `/v1/decide` y en los SDK de Go, JS y Python. Un servidor anterior no los manda.

| Campo | Cuándo sale | Qué es |
|---|---|---|
| `calibrada` | Contesta una capa y el servidor sabe qué capas están calibradas | `true` o `false` (ver arriba). No sale al escalar |
| `truncado` | Alguna capa que corrió vio solo una parte del texto | `true`: pasaba de `max_tokens` de L1 o `max_len` de L2 y la estrategia no es `trozos` (`cabeza_cola` también pierde el centro) |
| `tokens_originales` | Con `truncado` | Tokens del texto entero (el mayor entre las capas que truncaron) |
| `unk_pct` | Alguna capa tokenizó el texto (L1, o L2 con el tokenizador de e5) | Fracción de tokens `<unk>`, en [0,1] (la mayor entre esas capas) |

- **Antes**, el truncado solo aparecía en `/metrics` (`chispa_texto_largo_total`) y no había contador de `<unk>`.
- **Coste**: L1 saca la señal de su propia tokenización. Solo un texto truncado cuesta una pasada más, para contar sus tokens enteros. L2 tokeniza una vez más, frente al ONNX que domina su coste. Los bancos de Go no cambian de forma medible (ver [07-rendimiento-y-limites.md](07-rendimiento-y-limites.md)).

## Compatibilidad
- **v2 se carga.** Un `.chispa` v2 de kindling se abre como un v3 con solo `L0`.
- **v1 no se carga.** kindling v0.17.0 no la conoce; da `ErrVersion`.
- **Los ficheros viejos cargan.** `CONJ`, `L0TF`, `TORR`, `SOPL1` y `L0MULTI` se añadieron al final del enum sin renumerar. Un fichero sin ellas carga igual
  (el README del formato lo comprueba con `TestTORRAusenteEnTestdata` y `TestSOPL1AusenteEnTestdata`).
- **Un servidor viejo no abre un fichero nuevo.** Los tipos desconocidos se rechazan: un `.chispa` con `TORR`, `SOPL1` o `L0MULTI` (esta, con `ErrTipo` al abrir, sin leer a medias) pide un servidor que las conozca y uno con `tipos_l2`/`tau_escalate_l2` en `CALIB` pide `cb11a09` o posterior (campo desconocido en el JSON de CALIB).
- **Claves v1 y v2.** `META.version` 1 (todos los ficheros anteriores) usa las claves sin descripción; 2, las claves con descripción. No cambia ninguna sección ni el número de formato (sigue siendo 3). Un runtime que no conozca la v2 abre el fichero igual pero no encontrará las torres (cae a la torre al vuelo, si la hay).
- **`multi@…` y servidores anteriores.** Añadir estas claves no cambia ninguna sección ni el número de formato, así que **un servidor anterior abre el fichero sin error** y, como solo busca `multi#<id>` y `multi`, ignora `multi@…`/`multi_l2@…` y calibra multi con `noul`. Versión mínima del servidor para esos ficheros: la de la CALIB por esquema (commit `39fb6cc` o posterior). Excepción a «Un servidor viejo no abre un fichero nuevo»: aquí falla en silencio (a diferencia de `tipos_l2`, que sí se rechaza). No hay marca de versión en CALIB; no hagas rollback del binario con un modelo que traiga `multi@…`.
- **Cambio de pregunta.** Cambia el hash de `ESQUEMAS` y `CONJ` (desde la v2, también el de una descripción). L2c cae a L2 y L2 no encuentra la torre de esa pregunta: reentrena con la pregunta nueva.
- **La cabecera sigue en versión 3.** `L0MULTI` es un tipo nuevo al final del enum (sin renumerar); no cambia `map[Tipo]` ni la regla de «tipo repetido». Los v3 de un solo `L0` o `L0TF` se leen igual que antes.
- Un servidor `chispa-server` admite un solo `-modelo`. Para unir secciones de varios ficheros (y los L0 de cada uno en un `L0MULTI`): `deploy/mkmodelo -unir`
  ([`server/deploy/README.md`](../server/deploy/README.md)).

## Robustez
`FuzzAbrir` y `FuzzL0TF` corrieron sin fallos: ver el log al final de [`format/chispa3/README.md`](../format/chispa3/README.md#log-del-fuzzing).

## Qué no se probó
- Una corrida registrada de fuzzing de `TORR` (existen `FuzzDecodificarTORR` y `FuzzAbrirTORR`) y de `SOPL1`: el README del formato solo documenta las de `FuzzAbrir` y `FuzzL0TF`.
- El límite de secciones: el README del formato dice «6» por defecto, pero el código usa `numTipos-1` (10, el enum de `chispa3.go`); aquí manda el código.
- Windows y macOS: `mmap_otros.go` es un respaldo con lectura.
- Una sección `L1` real de 155 MB pasada por `l1.Desde` desde el cargador (la prueba usó una cabecera sintética).

## Clave de esquema v2 (`HashEsquema2`)

La clave v1 no recoge la descripción ni el `kind`: dos esquemas con las mismas opciones y descripciones distintas compartían torres. La v2 sí:

```
HashEsquema2 = sha256( pregunta · 0x1f · kind · 0x1f ·
                       por cada opción ordenada por id (orden de bytes): id · 0x1e · descripción · 0x1d )[:16]
```

Todo en UTF-8, sin prefijos de longitud; `kind` es `choice`, `score`, `noul` o `multi`; la descripción ausente es la cadena vacía. Son 16 bytes (los 16 primeros del sha256).

- **ESQUEMAS v2**: `u32 (versión<<16 | dim) · u32 n · n×[16]byte clave (estrictamente crecientes) · n×f32 escala · n×dim×int8`. En v1 la primera palabra es `dim` (≤ 4096), o sea versión 0 = v1; un lector viejo rechaza la v2 por dim fuera de rango. La clave de cada torre es `HashEsquema2(pregunta, kind, [opción])` en `choice` y `multi` (una torre por opción), y `HashEsquema2(pregunta, kind, opciones)` en `score` (todas) y `noul` (`false` y `true`, con la descripción de la petición si la trae, si no vacía).
- **CONJ v2**: igual que la v1 con `u32 versión = 2` y el hash de cada entrada de 16 bytes: `HashEsquema2(pregunta, "choice", todas las opciones)`.
- **Lector**: acepta v1 y v2 (`Esquemas.Version()`, `Conj.Version()`). El runtime (servidor `decide`, `chispamovil`) elige la función de clave con la versión de cada sección (`BuscarEsquema`); una sección v2 no responde a claves v1 ni al revés, así que no se mezclan. Con v2, la misma pregunta con otra descripción (u otro `kind`) es otro esquema: sus torres se calculan al vuelo. La caché de torres al vuelo se indexa por `HashEsquema(pregunta, [id, descripción])` (una torre por opción, no depende del fichero) y la de resultados por `ClaveEsquema` (ya incluye `kind` y descripciones).
- **SOPL1** no cambia: sigue con `ClaveConj` (v1).
- **Vectores de prueba**: `format/chispa3/vectores-clave-v2.json` (choice con descripciones, noul, score, opciones desordenadas, otra descripción), comprobados en Go (`clave2_test.go`) y Python (`tests/test_claves_v2.py`). Escritor Python: `formato.hash_esquema2`, `formato.esquemas(…, version=2)`, `formato.conj(…, version=2)`.

## Inventario de variantes de sección

Qué versiones y claves de sección existen, quién las escribe hoy y quién las lee. Es un inventario: **no se ha borrado ni cambiado nada** (ni código, ni fixtures, ni dorados). Las rutas son relativas a la raíz del repositorio.

Cómo leerlo:
- **Escribe hoy** = un camino de producción (`chispa train`, el exportador de L2, `mkmodelo`) que emite esa variante. Si solo la emite un test, un generador sintético o un fixture, cuenta como «solo compatibilidad».
- **Leen**: el lector de Go (`format/chispa3`) acepta todas las versiones; el servidor (`server/decide`) y `chispamovil` (`sdk/movil/chispamovil`) consumen lo que decodifica. Android e iOS no parsean secciones: hablan con `chispamovil` por la interfaz `Motor` (`sdk/android/chispa/src/main/kotlin/com/chispa/sdk/Capas.kt:13-38`). Python solo tiene lectores en `sdk/python/chispa/formato.py` que usan los tests de paridad (`leer_sopl1`, `leer_torr`, `leer_l0multi`); ningún camino de producción de Python lee estas secciones.
- Escritores de referencia: Python `sdk/python/chispa/formato.py` (`meta` :94, `calib` :108, `esquemas` :183, `conj` :215, `l0multi` :282, `torr` :344, `sopl1` :409); entrenador de Go `sdk/python/chispa/entrenador_go/entrenar.go`; exportador de L2 `model/l2/exportar.py` con `model/l2/empaquetar_go/main.go`; unión `server/deploy/mkmodelo/main.go`.

| Variante | Versión / clave | La escribe hoy | La leen | ¿Solo compatibilidad? | Tests y fixtures que dependen |
|---|---|---|---|---|---|
| SOPL1 | v1 (sin normas) | **Sí.** `chispa train --l1 --ejemplos` sin `--conjunta` (`train.py:152-179`; `:463-467` pasa `normas=None`) y `multi.py:183` | Go `sopl1.go:174-177`; servidor `modelo.go:138,143`; `chispamovil.go:263`. L1 usa solo k, escala y media; L2c cae a la media normalizada (`sopl1.go:434`) | No | `format/chispa3/testdata/soporte.chispa` (`sopl1_test.go`, `test_sopl1.py`); `l1_soporte_test.go`, `lambda_calib_test.go`, `test_train_sopl1.py` |
| SOPL1 | v2 (+ normas por ranura) | **No.** `formato.sopl1` la emitiría con `normas` sin `n_l2`, pero `--conjunta` pasa siempre el n (`conjunta.py:164`, `train.py:176-178`). Ningún escritor de Go la emite. Vivió del 2026-10-28 al 29 (`fd7564e` → `8f1d014`) | Igual que la v1 | **Sí**, ficheros de `--conjunta` generados entre esos dos commits | `soporte-v2.chispa` (`sopl1_test.go:307`, `test_sopl1.py`); `Normas:` sin `NL2:` en `sopl1_test.go`, `l2c_soporte_test.go`, `soporte_test.go` |
| SOPL1 | v3 (+ `n_l2`) | **Sí.** `chispa train --conjunta --shrinkage-lambda …` | Igual; L2c usa `n_l2` (`sopl1.go:434`, `NL2De` :457) | No | `soporte-v3.chispa` (`sopl1_test.go:407`, `test_sopl1.py`); `l2c_soporte_test.go:152`, `soporte_test.go:235` |
| TORR | v1 (texto de torre v1) | **Sí.** `chispa train --torre` con texto v1 (`torre.py:173-190`, `formato.py:353`); también `model/bench/movil/reexportar_fp16.py:134` y `model/bench/torre/paridad_servidor.py` | Go `torr.go:74-84`; servidor `modelo.go:95,203`; `chispamovil.go:226,296` | No | `testdata/torre.chispa` (`torr_test.go`, `test_torr.py`, `server/e2e/correr.sh`); `server/torre/testdata/torre_train.chispa` (también ESQUEMAS v1); `server/decide/l2_torre_test.go` |
| TORR | v2 (+ campo `texto`, `v2_sin_id`) | **Sí.** `chispa train --torre` con texto `v2_sin_id` (`formato.py:354`) | Igual que v1 | No | Sin fichero en el repositorio: se construye en memoria. `torr_test.go`, `texto_torre_v2_test.go` (servidor y móvil), `test_torr.py`, `test_texto_torre_zv.py`, `TextoTorreV2Tests.swift`, `TextoTorreV2Test.kt` |
| META.`texto_torre` | ausente (= v1) | **Sí.** Nunca se escribe `"v1"`: se omite (`formato.py:104`, `exportar.py:680`, `entrenar.go:370`) | `secciones.go:106` (`MetaTexto`); servidor `modelo.go:95`; `chispamovil.go:228` | No | `meta_texto_torre_test.go`, `version_texto_torre_zv_test.go` |
| META.`texto_torre` | `"v2_sin_id"` | **Sí.** Exportador (según el checkpoint), Go y `fijar_texto_torre` (`torre.py:221`) | Igual | No | Los de TORR v2 y `meta_tolerante_zv_test.go` |
| META.`texto_torre` + TORR | combinación incoherente (META v2 con TORR v1) | **No.** `chispa train` falla si no coinciden (`torre.py:208-217`) | Degrada a «sin torre al vuelo» (`torr.go:117-128`, `modelo.go:95-107`) | **Sí**, ficheros de R18 y del `chispa train` anterior a ZV | `meta_tolerante_zv_test.go` (servidor y móvil), `torre_motivo_test.go` (móvil), `TorreMotivoTests.swift` |
| CALIB | `tipos` + `tau_escalate` | **Sí.** Siempre; en ficheros solo-L2 (exportador, `empaquetar_go/main.go:80`) son los de L2 | `secciones.go:162-176`; L1 siempre; L2 como respaldo si falta `tipos_l2` (`ParaL2` :180) | No: el exportador de L2 solo escribe estas claves | Todos los fixtures con CALIB |
| CALIB | `tipos_l2`, `tau_escalate_l2` | **Sí.** `chispa train --l2/--ckpt` (`entrenar.go:362-364`, `train.py:301-315`) | `secciones.go:171-188`; servidor `modelo.go:121,208-218`; `chispamovil.go:364,583` | No | `calib_l2_casos.json` (cruzado Go/Python), `tipos_l2_test.go` (servidor y móvil), `test_train.py` |
| CALIB | `tau_l2c` | **Sí.** `--conjunta` (`entrenar.go:192,365`) | servidor `aplicarTauL2c` (`modelo.go:254`); `chispamovil.go:427` | No | `l2c_test.go`, `test_conjunta_v2.py` |
| CALIB | `tau_l2_esquema_nuevo` | **Sí.** `chispa train --tau-l2-esquema-nuevo` (`train.py:183-190`) | servidor `modelo.go:255-260`; `chispamovil.go:434-440` | No (si falta, 0,846 por defecto) | `tau_l2_esquema_nuevo` en `server/decide/*_test.go`, `TauEsquemaNuevoTests.swift` |
| CALIB | `l1_lambda_calib` | **Sí** con encogimiento (`entrenar.go:354`, `train.py:195`); ausente = calibrado sin encogimiento | servidor `modelo.go:128-173`; `chispamovil.go:250` | La **ausencia** es compatibilidad (genera aviso) | `lambda_calib_test.go`, `l1_soporte_test.go` |
| CALIB | `multi` / `multi#<id>` / `multi_l2` (sin `@`) | **No.** `multi.py` escribe solo `multi@<hex>` (`multi.py:33,106,115`). Únicas fuentes: `calibDorada` (`dorados_test.go:128-137`) y fixtures | servidor `multi.go:21-22,60-72`; `chispamovil.go:836-916` | **Sí** | `testdata/dorados/{l0l1,l0tf,l2}.chispa`, `calibDorada`; pruebas de multi sin esquema |
| CALIB | `multi@<hex>` / `multi_l2@<hex>` | **Sí.** `multi.py` y `model/l2/calibrar.py:266` | mismas rutas | No | `test_multi.py`, `test_multi_zb.py`, `r11_qb_test.go`, `r11_test.go` |
| L0MULTI | v1 (única) | **Sí.** `mkmodelo -unir` si son dos o más L0 (`mkmodelo/main.go:164-172`). Python `formato.l0multi` no tiene llamadas fuera de tests | Go `l0multi.go:104-108,152,212`; servidor `modelo.go:88` → `l0.go:78`; `chispamovil.go:270` | No (tipo nuevo, 2026-11-03, `52a9904`) | `testdata/l0multi.chispa`; `l0multi_test.go` (formato, servidor, móvil), `caida_unido_test.go`, `mkmodelo/main_test.go`, `test_l0multi.py` |
| L0 / L0TF | `L0` (L0 lineal heredado) | **Sí.** Entrenador de Go salvo en few-shot (`sin_l0`, `train.py:605`; `entrenar.go:117`) | `l0.go:78` y `chispamovil.go:270`; L0TF manda sobre L0 | No | `dorados/l0l1.chispa`, `server/testdata/caida/*.chispa`, `injection.chispa` |
| L0 / L0TF | `L0TF` | **Sí.** `--conjunta` (`entrenar.go:203`; Python `l0tf.py:89`) | Igual | No | `dorados/l0tf.chispa`, `conj-v2.chispa`, `l0tf_test.go` |
| L0 | contenedor `.chispa` v2 (sin cabecera v3) | **No** (ficheros previos a v3) | `legado.go:11-27` | **Sí** (documentado en «Compatibilidad») | `server/testdata/{injection,intencion,urgencia}.chispa`, `testdata/{injection,urgencia}.chispa` |
| CONJ | v1 (clave `ClaveConj`, 32 bytes) | **Casi no.** Solo si `--conjunta` recibe un `--l2` con ESQUEMAS v1 (`train.py:650-652`) | Go `conj.go:175-182`; servidor `modelo.go:184-194` (avisa si CONJ y ESQUEMAS difieren) | **Sí** en la práctica | `CodificarConj(` en `conj_test.go`, `clave2_test.go`, `conj_cabeza_test.go`, `l2c_test.go`, `l2c_soporte_test.go`. Sin fichero |
| CONJ | v2 (`HashEsquema2`, 16 bytes) | **Sí.** Sigue la versión de ESQUEMAS | Igual | No | `conj-v2.chispa`, `claves-v2*.chispa`, `conj_v2_test.go` (servidor y móvil), `test_conjunta_v2.py` |
| ESQUEMAS | v1 (`HashEsquema`, 32 bytes) | **No** en producción. El exportador solo emite v2 (`empaquetar_go/main.go:67`). Sí: generador e2e sintético (`server/e2e/gensint/main.go:165`), dorados (`dorados_test.go:335`) y tests | Go `secciones.go:363,404`; `BuscarEsquema` elige clave por versión (`clave2.go:44-60`; servidor `l2.go:829`, `l2c.go:55`; `chispamovil.go:1207`) | **Sí** | `CodificarEsquemas(` en 33 ficheros de test (23 en `server/decide`, 7 en `chispamovil`, 3 en `format/chispa3`); `dorados/l2.chispa`; `server/torre/testdata/torre_train.chispa` |
| ESQUEMAS | v2 (`HashEsquema2`, 16 bytes) | **Sí.** `model/l2/exportar.py:11` | Igual | No | `esquemas-v2.chispa`, `claves-v2*.chispa`, `conj-v2.chispa`, `test_esquemas_v2.py`, `test_claves_v2.py` |
| `huella_math` | no es una sección ni un campo de META: es una clave de `server/decide/testdata/dorados/entorno.json` | **Sí**, la escribe `go test ./decide -run Dorados -regenerar` (`dorados_test.go:356,398-405`) | solo `tolDoradoServidor` (`dorados_test.go:449-456`); ni el móvil ni Python la leen | No aplica: no es formato | `dorados_test.go`; ver [07](07-rendimiento-y-limites.md) |

Notas del inventario:
- `META.version` solo se valida (`>= 1`, `secciones.go:67`). La versión de claves la decide la versión de ESQUEMAS y de CONJ (`clave2.go:44-60`), no `META.version`. El párrafo «Claves v1 y v2» de «Compatibilidad» dice lo contrario y está desactualizado; el exportador escribe `META.version` 1 junto con ESQUEMAS v2.
- SOPL1 se indexa siempre con `ClaveConj` (hash v1), también junto a ESQUEMAS y CONJ v2. Esa clave no es una variante a retirar.
- La clave `calibracion` de CALIB es nueva (ver [Calibración por capa](#calibración-por-capa-calibracion)): hoy no la escribe ningún exportador. Las líneas citadas de `format/chispa3/secciones.go` pueden haberse desplazado con ella.
- No pude determinar qué ficheros ya están en manos de terceros: `models/` y el registro de modelos no se abrieron (ni están en este árbol de trabajo) y el repositorio no lleva una lista de distribución. Toda la propuesta de abajo queda condicionada a eso.
- Todas las versiones de sección nacieron entre 2026-10-28 y 2026-11-03 (SOPL1 v2 `fd7564e`, SOPL1 v3 `8f1d014`, CONJ/ESQUEMAS v2 `143df23`, `tipos_l2` `1361061`, `texto_torre` `5abb80a`, TORR v2 `9c9ecef`, L0MULTI `52a9904`). ESQUEMAS v1 y CONJ v1 son anteriores y los que más ficheros pueden tener fuera.

### Propuesta de limpieza (pendiente del OK de Juan)

**No se ha borrado nada.** Esto es solo una propuesta. Ningún cambio debe hacerse sin el OK de Juan y sin saber qué ficheros `.chispa` están distribuidos.

Principio: se puede quitar el **escritor** de una variante sin riesgo cuando nadie la emite; quitar el **lector** rompe cualquier fichero viejo que siga en uso (un runtime nuevo lo rechazaría con un error de carga). Por eso se propone quitar primero escritores y tests, y los lectores al final y uno a uno.

No tocar (las produce el exportador o `chispa train` hoy): SOPL1 v1 y v3, TORR v1 y v2, META.`texto_torre` ausente y `v2_sin_id`, CALIB `tipos`, `tipos_l2`, `tau_escalate`, `tau_escalate_l2`, `tau_l2c`, `tau_l2_esquema_nuevo`, `l1_lambda_calib`, `multi@…`, L0MULTI v1, L0, L0TF, CONJ v2 y ESQUEMAS v2. `tipos` debe seguir siendo el respaldo de L2 porque el exportador de L2 produce ficheros solo con `tipos`.

Orden propuesto:

1. **SOPL1 v2.** Nadie la emite desde el 2026-10-29 y el lector de la v3 ya cubre su caso. Quitar el camino «normas sin `n_l2`» de `formato.sopl1` y de `Codificar` en Go, regenerar o borrar `soporte-v2.chispa`, y reescribir en los tests de `l2c_soporte_test.go` y `soporte_test.go` las entradas con `Normas:` sin `NL2:`. Tests a regenerar: `sopl1_test.go:307` y `test_sopl1.py`. Riesgo: **bajo**, solo ficheros de `--conjunta` hechos en el día entre `fd7564e` y `8f1d014`. Si hay alguno, reentrenar con `chispa train --conjunta`.
2. **CALIB `multi` sin `@`.** Quitar el respaldo de `multi`, `multi#<id>` y `multi_l2` sin esquema (`multi.go:60-72`, `chispamovil.go:836-916`). Hay que regenerar los tres dorados (`go test ./decide -run Dorados -regenerar`, solo en amd64: `dorados_test.go:22`; los comparten el servidor, `chispamovil` y `test_dorados.py`) y actualizar `calibDorada` para que use `multi@…`. Riesgo: **medio**. Un fichero viejo con `multi` sin `@` seguiría abriéndose pero calibraría multi con `noul`, sin error (ya es el comportamiento documentado para los servidores anteriores).
3. **Degradación por META.`texto_torre` distinto de TORR.** Pasaría de «cargar sin torre al vuelo, con aviso» a error de carga. Tests: `meta_tolerante_zv_test.go` (servidor y móvil), `torre_motivo_test.go` (móvil), `TorreMotivoTests.swift`, `TextoTorreV2*`. Riesgo: **medio**, hay ficheros de R18 y del `chispa train` previo a ZV. Hacerlo solo si se confirma que ya no circulan; si no, dejarlo.
4. **CONJ v1 y ESQUEMAS v1** (juntos: se leen por versión y no se mezclan). Dejar solo `HashEsquema2`. Hay que reescribir los tests que usan `CodificarEsquemas(` (33 ficheros) y `CodificarConj(` (5), regenerar `dorados/l2.chispa` y `server/torre/testdata/torre_train.chispa`, y pasar `server/e2e/gensint` a v2. `ClaveEsquema`, `HashEsquema` y la clave de caché de `runtime/torre/torre.go:113` se miran aparte. Riesgo: **alto**. Es el cambio que deja sin torres a cualquier modelo ya distribuido con claves v1 (L2 no opina en sus esquemas salvo que traiga `TORR`) y obliga a reentrenar o reexportar esos modelos. Hacerlo después de una versión de aviso: el runtime registra un aviso al abrir una sección v1 antes de retirarla.
5. **Documentación.** Corregir el párrafo «Claves v1 y v2» de «Compatibilidad» (`META.version` no decide nada) cuando se retire alguna de las anteriores.

Alternativa de bajo riesgo si Juan prefiere no quitar lectores: añadir al cargador un aviso (no un error) por cada sección en versión retirable, y dejar pasar un ciclo de despliegue antes de borrar.
