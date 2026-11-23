# Auditoría «listo para el público» (interna: no publicar)

Fecha: 2026-11-23. Base: `main` en 14a3e30 (2 373 ficheros versionados, 1 853 commits). Rama de trabajo: `publico-marketing`.
**No se ha borrado ni movido nada**: esta lista es una propuesta y Juan decide. Este fichero debería salir de la rama (o irse a
`archivo/`) antes de hacer público el repo.

## 1. Bloqueante: la historia de git

El **árbol actual está limpio** de maquina-trabajo (0 coincidencias, sin distinguir mayúsculas), pero **la historia no**:

- Dos commits del 2026-11-02/03 (`9772110`, `4d967b2`) añaden el nombre de una máquina con «maquina-trabajo» en
  `model/bench/pong/ajuste-v2-20261102.md` y `model/bench/pong/duelo-20261102.md`; `3bd7c92` lo quita del árbol, pero sigue en
  esos dos commits. Mensajes de commit, ramas y tags: limpios.
- El **id de la cuenta AWS** (12 dígitos, dentro del nombre del bucket de pruebas `s3://chispa-pruebas-<id>`) sale en **159
  ficheros del árbol** y en **206 commits**.
- Los dos **correos de autor** de los commits (uno de ellos distinto del de `pyproject.toml`/`package.json`) quedan expuestos.

Opciones (decisión de Juan):

- **A. Publicar un árbol nuevo sin historia** (un commit inicial con el estado limpio). Lo más simple y seguro; se pierde la
  historia pública (la privada se conserva en otro remoto).
- **B. Reescribir la historia** con `git filter-repo` (sustituir el nombre de la máquina, el id de cuenta y, con `.mailmap`, los
  correos). Conserva la historia, pero cambia todos los sha (y las cifras del LEEME citan sha de commits: se romperían esas
  referencias) y obliga a forzar el push.

Con cualquiera de las dos, el id de cuenta sigue en el árbol hasta que se sustituya en los 159 ficheros (casi todos
`model/alt/aws/`, `model/l2/`, `server/deploy/`, `plan/`). Propuesta: cambiarlo por `<cuenta>` con un `sed` en una rama aparte,
después de mover lo que se archive.

## 2. Secretos (agente `auditor-secretos`)

**0 credenciales** en árbol e historia: ni claves AWS (AKIA/ASIA), ni tokens de GitHub, HuggingFace, Anthropic u OpenAI, ni claves
privadas, ni `.env`. Las 13 «contraseñas» por asignación son falsos positivos (`max_tokens=…`, un `api_key` de test en
`sdk/python/tests/test_cliente.py`). Nada que rotar. Sin rutas `/Users/…` en árbol ni historia.

No comprobado: contraseñas escritas en prosa dentro de `plan/*.md` y los LEEME; unos pocos correos gmail dentro de fixtures de
reseñas/licencias y uno en `plan/fase-6-estabilidad/brief-yb-score-movil.md` (sin mirar uno a uno).

## 3. Lo que propongo quitar o mover a `archivo/`

| Ruta | Ficheros | Por qué | Propuesta |
|---|---|---|---|
| `CLAUDE.md`, `.delega`, `.claudeignore` | 3 | instrucciones y config de agentes; `CLAUDE.md` cita un plan local y reglas de presupuesto | quitar del repo público |
| `plan/` | 105 | briefs y encargos para agentes; citan nodos, cuentas, buckets e instancias | mover a `archivo/plan/` o quitar |
| `model/informes/fase-{0,1,2}.md` | 3 | cierres de fase con nombres de nodos y de la orquestación | `archivo/` |
| `deploy/kindling/` | 6 | infraestructura personal de microVM (incluye su propio `go.mod`) | `archivo/` o quitar (el README lo enlaza en «De dónde viene») |
| `arena/scripts/lima-upgrade-kindling.sh` | 1 | lleva el usuario local de la máquina | quitar o parametrizar |
| `model/alt/aws/lanzados/`, `model/alt/techo_local/lanzados/`, `model/datos/aws/lanzados/` | 45 | guiones de lanzamiento de EC2 con ids de instancia y el bucket | `archivo/` |
| `model/alt/aws/empaquetar_*.sh` y los `aws/` de `model/l2`, `model/bench`, `model/cascada`, `model/maestro`, `model/evaluacion`, `model/meta`, `model/datos`, `server/deploy/aws/`, `sdk/android/aws/` | 158 | scripts de AWS con el bucket y la cuenta | `archivo/`, salvo lo que citen los tests (revisar con `go test`/`pytest` después de mover) |
| `server/deploy/resultados/*.log`, `model/bench/carga/crudo/`, `model/bench/torre/resultados-20261029*` | ~30 | logs crudos de colas y corridas | `archivo/` (mantener los `LEEME.md` que cita el README) |
| `model/lineas_base/pred/` | 23 (≈ 22 MB) | predicciones por línea base, los ficheros más grandes del repo | `archivo/`, o regenerarlas con su script |
| `model/eval0/out/`, `model/evaluacion/resultados`, `model/meta/resultados`, `model/l2/resultados/ajuste-20261029/`, `model/l2/experimentos/` | 89 | resultados de experimentos superados | `archivo/` |
| `demos/pong`, `demos/serpiente`, `demos/tetris-ale` | 62 | demos de juegos que confunden a quien llega buscando un clasificador | `archivo/`; dejar `demos/caida` y `demo/wasm` (y valorar unificar `demo/` y `demos/`) |
| `archivo/model-alt/resultados/g5-2058ffa/estado.tsv` | 1 | lleva el bucket con el id de cuenta | limpiar la línea |

Para revisar antes de publicar (no sé si son publicables):

- `model/etiquetado/muestras/lote-1.jsonl` (2,2 MB): muestra etiquetada a mano. Confirmar que no hay PQRS reales ni datos personales.
- `model/herramientas/train.jsonl` (5,4 MB) y `test.jsonl` (1,9 MB): confirmar origen y licencia en `docs/licencias-inventario.md`.
- `model/datos/fixtures/…/sib200/`: SIB-200 figura en el README como «no se redistribuye»; comprobar que el fixture es lo bastante
  pequeño o quitarlo.
- `server/carga/textos-es-co.txt` (544 KB): la doc dice que son inventados; confirmar.

## 4. Ids de instancia en la documentación

Hay 34 ids de instancia EC2 (ya terminadas) en el árbol. Muchos están en los **títulos de sección** de
`model/alt/resultados-20261113-p0-p2/LEEME.md`, y por eso también en las **anclas de los enlaces del README** (p. ej. `#175-resultado-de-r10-caudal-ec2-i-…`).
Riesgo bajo (no dan acceso), pero identifican la cuenta. Propuesta: en una rama aparte, quitar los ids de los títulos del LEEME y
actualizar las anclas del README a la vez (`docs/comprobar-enlaces.py` lo verifica). No lo he hecho aquí porque cambia el
cuaderno de resultados.

## 5. Lo que ya está bien

- `LICENSE` (Apache-2.0) y `NOTICE` con atribuciones de terceros.
- Enlaces relativos del README y `docs/`: 0 rotos (`docs/comprobar-enlaces.py`).
- Ningún peso de modelo (`.bin`, `.safetensors`, `.gguf`) versionado; los 5 `.onnx` son fixtures sintéticos de test (≤ 4,4 MB).
- `make test`, `make smoke` y `go test ./...` en los módulos principales: en verde.

## 6. Lo que se añadió en la rama

`README.md` (secciones «Para quién», «Pruébalo en 5 minutos» y «Contribuir y seguridad»), `README.en.md`, `CONTRIBUTING.md`,
`SECURITY.md`, `CODE_OF_CONDUCT.md`, `.github/` (plantillas de issue y de PR), `ejemplos/` (demo de 5 minutos) y `marketing/`.

## 7. Decisiones para Juan

1. Historia: publicar sin historia (A) o reescribirla (B).
2. Qué hacer con cada fila de la tabla del §3 (y si `plan/` se archiva o se quita).
3. Sustituir el id de cuenta en el árbol y quitar los ids de instancia de los títulos del LEEME.
4. El correo de autor que figura en `sdk/python/pyproject.toml` y `sdk/js/package.json`, y el scope npm `@juanbedoya/chispa`: si
   son los públicos que quieres.
5. `SECURITY.md` usa el aviso privado de GitHub: hay que activarlo al hacer público el repo (o poner un correo de contacto).
6. La política de `CONTRIBUTING.md` para el modo clásico («solo arreglos de errores») y el plazo de 7 días de `SECURITY.md`.
7. El «Lo que viene» del post de lanzamiento (batería sellada nueva, latencia en móvil, publicar los SDK): ¿es el plan real?
