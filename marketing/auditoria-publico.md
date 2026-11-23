# Auditoría «listo para el público» (interna: no publicar)

Fecha: 2026-11-23. Base: `main` antes de la limpieza (2 373 ficheros versionados, 1 854 commits). Rama de trabajo: `publico-marketing`.
**No se ha borrado ni movido ningún fichero**: la tabla del §3 es una propuesta y Juan decide. Este fichero debería salir de la
rama (o irse a `archivo/`) antes de anunciar el repo.

## 1. Historia de git: reescrita (pendiente del push forzado)

Juan decidió conservar la historia y reescribirla con `git filter-repo` (`--replace-text`, `--replace-message`, `--mailmap`), en
un espejo aparte. Sustituciones (el mapa con los valores reales está fuera del repo):

- nombres de máquina de trabajo → `maquina-trabajo`; id de la cuenta AWS → `cuenta-aws` (el bucket queda
  `s3://chispa-pruebas-cuenta-aws`); cada id de instancia EC2 → `inst-NN`, igual en los títulos del LEEME y en las anclas del README;
  ids de grupo de seguridad, subred, VPC y AMI → `sg-ejemplo`, `subnet-ejemplo`, `vpc-ejemplo`, `ami-ejemplo`; el rol de EC2 → `rol-ec2-pruebas`;
- nombres de host y rutas del homelab → `maquina-i7`, `host-i7`, `nodo-kindling`, `/mnt/nfs-lab`; rutas y usuario locales → `usuario`;
- correos de autor y el de `pyproject.toml`/`package.json` → el noreply de GitHub.

Después, un commit en `main` cambia las citas de commits del árbol (contenido, anclas y nombres de fichero como
`ud-…-<sha>.sh`) por los sha nuevos. Recuentos y refs que hay que forzar: informe de la sesión de limpieza.

Lo que git no puede limpiar: las vistas en caché de GitHub de commits viejos (accesibles por sha hasta que GitHub haga su recolección;
se puede pedir a su soporte que la adelante), forks y clones que otros ya tengan, y cualquier copia indexada fuera de GitHub.

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
| `arena/scripts/lima-upgrade-kindling.sh` | 1 | script de la máquina local (el usuario ya está sustituido) | quitar o archivar |
| `model/alt/aws/lanzados/`, `model/alt/techo_local/lanzados/`, `model/datos/aws/lanzados/` | 45 | guiones de lanzamiento de EC2 con ids de instancia y el bucket | `archivo/` |
| `model/alt/aws/empaquetar_*.sh` y los `aws/` de `model/l2`, `model/bench`, `model/cascada`, `model/maestro`, `model/evaluacion`, `model/meta`, `model/datos`, `server/deploy/aws/`, `sdk/android/aws/` | 158 | scripts de AWS con el bucket y la cuenta | `archivo/`, salvo lo que citen los tests (revisar con `go test`/`pytest` después de mover) |
| `server/deploy/resultados/*.log`, `model/bench/carga/crudo/`, `model/bench/torre/resultados-20261029*` | ~30 | logs crudos de colas y corridas | `archivo/` (mantener los `LEEME.md` que cita el README) |
| `model/lineas_base/pred/` | 23 (≈ 22 MB) | predicciones por línea base, los ficheros más grandes del repo | `archivo/`, o regenerarlas con su script |
| `model/eval0/out/`, `model/evaluacion/resultados`, `model/meta/resultados`, `model/l2/resultados/ajuste-20261029/`, `model/l2/experimentos/` | 89 | resultados de experimentos superados | `archivo/` |
| `demos/pong`, `demos/serpiente`, `demos/tetris-ale` | 62 | demos de juegos que confunden a quien llega buscando un clasificador | `archivo/`; dejar `demos/caida` y `demo/wasm` (y valorar unificar `demo/` y `demos/`) |

Para revisar antes de publicar (no sé si son publicables):

- `model/etiquetado/muestras/lote-1.jsonl` (2,2 MB): muestra etiquetada a mano. Confirmar que no hay PQRS reales ni datos personales.
- `model/herramientas/train.jsonl` (5,4 MB) y `test.jsonl` (1,9 MB): confirmar origen y licencia en `docs/licencias-inventario.md`.
- `model/datos/fixtures/…/sib200/`: SIB-200 figura en el README como «no se redistribuye»; comprobar que el fixture es lo bastante
  pequeño o quitarlo.
- `server/carga/textos-es-co.txt` (544 KB): la doc dice que son inventados; confirmar.

## 4. Ids de instancia en la documentación

Resuelto con la reescritura: los 36 ids de instancia pasan a `inst-01` … `inst-36` en títulos y anclas a la vez, y
`docs/comprobar-enlaces.py` da 0 enlaces rotos.

## 5. Lo que ya está bien

- `LICENSE` (Apache-2.0) y `NOTICE` con atribuciones de terceros.
- Enlaces relativos del README y `docs/`: 0 rotos (`docs/comprobar-enlaces.py`).
- Ningún peso de modelo (`.bin`, `.safetensors`, `.gguf`) versionado; los 5 `.onnx` son fixtures sintéticos de test (≤ 4,4 MB).
- `make test`, `make smoke` y `go test ./...` en los módulos principales: en verde.

## 6. Lo que se añadió en la rama

`README.md` (secciones «Para quién», «Pruébalo en 5 minutos» y «Contribuir y seguridad»), `README.en.md`, `CONTRIBUTING.md`,
`SECURITY.md`, `CODE_OF_CONDUCT.md`, `.github/` (plantillas de issue y de PR), `ejemplos/` (demo de 5 minutos) y `marketing/`.

## 7. Decisiones para Juan

1. Revisar la historia reescrita y hacer el push forzado (lo hace la sesión principal con el OK de Juan).
2. Qué hacer con cada fila de la tabla del §3 (y si `plan/` se archiva o se quita).
3. Pedir a GitHub que purgue la caché de los commits viejos, si se quiere que dejen de verse por sha.
4. El scope npm `@juanbedoya/chispa`: si es el nombre público que quieres.
5. `SECURITY.md` usa el aviso privado de GitHub: el repo ya es público, así que hay que activarlo ya (o poner un correo de contacto).
6. La política de `CONTRIBUTING.md` para el modo clásico («solo arreglos de errores») y el plazo de 7 días de `SECURITY.md`.
7. El «Lo que viene» del post de lanzamiento (batería sellada nueva, latencia en móvil, publicar los SDK): ¿es el plan real?
