# archivo

Material histórico que ya no usa el producto ni las pruebas. Se conserva con `git mv` (el historial sigue), sin borrar nada.

## `model-alt/`

Carpeta `model/alt/` movida aquí en la parte que es claramente histórica:

- `resultados/` (`g5-57edf3c`, `g5-41e8e2a`): salidas del hito 4 (cross-encoder frente a Chispa, destilación, latencia, sondeo). Las cifras vigentes están en `model/alt/resultados-20261113-p0-p2/LEEME.md`.

## Qué se quedó en `model/alt/` y por qué

Todo lo demás de `model/alt/` sigue en su sitio porque lo usa algo vivo o hay duda:

- El servidor Go (`server/`) cita sus scripts y los criterios `crit-*.json` en tests de paridad, en `Dockerfile.producto` y en la documentación; `runtime/tok` y `model/datos` (batería v3) también importan o citan módulos de `model.alt`.
- `aws/` y `techo_local/` son los ud vivos (W1 fs8, L2-qwen3emb, techo local) y empaquetan casi todos los módulos Python; los `crit-*.json` están congelados.
- Los tests de `model/alt/tests` cubren esos módulos.
- Los resultados (`resultados-20261113-p0-p2/`) y su LEEME los enlazan el README y la documentación.
