# Marketing de Chispa (borradores, sin publicar)

> [!WARNING]
> Todo lo de esta carpeta son **borradores**. Nada está publicado: ni el post, ni el hilo, ni la descripción del repo.
> Publicar, cambiar la visibilidad del repo o tocar su descripción en GitHub lo decide Juan.

| Pieza | ES | EN |
|---|---|---|
| Mensaje y posicionamiento (una página) | [`posicionamiento.md`](posicionamiento.md) | misma página, segunda mitad |
| Post de lanzamiento (blog) | [`post-lanzamiento.es.md`](post-lanzamiento.es.md) | [`post-lanzamiento.en.md`](post-lanzamiento.en.md) |
| Hilo corto para redes | [`hilo.md`](hilo.md) | misma página |
| Descripción y topics del repo de GitHub | [`github.md`](github.md) | misma página |
| Demo reproducible (comando + salida) | [`demo.md`](demo.md) | misma página |
| Auditoría «listo para el público» (interna) | [`auditoria-publico.md`](auditoria-publico.md) | — |

## Reglas de honestidad (para cualquier pieza nueva)

1. Cada cifra sale de la tabla de abajo, con su fuente. Si no está en la tabla, no se usa.
2. Nada de «el mejor», «SOTA», «state of the art», «líder», «revolucionario» ni comparaciones con modelos de API.
3. No se promete zero-shot del modo clásico (no vende), ni `score` sin rúbrica (es experimental), ni latencia en móvil (sin medir).
4. Cuando una diferencia no es estadísticamente firme, se dice (GPU frente al 4B sin ejemplos: +5,1 con IC95 [−4,3, +15,5]).
5. Se distingue «sellada» (una sola apertura, 8 tareas) de «dev» (indicativa).

## Cifras que se pueden usar

| Cifra | Qué es | Fuente en el repo |
|---|---|---|
| **0,814** / **0,887** | macro-F1 del modo clásico por `/v1/decide`, cascada de producción (`--conjunta`), entrenado con el train de massive / clinc150 | [`server/deploy/resultados/e2e-20261028/LEEME.md`](../server/deploy/resultados/e2e-20261028/LEEME.md) (tabla de T2.6c) |
| **p99 9–12 ms** | latencia por consulta del modo clásico, c6a.xlarge (CPU), 1 cliente; L0 contesta el 54–67 % | misma fuente; [`README.md`](../README.md#modo-clásico-congelado) |
| SDK móvil | Android (Kotlin) y motor `gomobile`; iOS (Swift) **sin compilar**; latencia en móvil **sin medir**; ningún paquete publicado | [`README.md`](../README.md#sdk), [`sdk/README.md`](../sdk/README.md) |
| **0,483** | exactitud macro en `choice`, modo CPU (kNN-e5, k = 8, 8 ejemplos por clase), 4 tareas selladas, IC95 [0,38, 0,60] | [`README.md`](../README.md#selladas-una-sola-apertura-8-tareas), LEEME §17.26 |
| **0,485** | lo mismo con Qwen3-4B sin ejemplos (zero-shot): empata con el kNN | ídem |
| **≈ 9 ms** | codificador e5 por consulta en CPU (8,6 ms en serie); kNN ≈ 1 ms | [`server/PRODUCTO.md`](../server/PRODUCTO.md) («Cifras medidas (R10)») |
| **0,501** | exactitud macro global (8 tareas selladas) del modo GPU: Qwen3-4B fs8 con demos por kNN | [`README.md`](../README.md#selladas-una-sola-apertura-8-tareas), LEEME §17.26 |
| **0,614** | lo mismo solo en `choice` (4 tareas), IC95 [0,43, 0,77] | ídem |
| 0,451 / 0,128 | global del 4B sin ejemplos / de la clase mayoritaria (referencias) | ídem |
| +5,1 [−4,3, +15,5] | GPU frente al 4B sin ejemplos, global: **no es firme** | ídem |
| 10,6–12,4 pet./s | caudal del modo GPU por g5 (A10G) con la mezcla real, limitado por vLLM | [`server/PRODUCTO.md`](../server/PRODUCTO.md) |

LEEME = [`model/alt/resultados-20261113-p0-p2/LEEME.md`](../model/alt/resultados-20261113-p0-p2/LEEME.md).
