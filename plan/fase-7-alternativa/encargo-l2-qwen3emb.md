# Encargo: Qwen3-Embedding-0.6B como L2 del servidor

Pedido por Juan («Sí, intégralo») a través de la sesión «LLM arquitectura y evolución», 2026-11-19.

## Objetivo y entregable
- La rama `l2-qwen3emb` (sin push), con el codificador L2 del servidor Go intercambiable por config.
  - `e5-small` sigue por defecto y es la vuelta atrás.
  - `qwen3-emb` es Qwen3-Embedding-0.6B en ONNX, con tokenizador BPE en Go.
- Las cifras de paridad, e2e y latencia de la corrida en c7i, en el LEEME.

## Criterio de aceptación (congelado)
- Fichero: `model/alt/crit-l2-qwen3emb.json`, sha256 `58a62d618a4f917c1c4c232934a31394874a59f4cb5b589e945efd91511ade0b`.
- Paridad: tokens 200/200 idénticos a transformers 5.22.2, y coseno ≥ 0,999 en cada texto del fixture.
- e2e en dev por HTTP: el macro del kNN a ≤ 0,5 puntos del Python de R11 con los mismos vecinos (v32 0,4675; v8 0,4683).
- Reglas de elección:
  - El int8 se usa solo si queda a ≤ 0,5 puntos del fp32.
  - Si la p50 en CPU pasa de 150 ms, la nota de PRODUCTO.md es «recomendado con GPU o cuando la latencia no importe».

## Línea base
- `llm-rondas-r10` (53e0415) más los 4 commits de R11 (736a001, la referencia en Python).
- No se fusiona con `integracion-uso-general`: eso lo decide Juan.

## Presupuesto
- Worker: ≤ 240 min y ≤ 25 $ de cuenta.
- AWS: ≤ 2 USD, solo c7i.xlarge (una g5 solo si hace falta para la referencia), con autoapagado. Los ud los lanza la sesión madre tras leerlos.

## Fuera de alcance
- Cambiar L3, la cascada o los umbrales de escalado.
- Abrir las selladas.
- Usar datos PQRS en el fixture.
- Hacer push o fusionar ramas.

## Decisiones que son de Juan
- Unir ramas y publicar.
- Cualquier gasto por encima de 2 USD.
- Cambiar el criterio.

## Cambios
| Fecha | Qué cambió | Por qué | OK |
|---|---|---|---|
