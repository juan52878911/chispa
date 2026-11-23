# Ejemplos

## `ferreteria/`: el modo clásico en 5 minutos

Clasificar los mensajes que llegan a una ferretería colombiana en 5 opciones (`cotizacion`, `pedido`, `garantia`, `asesoria`,
`factura`). El esquema y los datos son **inventados** para esta demo: no es una tarea de medida ni una cifra de calidad.

Requisitos: Go ≥ 1.27 y Python ≥ 3.9 (sin paquetes extra). Solo CPU; no se descarga ningún modelo. La primera vez, Go baja el
módulo de [kindling](https://github.com/juan52878911/kindling) (la capa L0) si no está en su caché.

```sh
ejemplos/ferreteria/correr.sh          # desde la raíz del repositorio; PUERTO=8091 para cambiar el puerto
```

Qué hace:

1. `chispa train` entrena un `.chispa` con [`esquema.json`](ferreteria/esquema.json) y los 50 ejemplos de
   [`ejemplos.csv`](ferreteria/ejemplos.csv) (10 por opción). Con 10 o más por opción entrena **L0** (TF-IDF de caracteres con
   regresión logística). Sin `--l1` ni L2, el fichero solo lleva META, L0 y CALIB (unos 9 KB).
2. Compila `chispa-server` y lo arranca en `127.0.0.1:8090` con ese `.chispa`.
3. Manda por `POST /v1/decide` los 5 mensajes de [`pruebas.txt`](ferreteria/pruebas.txt), que no están en los ejemplos.

Salida esperada (medida en un Mac M-series el 2026-11-23; las respuestas y probabilidades son deterministas, `ms` varía):

```
1/3 Entrenando ferreteria.chispa (L0: TF-IDF de caracteres + logística)…
ferreteria.chispa: META+L0+CALIB, 9077 bytes, L0 validación 0.500, 0.9 s
2/3 Compilando y arrancando chispa-server en 127.0.0.1:8090…
3/3 Clasificando pruebas.txt (mensajes que no están en los ejemplos):
  cotizacion  p=0.33 escalar escalate=true  ms=0.029  «¿Cuánto me sale el bulto de cemento gris de 50 kg?»
  pedido      p=0.38 escalar escalate=true  ms=0.009  «Ya hice el pago, ¿a qué hora me llega el domicilio?»
  garantia    p=0.49 escalar escalate=true  ms=0.008  «La pulidora no prende, la compré hace una semana»
  asesoria    p=0.52 escalar escalate=true  ms=0.011  «¿Qué pintura me sirve para una pared exterior que le da el sol?»
  factura     p=0.65 escalar escalate=true  ms=0.014  «Me hace falta la factura de la compra con el NIT de mi empresa»
```

Cómo leerla:

- `answer` es la opción elegida y `p` su probabilidad; la respuesta trae además una probabilidad por opción (`probs`).
- **Las 5 escalan** (`layer: escalar`, `escalate: true`): la confianza de L0 no llega a su umbral (τ = 0,8 por defecto) y Chispa
  te lo dice en vez de fingir. `answer` sigue siendo su mejor pista, y aquí acierta las 5, pero con 5 mensajes eso no prueba nada.
- La «validación 0.500» es la exactitud de L0 sobre la parte de los 50 ejemplos que `chispa train` aparta: con 10 por opción es
  poco y ruidosa.
- Para que el modo clásico conteste solo, entrénalo con datos reales de tu tarea: con el train completo de massive y clinc150 dio
  macro-F1 de 0,814 y 0,887 con p99 de 9–12 ms en CPU ([fuente](../server/deploy/resultados/e2e-20261028/LEEME.md)), usando además
  L2 (`--conjunta`, ≥ 20 ejemplos por opción). Ver [`docs/02-inicio-rapido.md`](../docs/02-inicio-rapido.md).

La misma petición a mano, con el servidor en marcha:

```sh
curl -s 127.0.0.1:8090/v1/decide -d '{"state":"¿Cuánto me sale el bulto de cemento gris de 50 kg?","questions":[{"id":"q1","question":"¿Qué necesita la persona que escribe a la ferretería?","kind":"choice","options":[{"id":"cotizacion"},{"id":"pedido"},{"id":"garantia"},{"id":"asesoria"},{"id":"factura"}]}]}'
```

## Los modos CPU y GPU

Necesitan artefactos que no van en el repositorio (el ONNX de e5 y, en modo GPU, vLLM con Qwen3-4B). Guía, configs y contrato en
[`server/PRODUCTO.md`](../server/PRODUCTO.md); las configs de arranque son `server/deploy/config.modo-cpu.json` y
`server/deploy/config.modo-gpu.json`. En esos modos los ejemplos pueden ir dentro de la petición (`examples[]`), sin entrenar nada.
