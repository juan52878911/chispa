# Demo reproducible / Reproducible demo

## Español

Modo clásico, solo CPU, sin descargar modelos. Requisitos: Go ≥ 1.27 y Python ≥ 3.9. Tarda unos 5 s en un portátil reciente
(medido: 4,4 s en un Mac M-series, con el módulo de Go ya en caché).

```sh
git clone https://github.com/juan52878911/chispa && cd chispa
ejemplos/ferreteria/correr.sh
```

Salida esperada (respuestas y probabilidades deterministas; `ms` cambia en cada corrida):

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

Qué enseñar al presentarla: las 5 respuestas son correctas, pero **las 5 escalan**. Con 10 ejemplos por opción la confianza no
llega al umbral y Chispa lo dice. Es el mensaje del producto: no finge. Para que conteste solo hacen falta datos reales (el 0,814 y
el 0,887 de macro-F1 son con el train completo de massive y clinc150). Los `ms` son el tiempo de decisión dentro del servidor
(L0), sin red ni el arranque; no son la cifra de latencia del producto (p99 de 9–12 ms, medida con L2 en CPU de nube).

Detalle de cada paso: [`ejemplos/README.md`](../ejemplos/README.md).

## English

Classic mode, CPU only, no model downloads. Requirements: Go ≥ 1.27 and Python ≥ 3.9. About 5 s on a recent laptop (measured:
4.4 s on an M-series Mac with the Go module already cached).

```sh
git clone https://github.com/juan52878911/chispa && cd chispa
ejemplos/ferreteria/correr.sh
```

The expected output is the block above (answers and probabilities are deterministic; `ms` changes per run). The messages are in
Spanish: a quote request, an order, a warranty claim, a how-to question and an invoice request.

What to point out when presenting it: all 5 answers are right, but **all 5 escalate**. With 10 examples per option confidence stays
below the threshold, and Chispa says so — that is the product message: it does not pretend. Answering on its own needs real data
(the 0.814 and 0.887 macro-F1 use the full massive and clinc150 train sets). The `ms` values are the in-server decision time (L0),
without network or startup; they are not the product latency figure (p99 9–12 ms, measured with L2 on a cloud CPU).
