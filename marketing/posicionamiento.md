# Mensaje y posicionamiento (borrador)

Cifras y fuentes: [`README.md`](README.md#cifras-que-se-pueden-usar) de esta carpeta.

## Español

**Una frase.** Chispa es un servidor de clasificación tipada de código abierto: le das un texto y una pregunta cerrada con sus
opciones, y te devuelve la opción, una probabilidad por opción y un aviso cuando no está seguro.

**El problema.** Muchos equipos tienen que poner una etiqueta cerrada a miles de textos cortos (tickets, PQRS, correos, chats):
¿es una queja o una consulta?, ¿qué área lo atiende?, ¿es spam? Mandarlo todo a un LLM grande es caro, lento y saca los datos de
casa; entrenar un modelo propio parece un proyecto de meses.

**Lo que ofrece Chispa.**

- **Corre donde tú quieras, con tus ejemplos.** Servidor en Go, licencia Apache-2.0 para código y pesos, en tu máquina o en tu nube.
- **Tres formas de usarlo, según lo que tengas:**
  - *Modo clásico* (datos propios, solo CPU o móvil): un `.chispa` entrenado por tarea. Macro-F1 de 0,814 en massive y 0,887 en
    clinc150, con p99 de 9–12 ms por consulta en CPU. Incluye SDK para Android y un motor móvil en Go.
  - *Modo CPU* (pocos ejemplos, sin GPU ni LLM): kNN sobre e5 con los ejemplos dentro de la petición. En `choice`, 0,483 en tareas
    selladas, a la par del Qwen3-4B sin ejemplos (0,485), con un codificador de ≈ 9 ms por consulta.
  - *Modo GPU* (pocos ejemplos y una GPU): Qwen3-4B con 8 demostraciones elegidas por kNN. 0,501 global en las 8 tareas selladas
    (0,614 en `choice`).
- **Dice cuándo no sabe.** Cada respuesta trae `escalate`: si la confianza no llega al umbral, lo avisa en vez de inventar.
- **Cifras con fuente.** Cada número del README enlaza su fichero de resultados, con intervalos de confianza y una lista de lo
  que se probó y no funcionó.

**Para quién.** Equipos de producto y de datos que clasifican texto corto, quieren controlar dónde
corren sus datos y prefieren un componente pequeño y medible a una API.

**Lo que no decimos.** No es un modelo que acierte sin ejemplos; `score` es experimental y solo ayuda con una rúbrica escrita por
ti; la ventaja del modo GPU sobre el mismo LLM sin ejemplos (+5,1) no es estadísticamente firme con 8 tareas; la latencia en móvil
está sin medir; los SDK no están publicados en ningún registro todavía.

**Llamada a la acción.** «Pruébalo en 5 minutos en tu portátil: `ejemplos/ferreteria/correr.sh`.»

## English

**One line.** Chispa is an open-source typed classification server: send a text and a closed question with its options, get back
the chosen option, a probability per option, and a flag when it is not sure.

**The problem.** Many teams need to put a closed label on thousands of short texts (tickets, citizen requests, emails, chats).
Sending everything to a large LLM is costly, slow and moves the data out; training your own model sounds like a months-long project.

**What Chispa offers.**

- **Runs where you want, from your examples.** Go server, Apache-2.0 for code and weights, on your machine or your cloud.
- **Three ways to use it, depending on what you have:**
  - *Classic mode* (your own labelled data, CPU or mobile): a `.chispa` model trained per task. Macro-F1 0.814 on massive and
    0.887 on clinc150, p99 9–12 ms per query on CPU. Ships an Android SDK and a Go mobile engine.
  - *CPU mode* (a few examples, no GPU, no LLM): kNN over e5 embeddings with the examples inside the request. On `choice`, 0.483 on
    sealed tasks, on par with Qwen3-4B without examples (0.485), with an encoder at ≈ 9 ms per query.
  - *GPU mode* (a few examples and a GPU): Qwen3-4B with 8 demonstrations picked by kNN. 0.501 overall on 8 sealed tasks (0.614 on
    `choice`).
- **Says when it does not know.** Every answer carries `escalate`: below the confidence threshold it tells you instead of guessing.
- **Numbers with sources.** Every number in the README links its results file, with confidence intervals and a list of what was
  tried and did not work.

**Who it is for.** Product and data teams classifying short text who want control over where
their data runs and prefer a small, measurable component to an API.

**What we do not claim.** It does not classify well without examples; `score` is experimental and needs a rubric you write; the
GPU-mode gain over the same LLM without examples (+5.1) is not statistically firm with 8 tasks; mobile latency is not measured; the
SDKs are not published to any registry yet.

**Call to action.** "Try it on your laptop in 5 minutes: `ejemplos/ferreteria/correr.sh`."
