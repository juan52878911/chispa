# Hilo corto para redes (borrador, sin publicar)

Cada mensaje cabe en 280 caracteres. Cifras y fuentes: [`README.md`](README.md#cifras-que-se-pueden-usar).

## Español

1/6 Publicamos Chispa, un servidor de clasificación de texto de código abierto (Apache-2.0). Le das un texto y una pregunta
cerrada con sus opciones; te devuelve la opción, una probabilidad por opción y un aviso cuando no está seguro.
https://github.com/juan52878911/chispa

2/6 ¿Tienes datos etiquetados? Modo clásico: entrenas un .chispa por tarea y lo sirves en CPU. Macro-F1 de 0,814 en massive y
0,887 en clinc150, con p99 de 9–12 ms por consulta. Trae SDK para Android.

3/6 ¿Solo unos pocos ejemplos y sin GPU? Modo CPU, sin LLM: los ejemplos van en la petición y responde un kNN sobre e5. En choice,
0,483 en tareas selladas, a la par del Qwen3-4B sin ejemplos (0,485), con ≈ 9 ms de codificador.

4/6 ¿Pocos ejemplos y una GPU? Modo GPU: Qwen3-4B con 8 demostraciones elegidas por kNN. 0,501 global en 8 tareas selladas
(0,614 en choice). Ojo: la mejora sobre el mismo LLM sin ejemplos no es estadísticamente firme con 8 tareas.

5/6 Cada cifra enlaza su fichero de resultados, con intervalos de confianza, y el README lista lo que probamos y no funcionó. Lo que
no prometemos: aciertos sin ejemplos, ni score sin una rúbrica tuya.

6/6 Pruébalo en 5 minutos, solo con Go y Python y sin descargar modelos: `ejemplos/ferreteria/correr.sh`. Si lo pruebas con tus
datos, cuéntanos dónde falla.

## English

1/6 We're releasing Chispa, an open-source (Apache-2.0) text classification server. Send a text and a closed question with its
options; get back the option, a probability per option and a flag when it isn't sure.
https://github.com/juan52878911/chispa

2/6 Have labelled data? Classic mode: train one .chispa per task and serve it on CPU. Macro-F1 0.814 on massive and 0.887 on
clinc150, p99 9–12 ms per query. Ships an Android SDK.

3/6 Only a few examples and no GPU? CPU mode, no LLM: examples go in the request and a kNN over e5 answers. On choice, 0.483 on
sealed tasks, on par with Qwen3-4B without examples (0.485), with a ≈ 9 ms encoder.

4/6 A few examples and a GPU? GPU mode: Qwen3-4B with 8 demonstrations picked by kNN. 0.501 overall on 8 sealed tasks (0.614 on
choice). Caveat: the gain over the same LLM without examples is not statistically firm with 8 tasks.

5/6 Every number links its results file, with confidence intervals, and the README lists what we tried that didn't work. What we
don't promise: accuracy without examples, or score without your own rubric.

6/6 Try it in 5 minutes with just Go and Python, no model downloads: `ejemplos/ferreteria/correr.sh`. If you try it on your data,
tell us where it breaks.
