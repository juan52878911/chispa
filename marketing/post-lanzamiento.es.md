# Chispa: clasificación de texto con tus ejemplos, en tu máquina y con cifras que puedes revisar

*Borrador del post de lanzamiento. Sin publicar. Cifras y fuentes: [`README.md`](README.md#cifras-que-se-pueden-usar).*

Hay una tarea que se repite en casi cualquier equipo que recibe texto: leer un mensaje corto y ponerle una etiqueta cerrada.
¿Es una queja, una consulta o una petición? ¿Qué área lo atiende? ¿Hay que escalarlo? Hoy publicamos **Chispa**, un servidor de
clasificación tipada de código abierto (Apache-2.0) pensado justo para eso.

## Qué hace

Le mandas un texto y una o varias preguntas cerradas con sus opciones:

```json
{"state": "Ya hice el pago, ¿a qué hora me llega el domicilio?",
 "questions": [{"id": "q1", "question": "¿Qué necesita la persona?", "kind": "choice",
                "options": [{"id": "cotizacion"}, {"id": "pedido"}, {"id": "garantia"}, {"id": "asesoria"}, {"id": "factura"}]}]}
```

y te devuelve la opción elegida, una probabilidad por opción, la capa que contestó y un campo `escalate`. Ese último importa:
cuando la confianza no llega al umbral, Chispa lo dice en vez de fingir, y tú decides qué hacer con ese caso (mandarlo a una
persona, a otro modelo o a una cola).

Hay tres tipos de pregunta: `choice` (elegir una opción), sí/no y `score` (un nivel ordinal; este es experimental).

## Tres modos, según lo que tengas

**Si tienes datos etiquetados de tu tarea: el modo clásico.** Entrenas un fichero `.chispa` por tarea con `chispa train` y lo
sirves en CPU. Por dentro es una cascada: una capa lineal muy barata contesta lo fácil y, si duda, pasa a un codificador e5. Con
el train de massive y de clinc150, por el servidor real, dio **macro-F1 de 0,814 y 0,887** con **p99 de 9 a 12 ms por consulta**
en una CPU de nube. Trae SDK para Android y un motor móvil en Go para correr en el teléfono (la latencia en móvil aún no la
hemos medido, y el SDK de iOS todavía no se ha compilado).

**Si solo tienes unos pocos ejemplos y no tienes GPU: el modo CPU.** Sin LLM. Mandas tus ejemplos dentro de la propia petición
(`examples[]`) y Chispa responde con un kNN sobre vectores e5. En 4 tareas `choice` selladas, con 8 ejemplos por clase, sacó
**0,483** de exactitud macro: prácticamente lo mismo que el Qwen3-4B sin ejemplos (**0,485**), pero con un codificador de
**≈ 9 ms** por consulta en CPU.

**Si tienes pocos ejemplos y una GPU: el modo GPU.** Qwen3-4B por logprobs, con 8 demostraciones que el kNN elige para cada
consulta. En las 8 tareas selladas: **0,501** global y **0,614** en `choice`.

## Por qué puedes fiarte de las cifras (y de sus límites)

Cada número del README enlaza su fichero de resultados en el repositorio. Las tareas «selladas» se abrieron una sola vez, con la
configuración congelada de antemano, y los resultados llevan intervalos de confianza. También publicamos lo que no funcionó: una
lista de hipótesis medidas y refutadas (destilación, cascadas sin datos, LLM más pequeños, cuantizaciones que perdían demasiado).

Y lo que no prometemos:

- Chispa **no** acierta sin ejemplos: el zero-shot del modo clásico no vende.
- La mejora del modo GPU sobre el mismo LLM sin ejemplos (+5,1 puntos globales) **no es estadísticamente firme** con 8 tareas
  (IC95 [−4,3, +15,5]).
- `score` es experimental y solo ayuda con una rúbrica por nivel escrita por ti.
- Los SDK todavía no están publicados en ningún registro: se usan desde el repositorio.

## Pruébalo en 5 minutos

Solo necesitas Go y Python; no se descarga ningún modelo:

```sh
git clone https://github.com/juan52878911/chispa && cd chispa
ejemplos/ferreteria/correr.sh
```

Entrena un `.chispa` con 50 ejemplos inventados de una ferretería, lo sirve y clasifica 5 mensajes nuevos. Acierta los 5, pero
los 5 salen con `escalate: true`: con 10 ejemplos por opción la confianza no llega al umbral, y Chispa lo avisa. Para que conteste
solo hacen falta datos reales de tu tarea. La salida esperada está en [`demo.md`](demo.md).

## Lo que viene

Confirmar las cifras del modo GPU con una batería nueva sellada, medir la latencia en Android e iOS y publicar los SDK. Si
clasificas texto corto y quieres probarlo con tus datos, abre un issue: nos interesa saber dónde falla.

Repositorio: <https://github.com/juan52878911/chispa> · Licencia Apache-2.0
