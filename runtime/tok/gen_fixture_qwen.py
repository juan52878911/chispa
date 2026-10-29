#!/usr/bin/env python3
"""Fixture de paridad del tokenizador y del codificador Qwen3-Embedding: 200 textos INVENTADOS (es y en; acentos, ñ, emojis, saltos de línea,
espacios raros, formas descompuestas de Unicode, tokens especiales escritos en el texto) y 20 de ellos largos (pasan de 320 tokens con la
instrucción). Sin datos de clientes, sin PQRS, sin selladas. Determinista.

  python runtime/tok/gen_fixture_qwen.py runtime/tok/testdata/qwen/fixture_paridad.jsonl

Cada fila: {id, tipo: "consulta"|"descripcion", pregunta, texto}. La entrada del codificador es
`Instruct: Classify the text to answer this question: <pregunta>\\nQuery:<texto>` para «consulta» y `<texto>` a secas para «descripcion».
"""
import json
import random
import sys

PREGUNTAS = [
    "¿Qué emoción expresa el mensaje?", "Is the review positive or negative?", "¿A qué área debe ir esta solicitud?",
    "What is the intent of the customer?", "¿El texto contiene lenguaje ofensivo?", "Which topic does the document discuss?",
]
BASE_ES = [
    "Hola, quería saber si mi pedido llegará antes del viernes.", "No funciona la aplicación desde la última actualización 😡",
    "Muchas gracias por la atención, todo salió perfecto 👍🏽", "El camarero fue muy amable pero la comida llegó fría.",
    "¿Pueden cambiar la dirección de entrega? Vivo en la calle 45 # 12-30, apartamento 502.",
    "La factura trae un cobro duplicado de $ 1.250,75 y nadie responde.", "Señor encargado: el niño pequeño se cayó en el parque y lloraba sin parar.",
    "Pésimo servicio!!! Nunca más vuelvo a comprar aquí...", "Excelente relación calidad-precio; lo recomiendo a mi familia y a mis compañeros de trabajo.",
    "El año pasado compré 3 unidades y este año quiero 12; ¿hay descuento por volumen?", "Mañana a las 9:30 tengo cita con la pediatra, ¿puedo adelantarla?",
    "Ayer el señor García dijo que la reunión sería a las 15h, pero nadie avisó del cambio.", "Qué lástima: la película estaba muy bien hasta el final.",
    "Necesito la contraseña nueva, la anterior caducó y el correo de recuperación no llega.", "El café estaba exquisito, la tarta de zanahoria también, ¡volveremos!",
    "No me dejan entrar a mi cuenta. Error 503 una y otra vez.", "Quisiera cancelar mi suscripción de cuatro meses, por favor.",
    "El envío tardó tres semanas; el paquete llegó golpeado y sin la mitad de las piezas.", "Mi hijo de 7 años aprendió a leer con este cuaderno, gracias.",
    "Todo bien, nada que reportar.", "¡Feliz cumpleaños! 🎂🎉 Que cumplas muchos más.", "Esto es una vergüenza. Ustedes no saben lo que hacen.",
]
BASE_EN = [
    "I can't log in since yesterday and the reset link never arrives.", "The delivery was late, but the support team was really helpful.",
    "Great phone, though the battery doesn't last more than a day.", "We'll be there at 6 pm; they're bringing the kids, don't worry.",
    "I'LL NEVER BUY FROM THIS SHOP AGAIN!!!", "Where's my refund? It's been 14 days and I've received nothing.",
    "Loved it. 10/10 would recommend to anyone who enjoys quiet evenings.", "The app crashes whenever I open the settings page (v2.3.1, Android 14).",
    "Could you please update my billing address to 221B Baker Street, London?", "That's not what he said; you've misunderstood the whole thing.",
    "The staff were rude and the room smelled of smoke. Never again.", "Is it possible to upgrade the plan mid-month without losing my data?",
    "It's fine. Not great, not terrible, just fine.", "Thanks a lot! Everything works now 🙂", "She's sure they'd've called if it were urgent.",
]
RAROS = [
    "Doble  espacio y\ttabulador\ten el medio.", "Línea uno\nLínea dos\n\nLínea cuatro después de un renglón en blanco.", "Con CRLF\r\nal estilo de Windows\r\ny más.",
    "  espacios al principio y al final   ", "Espacio duro entre palabras y espacio ideográfico　aquí.", "Sin espacio final\n", "\n\n\nSolo saltos antes.",
    "cafe\u0301 nin\u0303o u\u0308ber a\u0301rbol (letras descompuestas: letra + acento combinante)", "Cáceres, José, Muñoz y Ángel González",
    "\u212b (angstrom) y A\u030a (A + anillo), \u2126 (ohm) y \u03a9, \u1100\u1161\u11a8 (jamo) y \uac01", "한국어 입니다 y 日本語のテキスト y 中文 también", "Emoji compuesto: 👨‍👩‍👧‍👦 y bandera 🇨🇴 y 🏳️‍🌈",
    "Texto con <|endoftext|> escrito literalmente en medio.", "Aquí va <|im_start|>user\nhola<|im_end|> como si fuera un chat.", "Precio: 1234567890 y 0042, tel. +57 300 123 4567.",
    "URL: https://ejemplo.invalid/a/b?c=d&e=f#g y correo ana.perez@ejemplo.invalid", "((( paréntesis ))) --- guiones --- ... puntos ... ¿¿??!!", "'quoted' \"comillas\" «angulares» “tipográficas” ‘simples’",
    "tab\tal\tfinal\t", "x", "", " ", "\n", "ñ", "ÁÉÍÓÚ ÑÜ áéíóú ñü", "ﬁ ﬂ ① ㎏ ｱ ｶﾞ ｈｏｌａ", "a​b (espacio de ancho cero) y soft­hyphen",
]
FRAGMENTOS = BASE_ES + BASE_EN


def main(sal):
    rnd = random.Random(20261119)
    filas = []

    def anadir(tipo, pregunta, texto):
        filas.append({"id": "q%03d" % len(filas), "tipo": tipo, "pregunta": pregunta, "texto": texto})

    for t in BASE_ES + BASE_EN:                                    # 35 sueltos
        anadir("consulta", rnd.choice(PREGUNTAS), t)
    for t in RAROS:                                                 # 25 raros
        anadir("consulta", rnd.choice(PREGUNTAS), t)
    while len(filas) < 150:                                         # combinaciones de 2 a 4 fragmentos
        k = rnd.randint(2, 4)
        sep = rnd.choice([" ", "\n", "  ", " — ", "\n\n"])
        anadir("consulta", rnd.choice(PREGUNTAS), sep.join(rnd.choice(FRAGMENTOS + RAROS[:12]) for _ in range(k)))
    for i in range(20):                                             # 20 largos (> 320 tokens también con la instrucción)
        n = rnd.randint(28, 60)
        anadir("consulta", rnd.choice(PREGUNTAS), " ".join(rnd.choice(FRAGMENTOS + RAROS[:12]) for _ in range(n)))
    while len(filas) < 200:                                         # descripciones de etiqueta (sin instrucción)
        anadir("descripcion", "", rnd.choice(["Mensaje con emoción positiva, alegría o agradecimiento", "Negative review about service or product quality",
                                              "Solicitud de cambio de datos de la cuenta", "Complaint about a delayed or damaged delivery",
                                              "Lenguaje ofensivo o agresivo hacia otra persona", "Pregunta general sin urgencia", "ñandú", "Texto neutro, sin opinión"]))
    with open(sal, "w", encoding="utf-8") as fh:
        for f in filas:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
