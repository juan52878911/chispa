# L2x frente a la batería no vista

| conjunto | n | l2x | l2 | cruzado | knn-e5 |
|---|---|---|---|---|---|
| hatecheck-es (única tarea de verdad no vista) | 2000 | 0.537 [0.513–0.557] | 0.514 [0.492–0.535] | 0.616 [0.594–0.636] | 0.693 [0.671–0.713] |
| inyecciones | 116 | 0.517 [0.431–0.603] | 0.491 [0.397–0.578] | 0.707 [0.621–0.793] | 0.767 [0.690–0.845] |
| jailbreak | 400 | 0.618 [0.570–0.665] | 0.652 [0.608–0.700] | 0.665 [0.618–0.708] | 0.565 [0.517–0.613] |
| massive-es | 1503 | 0.486 [0.459–0.510] | 0.411 [0.386–0.436] | 0.454 [0.429–0.478] | 0.500 [0.474–0.526] |
| massive-esc | 1438 | 0.531 [0.505–0.558] | 0.563 [0.538–0.588] | 0.622 [0.596–0.647] | 0.570 [0.543–0.595] |
| tuits-es | 1194 | 0.539 [0.511–0.567] | 0.554 [0.527–0.580] | 0.554 [0.526–0.583] | 0.482 [0.453–0.511] |
| jevbench | 231 | 0.506 [0.437–0.567] | 0.498 [0.433–0.563] | 0.511 [0.446–0.576] | – |
| **agrupada (6 tareas)** | 6651 | 0.529 [0.517–0.541] | 0.516 [0.504–0.528] | 0.574 [0.562–0.586] | 0.578 [0.566–0.590] |
| **macro-F1 medio** | | 0.495 | 0.448 | 0.596 | 0.573 |

## McNemar pareado (L2x frente a cada uno)

- l2x - l2, hatecheck-es: diferencia +2.3 pts [-0.2; +4.7], p = 0.077
- l2x - l2, inyecciones: diferencia +2.6 pts [-5.2; +11.2], p = 0.6776
- l2x - l2, jailbreak: diferencia -3.5 pts [-6.2; -1.2], p = 0.0094
- l2x - l2, massive-es: diferencia +7.4 pts [+4.5; +10.2], p = 0.0
- l2x - l2, massive-esc: diferencia -3.1 pts [-6.0; -0.4], p = 0.0331
- l2x - l2, tuits-es: diferencia -1.4 pts [-4.4; +1.7], p = 0.3945
- l2x - l2, jevbench: diferencia +0.9 pts [-7.8; +8.7], p = 0.9196
- l2x - l2, AGRUPADA: diferencia +1.3 pts [-0.1; +2.6], p = 0.060635
- l2x - cruzado, hatecheck-es: diferencia -7.8 pts [-9.8; -6.0], p = 0.0
- l2x - cruzado, inyecciones: diferencia -19.0 pts [-27.6; -10.3], p = 0.0001
- l2x - cruzado, jailbreak: diferencia -4.8 pts [-12.0; +2.8], p = 0.2321
- l2x - cruzado, massive-es: diferencia +3.2 pts [+0.6; +5.8], p = 0.015
- l2x - cruzado, massive-esc: diferencia -9.0 pts [-11.3; -7.0], p = 0.0
- l2x - cruzado, tuits-es: diferencia -1.5 pts [-4.2; +1.2], p = 0.2972
- l2x - cruzado, jevbench: diferencia -0.4 pts [-8.2; +7.4], p = 1.0
- l2x - cruzado, AGRUPADA: diferencia -4.5 pts [-5.6; -3.4], p = 0.0

## Criterios

- C1 macro-F1 medio (6 tareas) >= L2 + 12 pts: NO PASA (valor 0.4952, umbral 0.5683)
- C2 macro-F1 medio >= cruzado (hito 4) - 1 pt: NO PASA (valor 0.4952, umbral 0.5857)
- C3 exactitud agrupada >= cruzado - 1 pt: NO PASA (valor 0.5291, umbral 0.5639)
- C4 AUROC de la confianza >= 0,78: NO PASA (valor 0.6794816099575485, umbral 0.78)
- C5 latencia p50 <= 60 ms en el i7 (int8, 4 hilos, <= 8 opciones): PENDIENTE (valor None, umbral 60)
- Refutación: SÍ (hatecheck-es, inyecciones, jailbreak, massive-esc, tuits-es)
