# Contribuir a Chispa

Gracias por mirar el proyecto. *English below.*

## Antes de abrir un issue o un PR

- **Errores**: abre un issue con la plantilla «Error» (versión o commit, sistema, comando exacto, salida y lo que esperabas).
- **Ideas y cambios grandes**: abre primero un issue con la plantilla «Propuesta» para hablarlo antes de escribir código.
- **Vulnerabilidades**: nunca en un issue público; sigue [`SECURITY.md`](SECURITY.md).
- **El modo clásico está congelado** (`.chispa` entrenado, cascada L0–L2, torres, SDK móvil): se aceptan arreglos de errores, no
  funciones nuevas. El trabajo nuevo va al servidor de dos modos ([`server/PRODUCTO.md`](server/PRODUCTO.md)).

## Cómo preparar un cambio

1. Haz un fork y una rama desde `main`.
2. Mantén el cambio pequeño y con un solo propósito; explica el porqué en el mensaje de commit.
3. Pasa las pruebas antes de pedir revisión, desde la raíz:

   ```sh
   make test                         # go vet + go test del banco adversarial (arena/)
   make smoke                        # entrena un modelo pequeño y lanza 50 ataques contra el servidor
   (cd server && go test ./...)      # y en cada módulo de Go que toques (hay un go.mod por carpeta)
   python3 docs/comprobar-enlaces.py # si tocas README.md o docs/
   ```

4. Si tocas Python, corre los tests de la carpeta que cambies (`python3 -m pytest sdk/python/tests`, por ejemplo).

## Reglas del proyecto

- **Cifras con fuente.** Toda cifra de calidad o de latencia lleva su fichero de resultados en el repositorio. Si no está medida,
  se escribe «sin medir». Los criterios se fijan antes de medir y no se mueven después.
- **Datos con licencia verificada.** No se aceptan datos con licencias -SA o -NC, ni datos de Twitter/X o Reddit. Las fuentes y sus
  licencias se apuntan en [`docs/licencias-inventario.md`](docs/licencias-inventario.md).
- **Nada de datos personales reales** en el repositorio, ni siquiera en fixtures: usa datos inventados o anonimizados.
- **Licencia.** Al contribuir aceptas que tu aporte se publica bajo Apache-2.0 ([`LICENSE`](LICENSE), sección 5).
- Normas de convivencia: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## Contributing (English)

- **Bugs**: open an issue with the "Bug" template (version or commit, OS, exact command, output, expected result).
- **Ideas and large changes**: open a "Proposal" issue first so we can discuss before code is written.
- **Security issues**: never in a public issue; see [`SECURITY.md`](SECURITY.md).
- **Classic mode is frozen**: bug fixes are welcome, new features go to the two-mode server.
- Before asking for review run `make test`, `make smoke` and `go test ./...` in every Go module you touched.
- Every quality or latency number needs its results file in the repo; unmeasured things are written as "not measured".
- No data under -SA or -NC licenses, no Twitter/X or Reddit data, no real personal data.
- Contributions are licensed under Apache-2.0 (section 5 of [`LICENSE`](LICENSE)).
- Docs are written in Spanish; issues and PRs in English are welcome too.
