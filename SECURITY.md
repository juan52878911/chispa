# Seguridad

## Cómo informar de una vulnerabilidad

**No abras un issue público.** Usa el aviso privado de GitHub: pestaña *Security* → *Report a vulnerability*
(«Private vulnerability reporting»). Incluye la versión o el commit, cómo reproducirlo y el impacto que ves.

Respuesta: intentamos acusar recibo en 7 días y acordar contigo la fecha de publicación del arreglo. Es un proyecto mantenido por
una persona, sin programa de recompensas.

## Qué entra

- `chispa-server` (`server/`): validación de peticiones, límites, autenticación por `api_keys`, carga de `.chispa` y de bancos.
- El formato `.chispa` y su cargador (`format/chispa3/`): un fichero malicioso no debería poder hacer más que fallar al cargar.
- Los SDK (`sdk/`).

Fuera: las salidas del clasificador en sí (una respuesta equivocada no es una vulnerabilidad; los ataques adversariales medidos
están en [`arena/`](arena/README.md)), y los despliegues de terceros.

## Versiones

Solo se mantiene `main`. No hay versiones publicadas todavía.

---

## Security (English)

**Do not open a public issue.** Use GitHub's private vulnerability reporting (*Security* tab → *Report a vulnerability*), with the
version or commit, steps to reproduce and impact. We aim to acknowledge within 7 days. Single maintainer, no bug bounty.
Only `main` is supported.
