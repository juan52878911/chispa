#!/usr/bin/env python3
"""Comprueba los enlaces relativos (y sus #anclas) de README.md y docs/*.md.

Uso, desde la raíz del repositorio: python3 docs/comprobar-enlaces.py
Sale con 1 si algún enlace relativo no existe o su ancla no está en el destino.
"""
import re, sys, unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENLACE = re.compile(r'\]\(([^)\s]+)\)')


def slug(titulo):
    t = re.sub(r'`|\*', '', titulo).strip().lower()
    t = re.sub(r'<[^>]+>', '', t)
    t = ''.join(c for c in t if c.isalnum() or c in ' -_' or unicodedata.category(c).startswith('L'))
    return t.replace(' ', '-')


def anclas(ruta):
    vistos, res = {}, set()
    dentro = False
    for l in ruta.read_text(encoding='utf-8').split('\n'):
        if l.startswith('```'):
            dentro = not dentro
        m = None if dentro else re.match(r'#{1,6}\s+(.*)', l)
        if m:
            s = slug(m.group(1))
            n = vistos.get(s, 0)
            vistos[s] = n + 1
            res.add(s if n == 0 else f'{s}-{n}')
    return res


def main():
    ficheros = [RAIZ / 'README.md'] + sorted((RAIZ / 'docs').glob('*.md'))
    malos = total = 0
    for f in ficheros:
        dentro = False
        for n, l in enumerate(f.read_text(encoding='utf-8').split('\n'), 1):
            if l.startswith('```'):
                dentro = not dentro
            if dentro:
                continue
            for url in ENLACE.findall(l):
                if re.match(r'[a-z]+:', url):
                    continue
                total += 1
                ruta, _, ancla = url.partition('#')
                dest = f if not ruta else (f.parent / ruta).resolve()
                if not dest.exists():
                    print(f'{f.relative_to(RAIZ)}:{n}: no existe {url}'); malos += 1
                elif ancla and dest.suffix == '.md' and ancla not in anclas(dest):
                    print(f'{f.relative_to(RAIZ)}:{n}: ancla no encontrada {url[:90]}'); malos += 1
    print(f'{total} enlaces relativos comprobados, {malos} rotos')
    sys.exit(1 if malos else 0)


main()
