"""Comprobación automática de licencias de la batería v3 (sin red: solo lee texto que ya se bajó).

Admitidas: CC0, CC BY, MIT, Apache-2.0, ODC-BY, CDLA-Permissive y dominio público. Fuera, con su motivo: -SA, -NC, -ND, licencia ausente,
ambigua o contradictoria entre cabecera y cuerpo, y fuentes de Twitter/X o Reddit. Lo que no se puede comprobar queda fuera: nunca se «adivina».
"""
import hashlib
import re

ADMITIDAS = {
    "cc0-1.0": "CC0", "cc-by-4.0": "CC BY", "cc-by-3.0": "CC BY", "cc-by-2.5": "CC BY", "cc-by-2.0": "CC BY",
    "mit": "MIT", "apache-2.0": "Apache-2.0", "odc-by": "ODC-BY", "odc-by-1.0": "ODC-BY",
    "cdla-permissive-1.0": "CDLA-Permissive", "cdla-permissive-2.0": "CDLA-Permissive", "dominio-publico": "dominio público",
}
_ALIAS = {"cc0": "cc0-1.0", "cc-0": "cc0-1.0", "cc0-1": "cc0-1.0", "cc-by": "cc-by-4.0", "apache": "apache-2.0", "apache-2": "apache-2.0", "apache2": "apache-2.0",
          "odc-by-1": "odc-by-1.0", "odc-by-1.0": "odc-by-1.0", "public-domain": "dominio-publico", "pd": "dominio-publico", "publicdomain": "dominio-publico"}
# señales de que la licencia es más restrictiva que la cabecera (cuerpo de la tarjeta o ficha)
_ALERTAS_LICENCIA = [
    (r"cc[ -]?by[ -]?nc", "NC"), (r"cc[ -]?by[ -]?sa", "SA"), (r"cc[ -]?by[ -]?nd", "ND"), (r"non[ -]?commercial|no[ -]?comercial|sin fines comerciales", "NC"),
    (r"share[ -]?alike|compartir[ -]?igual", "SA"), (r"research (purposes )?only|solo (para )?investigaci[oó]n|academic (use )?only", "solo investigación"),
    (r"\bgpl\b|\bagpl\b|\blgpl\b|\bodbl\b", "copyleft"),
]
_ORIGENES = [(r"\btwitter\b|\btweets?\b|\bx\.com\b", "Twitter/X"), (r"\breddit\b|\bsubreddit\b", "Reddit")]


def sha256_texto(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def normalizar(lic):
    s = re.sub(r"[\s_]+", "-", str(lic or "").strip().lower())
    return _ALIAS.get(s, s)


def cabecera_yaml(texto):
    """Campo `license` de la cabecera YAML de una tarjeta de HF: lista de ids normalizados (vacía si no hay)."""
    m = re.match(r"\ufeff?---\s*\n(.*?)\n---\s*(\n|$)", texto or "", re.S)
    if not m:
        return []
    import yaml
    try:
        d = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return []
    v = d.get("license") if isinstance(d, dict) else None
    if v is None:
        return []
    return [normalizar(x) for x in (v if isinstance(v, list) else [v])]


def _cuerpo(texto):
    return re.sub(r"\A\ufeff?---\s*\n.*?\n---\s*(\n|$)", "", texto or "", count=1, flags=re.S)


def _alertas_cuerpo(texto, declarada_ok):
    out = []
    cuerpo = _cuerpo(texto).lower()
    for pat, nombre in _ALERTAS_LICENCIA:
        for m in re.finditer(pat, cuerpo):
            ctx = cuerpo[max(0, m.start() - 80): m.end() + 80]
            # una mención suelta («el original era NC, esto no») también cuenta: es conservador a propósito
            out.append("%s en el texto («…%s…»)" % (nombre, re.sub(r"\s+", " ", ctx).strip()))
            break
    return out


def _origen_prohibido(texto):
    low = (texto or "").lower()
    return [n for pat, n in _ORIGENES if re.search(pat, low)]


def evaluar_tarjeta(texto, esperada=None, origen_declarado=""):
    """Evalúa la tarjeta (README.md) de un dataset de HF. -> dict(estado, licencia, motivo, avisos)."""
    decl = cabecera_yaml(texto)
    r = dict(estado="fuera", licencia=",".join(decl), motivo="", avisos=[], sha256_texto=sha256_texto(texto or ""))
    if not (texto or "").strip():
        r["motivo"] = "sin tarjeta: licencia no verificable"
        return r
    if not decl:
        r["motivo"] = "la tarjeta no declara licencia (campo license ausente)"
        return r
    if len(decl) > 1:
        r["motivo"] = "licencia múltiple o ambigua en la cabecera: %s" % ", ".join(decl)
        return r
    lic = decl[0]
    if lic not in ADMITIDAS:
        r["motivo"] = "licencia no admitida o ambigua: «%s»" % lic
        return r
    al = _alertas_cuerpo(texto, lic)
    if al:
        r["motivo"] = "cabecera %s pero el cuerpo de la tarjeta contradice: %s" % (lic, "; ".join(al))
        return r
    orig = _origen_prohibido(texto + " " + (origen_declarado or ""))
    if orig:
        r["motivo"] = "la tarjeta o el origen declarado menciona %s: términos de la plataforma" % "/".join(orig)
        return r
    if esperada and normalizar(esperada) != lic:
        r["avisos"].append("licencia esperada %s, la tarjeta dice %s (las dos admitidas)" % (esperada, lic) if normalizar(esperada) in ADMITIDAS else
                           "licencia esperada %s, la tarjeta dice %s" % (esperada, lic))
    r.update(estado="admitida", licencia=lic)
    return r


def evaluar_ficha(texto, patron_exigido, licencia_si_pasa="dominio-publico", origen_declarado=""):
    """Evalúa un texto de licencia o ficha que NO es tarjeta de HF (LICENSE, página de términos): admite solo si el texto trae lo exigido
    (`patron_exigido`, regex, sin distinguir mayúsculas) y ninguna alerta de -SA/-NC/-ND ni origen Twitter/X o Reddit."""
    r = dict(estado="fuera", licencia="", motivo="", avisos=[], sha256_texto=sha256_texto(texto or ""))
    if not (texto or "").strip():
        r["motivo"] = "ficha o licencia vacía o no descargada: licencia no verificable"
        return r
    if not patron_exigido or not re.search(patron_exigido, texto, re.I):
        r["motivo"] = "la ficha no contiene el texto exigido para esta licencia (/%s/)" % patron_exigido
        return r
    al = _alertas_cuerpo(texto, licencia_si_pasa)
    if al:
        r["motivo"] = "la ficha contradice la licencia: %s" % "; ".join(al)
        return r
    orig = _origen_prohibido(origen_declarado)
    if orig:
        r["motivo"] = "origen declarado %s: términos de la plataforma" % "/".join(orig)
        return r
    r.update(estado="admitida", licencia=licencia_si_pasa)
    return r
