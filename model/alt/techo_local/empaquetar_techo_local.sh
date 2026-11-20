#!/usr/bin/env bash
# Techo local: empaqueta lo que lee techo_local.sh y escribe su user-data. NO sube ni lanza nada: se suben a mano, por nombre exacto, a
# s3://chispa-pruebas-cuenta-aws/techo-local-SHA/ los dos tgz de DIR_SALIDA.
#   codigo.tgz   de `git archive SHA`: el mismo conjunto que T3/D (model/alt con techo_local*, datos, meta, evaluacion, l2 mínimo, sdk/python), sin model/alt/resultados* ni lanzados/; se comprueba
#                que importa solo con eso (techo_local, cliente, lineas_base_v2_pred...).
#   dev.tgz      SOLO los 6 jsonl de DESARROLLO de DIR_DEV (bateria-v2/dev); si trae algo más, falla: las selladas no se copian.
#   model/alt/techo_local/lanzados/ud-techo-local-SHA.sh   techo_local.sh de SHA con __CHISPA_S3__, el sha256 del criterio, las revisiones de los dos candidatos (las lee del criterio), los sha256
#                de los dos tgz y los recuentos de dev. Debe caber en los 16 384 bytes de user-data de EC2; las fases viven en codigo.tgz.
# Entorno:  CRIT_SHA256_ESPERADO=<sha256 con el que la sesión principal congeló crit-techo-local.json> (obligatorio: si el criterio de SHA no lo cumple, falla)
#           CHISPA_ENSAYO=1   prueba en seco: acepta un criterio sin congelar y con revisiones PENDIENTE (usa revisiones de ceros); CHISPA_UD_DIR cambia dónde se escribe el ud.
# Uso: CRIT_SHA256_ESPERADO=... bash model/alt/techo_local/empaquetar_techo_local.sh SHA DIR_DEV DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; dev=${2:?DIR_DEV}; sal=${3:?DIR_SALIDA}
ensayo=${CHISPA_ENSAYO:-0}
shf() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
dev=$(cd "$dev" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
for f in model/alt/techo_local/techo_local.sh model/alt/techo_local/techo_local_fases.sh model/alt/crit-techo-local.json model/alt/techo_local.py model/alt/techo_local_cliente.py model/alt/rubricas-r10-t3b.json; do
  git cat-file -e "$sha:$f" || { echo "$sha no trae $f" >&2; exit 1; }
done
esperadas="cfpb-quejas congreso-eeuu-partido finanzas-documentos horeca-resenas jql-valor-educativo usajobs-nivel"
hay=$(cd "$dev" && ls | grep -v '^\._' | tr '\n' ' ')
want=$(for t in $esperadas; do printf '%s.jsonl ' "$t"; done)
[ "$hay" = "$want" ] || { echo "DIR_DEV debe traer exactamente los 6 jsonl de dev; trae: $hay" >&2; exit 1; }
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git show "$sha:model/alt/crit-techo-local.json" > "$t/crit.json"
crit=$(shf "$t/crit.json")
if [ "$ensayo" != 1 ]; then
  [ -n "${CRIT_SHA256_ESPERADO:-}" ] || { echo "falta CRIT_SHA256_ESPERADO (el sha256 con el que se congeló el criterio)" >&2; exit 1; }
  [ "$crit" = "$CRIT_SHA256_ESPERADO" ] || { echo "el criterio de $sha ($crit) no es el congelado ($CRIT_SHA256_ESPERADO)" >&2; exit 1; }
fi
revs=$(python3 -I - "$t/crit.json" "$ensayo" <<'PY'
import json, re, sys
m = json.load(open(sys.argv[1]))["modelos"]
r = []
for k in ("qwen3-30b-a3b", "gpt-oss-20b"):
    v = m[k]["rev"]
    if not re.fullmatch(r"[0-9a-f]{40}", v):
        if sys.argv[2] != "1":
            sys.exit("la revisión de %s no está fijada (%r): hace falta un commit de 40 hex de huggingface.co" % (k, v))
        v = "0" * 40
    r.append(v)
print(*r)
PY
)
rq=${revs% *}; ro=${revs#* }
precio=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['instancia']['precio_usd_hora'])" "$t/crit.json")
MODELALT="model/alt/techo_local model/alt/techo_local.py model/alt/techo_local_cliente.py model/alt/crit-techo-local.json"
export COPYFILE_DISABLE=1
git archive --format=tar "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' ':(exclude)model/alt/techo_local/lanzados' ':(exclude)model/alt/aws/lanzados' ':(exclude)model/datos/aws/lanzados' | tar -x -C "$t"
rm -f "$t/crit.json"; find "$t" -name '._*' -delete
for f in $MODELALT; do [ -e "$t/$f" ] || { echo "el tgz no trae $f" >&2; exit 1; }; done
( cd "$t" && python3 -I -c "
import sys; sys.path.insert(0, '.')
import model.alt.techo_local, model.alt.techo_local_cliente, model.alt.lineas_base_v2_pred, model.alt.lineas_base_v2b, model.alt.auditoria, model.alt.llm_logprobs, model.alt.apertura_v2" ) \
  || { echo "el código empaquetado no importa solo con lo que lleva el tgz" >&2; exit 1; }
tarz() { tar --no-xattrs -czf "$1" -C "$2" "${@:3}" 2>/dev/null || tar -czf "$1" -C "$2" "${@:3}"; }
rm -f "$sal/codigo.tgz" "$sal/dev.tgz"

tarz "$sal/codigo.tgz" "$t" model sdk
( cd "$dev" && { tar --no-xattrs -czf "$sal/dev.tgz" $want 2>/dev/null || tar -czf "$sal/dev.tgz" $want; } )
for x in codigo dev; do tar -tzf "$sal/$x.tgz" > "$t/$x.lst"; [ "$(grep -c '\._' "$t/$x.lst")" = 0 ] || { echo "$x.tgz trae ._*" >&2; exit 1; }; done
[ "$(grep -ci 'sellad' "$t/dev.lst")" = 0 ] || { echo "dev.tgz trae algo sellado" >&2; exit 1; }
[ "$(wc -l < "$t/dev.lst" | tr -d ' ')" = 6 ] || { echo "dev.tgz no trae 6 ficheros" >&2; exit 1; }
[ "$(grep -c 'model/alt/techo_local/techo_local_fases.sh' "$t/codigo.lst")" = 1 ] || { echo "codigo.tgz sin las fases" >&2; exit 1; }
cuenta=$(python3 -I - "$dev" <<'PY'
import glob, json, sys
t = s = 0
for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            sp = json.loads(l)["split"]
            t += sp == "test"
            s += sp == "train"
print(t, s)
PY
)
nt=${cuenta% *}; ns=${cuenta#* }
udd=${CHISPA_UD_DIR:-model/alt/techo_local/lanzados}; mkdir -p "$udd"
ud=$udd/ud-techo-local-$sha.sh
git show "$sha:model/alt/techo_local/techo_local.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/techo-local-$sha#" -e "s#__CRIT_SHA__#$crit#" -e "s#__COD_SHA256__#$(shf "$sal/codigo.tgz")#" \
  -e "s#__DEV_SHA256__#$(shf "$sal/dev.tgz")#" -e "s#__N_TEST__#$nt#" -e "s#__N_SOP__#$ns#" -e "s#__REV_Q30__#$rq#" -e "s#__REV_OSS__#$ro#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
[ "$(wc -c < "$ud")" -lt 15500 ] || { echo "el ud pesa $(wc -c < "$ud") bytes: el user-data de EC2 admite 16384 como máximo (objetivo < 15500)" >&2; exit 1; }
bash -n <(git show "$sha:model/alt/techo_local/techo_local_fases.sh")
min=$(sed -n 's/^MIN=\${CHISPA_MIN:-\([0-9]*\)}.*/\1/p' "$ud")
(cd "$sal" && ls -l codigo.tgz dev.tgz && echo "codigo.tgz $(shf codigo.tgz)" && echo "dev.tgz $(shf dev.tgz)")
echo "ud: $ud ($(wc -c < "$ud") bytes, sha256 $(shf "$ud")); criterio sha256 $crit; revisiones $rq $ro; dev: test=$nt soporte=$ns"
[ "$ensayo" = 1 ] && echo "ENSAYO: criterio sin congelar y revisiones de ceros; este ud NO se lanza"
echo "coste: shutdown -h +$min con g6e.xlarge a $precio USD/h = $(python3 -I -c "print(round($min * $precio / 60, 2))") USD de cómputo (+ ~0,04 de EBS); tope de la corrida 4 USD"
echo "sube a s3://chispa-pruebas-cuenta-aws/techo-local-$sha/: codigo.tgz y dev.tgz; lanza $ud en una g6e.xlarge con volumen gp3 >= 160 GB (throughput 500 MB/s: los pesos de 31 GB se leen dos veces)"
