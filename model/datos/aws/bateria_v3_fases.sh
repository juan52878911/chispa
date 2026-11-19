#!/bin/bash
# Batería v3, FASE 2: fases (las carga con `.` el user-data bateria_v3.sh tras bajar y comprobar el sha256 de codigo.tgz, dev.tgz y selladas.tgz de v2 y el manifiesto; comparte sus variables y funciones:
# BASE, OUT, LOCAL, MIN, ESPERA_MIN, HUB, PYARROW, PYYAML, MANIF, N_TAREAS, estado, fallo, cierre, quedan_min).
# Modo local (CHISPA_LOCAL=1, prueba en seco sin red): CHISPA_PY, CHISPA_V3_ESPEJO (carpeta con hf/<owner>__<repo>/, ver model/datos/tests/fixtures_v3), CHISPA_V3_MANIFIESTO, CHISPA_V3_RESERVAS=1, CHISPA_V3_ENTRENAMIENTO (manifiesto de entrenamiento de pruebas), CHISPA_V3_N_DEV y CHISPA_V3_N_SELLADAS (nº de jsonl esperados en los .tgz de v2; 6 y 8).
# Nada de lo que imprime esto lleva texto de filas: solo recuentos, estados y motivos.
fases_main() {
CH=$BASE/codigo; DATOS=$OUT/datos; TRAB=$BASE/trabajo; export PYTHONPATH=$CH
mkdir -p "$DATOS" "$TRAB"

# --- 2. entorno: se comprueba que importa todo y que las fuentes responden ANTES de bajar un solo dato
echo "== 2. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  LIBRES=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "${LIBRES:-0}" -ge 25 ] || { fallo "volumen pequeno: ${LIBRES:-?} GB libres (hacen falta >= 25)"; estado entorno "FALLO disco"; cierre; }
  dnf install -y -q python3.11 python3.11-pip tar gzip unzip || { fallo "dnf install"; estado entorno "FALLO dnf"; cierre; }
  python3.11 -m venv "$BASE/venv" && "$BASE/venv/bin/pip" install -q --no-cache-dir "huggingface_hub==$HUB" "pyarrow==$PYARROW" "PyYAML==$PYYAML" || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  PY=$BASE/venv/bin/python
fi
"$PY" -c "
import pyarrow, huggingface_hub, yaml, zipfile, csv
if '$LOCAL' != '1':
    assert pyarrow.__version__ == '$PYARROW' and huggingface_hub.__version__ == '$HUB' and yaml.__version__ == '$PYYAML', (pyarrow.__version__, huggingface_hub.__version__, yaml.__version__)
" || { fallo "imports o versiones"; estado entorno "FALLO imports"; cierre; }
"$PY" -c "import model.datos.bateria_v3, model.datos.bateria_v3_licencias, model.datos.bateria_v3_solapes, model.datos.bateria_v3_candidatas, model.datos.bateria_v3_huellas" || { fallo "imports del codigo"; estado entorno "FALLO imports codigo"; cierre; }
CN=$("$PY" -c "import json;print(len(json.load(open('$MANIF'))['tareas']))")
[ "$CN" = "$N_TAREAS" ] || { fallo "el manifiesto trae $CN tareas y se esperaban $N_TAREAS"; estado entorno "FALLO tareas"; cierre; }
if [ "$LOCAL" != 1 ]; then
  for u in https://huggingface.co/api/datasets/openai/gsm8k https://www.ncei.noaa.gov/ https://static.nhtsa.gov/ https://github.com/github/advisory-database https://clinicaltrials.gov/ https://vaers.hhs.gov/ https://www.osha.gov/ https://www.cpsc.gov/; do
    curl -fsS -m 25 -o /dev/null "$u" && echo "red OK $u" || echo "AVISO: sin respuesta de $u (las tareas de esa fuente quedaran fuera por descarga)"
  done
fi
estado entorno "OK $("$PY" -c "import sys, pyarrow, huggingface_hub; print('python', sys.version.split()[0], 'pyarrow', pyarrow.__version__, 'huggingface_hub', huggingface_hub.__version__)"); $(nproc) vCPU; $(free -g | awk '/Mem:/{print $2}') GB RAM; $N_TAREAS tareas; \
v2: dev.tgz y selladas.tgz con sha256 comprobado"

# --- 2b. huellas DENTRO de la instancia: v2 (dev y selladas) y fuentes públicas del entrenamiento. Salen hashes y sketches; los textos se quedan aquí (el log se audita contra ellos al final)
echo "== 2b. huellas $(date -u +%T) quedan $(quedan_min) min"
HUE=$BASE/huellas; MENT=${CHISPA_V3_ENTRENAMIENTO:-$CH/model/alt/manifiesto-entrenamiento.json}
( cd "$CH" && "$PY" -m model.datos.bateria_v3_huellas v2 --dev "$BASE/in/dev.tgz" --selladas "$BASE/in/selladas.tgz" --trabajo "$BASE/v2txt" --salida "$HUE" --n-dev "${CHISPA_V3_N_DEV:-6}" --n-selladas "${CHISPA_V3_N_SELLADAS:-8}" ) > "$OUT/huellas-v2.out" 2>&1
RC=$?; cat "$OUT/huellas-v2.out"
[ "$RC" = 0 ] && [ -f "$HUE/huellas-v2.jsonl.gz" ] || { fallo "huellas de v2 rc=$RC"; estado huellas "FALLO v2"; cierre; }
( cd "$CH" && timeout $(( ($(quedan_min) - ESPERA_MIN - 30) * 60 )) "$PY" -m model.datos.bateria_v3_huellas entrenamiento --manifiesto "$MENT" --trabajo "$BASE/entr" --salida "$HUE" --muestra "$BASE/muestra" --tope-min "${CHISPA_V3_TOPE_HUELLAS_MIN:-35}" ) > "$OUT/huellas-entr.out" 2>&1
RC=$?; cat "$OUT/huellas-entr.out"
[ "$RC" = 0 ] && [ -f "$HUE/huellas-entrenamiento.jsonl.gz" ] && [ -f "$HUE/estado-huellas.json" ] || { fallo "huellas del entrenamiento rc=$RC (124 = tope de tiempo)"; estado huellas "FALLO entrenamiento"; cierre; }
estado huellas "OK $("$PY" -I -c "
import json, sys
e = json.load(open(sys.argv[1]))
print('v2=%d entrenamiento=%d comprobadas=%d solape_no_comprobado=%d' % (e['v2']['huellas'], e['entrenamiento']['huellas'], len(e['entrenamiento']['comprobados']), len(e['no_comprobado'])))" "$HUE/estado-huellas.json")"
rm -rf "$BASE/entr" "$BASE/in/dev.tgz" "$BASE/in/selladas.tgz"       # los .tgz de v2 ya son huellas; los textos extraídos quedan en v2txt solo hasta la auditoría del log

# --- 3. reservas de v2: tarjetas (en la revision de v2) y fichas de origen de promise, ropa, rtp y CFPB, con sha256 y la comprobacion automatica
echo "== 3. reservas v2 $(date -u +%T) quedan $(quedan_min) min"
if [ "$LOCAL" != 1 ] || [ "${CHISPA_V3_RESERVAS:-0}" = 1 ]; then
  ( cd "$CH" && timeout 600 "$PY" -m model.datos.bateria_v3 reservas-v2 --manifiesto "$MANIF" --trabajo "$TRAB" --salida "$DATOS" ) > "$OUT/reservas.out" 2>&1; cat "$OUT/reservas.out"
  [ -f "$DATOS/reservas-v2.json" ] && estado reservas-v2 "OK $(wc -c < "$DATOS/reservas-v2.json") bytes de resultado" || { fallo "reservas-v2 sin resultado"; estado reservas-v2 "FALLO"; }
else estado reservas-v2 "OMITIDO (local)"; fi

# --- 4. construir: ficha -> licencia -> descarga con revision y sha256 -> esquema -> solapes -> particion -> sellado. Solo imprime recuentos.
echo "== 4. construir $(date -u +%T) quedan $(quedan_min) min"
RESTA=$(( ($(quedan_min) - ESPERA_MIN - 6) * 60 )); [ "$RESTA" -gt 120 ] || { fallo "sin tiempo para construir"; estado construir "FALLO tiempo"; cierre; }
( cd "$CH" && timeout "$RESTA" "$PY" -m model.datos.bateria_v3 construir --manifiesto "$MANIF" --trabajo "$TRAB" --salida "$DATOS" --huellas "$BASE/huellas" ) > "$OUT/construir.out" 2>&1
RC=$?; cat "$OUT/construir.out"
[ "$RC" = 0 ] && [ -f "$DATOS/manifiesto.json" ] || { fallo "construir rc=$RC (124 = tope de tiempo)"; estado construir "FALLO rc=$RC"; cierre; }
rm -rf "$TRAB"      # los datos de origen no se suben ni se guardan: solo el manifiesto con sus sha256

# --- 5. comprobaciones: cada sellada con permisos 0444, su sha256 y su recuento igual a los del manifiesto; ninguna licencia sin guardar
"$PY" -I - "$DATOS" <<'PYEOF' > "$OUT/comprobacion.txt" 2>&1
import hashlib, json, os, stat, sys
d = sys.argv[1]
m = json.load(open(d + "/manifiesto.json"))
mal = []
for t in m["tareas"]:
    if t["estado"] == "construida":
        p = d + "/" + t["fichero_test"]
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        n = sum(1 for _ in open(p, "rb"))
        if sha != t["sha256"] or n != t["n_test"] or stat.S_IMODE(os.stat(p).st_mode) != 0o444:
            mal.append(t["nombre"])
        if not os.path.exists(d + "/licencias/%s.txt" % t["nombre"]) or len(t["licencia_sha256"]) != 64:
            mal.append(t["nombre"] + ":licencia")
    elif t["estado"] == "fuera" and not t["motivo_fuera"]:
        mal.append(t["nombre"] + ":sin motivo")
r = m["resumen"]
print("construidas=%d fuera=%d pendientes=%d" % (r["por_estado"].get("construida", 0), r["por_estado"].get("fuera", 0), r["por_estado"].get("pendiente", 0)))
for tp, v in r["por_tipo"].items():
    print("tipo %s: candidatas=%d construidas=%d cuentan=%d (es %d, en %d) informativo=%d%s" % (tp, v["candidatas"], v["construidas"], v["cuentan"], v["es"], v["en"], v["informativo"], " N_BAJO" if v["n_bajo"] else ""))
print("espanol: cuentan=%d informativo=%d%s" % (r["por_idioma"]["es"]["cuentan"], r["por_idioma"]["es"]["informativo"], " N_BAJO" if r["por_idioma"]["es"]["n_bajo"] else ""))
print("MAL: " + ",".join(mal) if mal else "OK: sellada, sha256 y licencias coinciden con el manifiesto")
sys.exit(1 if mal else 0)
PYEOF
RC=$?; cat "$OUT/comprobacion.txt"
[ "$RC" = 0 ] && estado comprobacion "OK $(head -1 "$OUT/comprobacion.txt")" || { fallo "comprobacion de selladas"; estado comprobacion "FALLO"; }
cierre
}
