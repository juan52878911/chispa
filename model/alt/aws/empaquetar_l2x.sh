#!/usr/bin/env bash
# Paquete de código de L2x (P3): model/alt y lo que importa + sdk/python (el módulo `chispa`, que las dos g5 anteriores no tenían).
# Uso: bash model/alt/aws/empaquetar_l2x.sh [rama-o-sha] > codigo.tgz   (por defecto HEAD). No sube nada.
# Sin las carpetas model/alt/resultados*: el tarball depende solo del código, no de los informes que se commitean después.
# La batería no vista va aparte (datos): bateria-no-vista.tgz = los *.jsonl y manifiesto.json de `python -m model.alt.bateria_no_vista`
# más baseline-top1.json (`p0_medir base-top1`), todo en la raíz del tarball.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git archive --format=tar.gz "${1:-HEAD}" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*'
