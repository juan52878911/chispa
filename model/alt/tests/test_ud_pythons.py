"""Los user-data de la g5 llaman a cada paquete con el python que lo tiene (fallo de T3b en la g5: huggingface_hub y numpy con el python equivocado) y lo comprueban al principio."""
import re
import subprocess
from pathlib import Path

AWS = Path(__file__).resolve().parents[1] / "aws"


def test_pythons_de_los_ud_de_g5():
    for nombre in ("r10_t3b.sh", "r10_d.sh"):
        ud = (AWS / nombre).read_text(encoding="utf-8")
        assert subprocess.run(["bash", "-n", str(AWS / nombre)]).returncode == 0
        # la descarga del modelo y de e5 va con el venv de vLLM (trae huggingface_hub), no con el venv de transformers, que aún se está instalando
        assert 'HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub' in ud and '"$NPY" -c "from huggingface_hub' not in ud
        # medir y todo lo que solo necesita numpy corre con el venv de vLLM
        assert '"${VPY:-python3}" -m model.alt.lineas_base_v2' in ud and "NPY:-python3" not in ud
        despues = ud.split("# --- 2. entradas", 1)[1]
        assert '"$PY"' not in despues   # el python de la DLAMI solo sirve para la GPU y para crear los venv
        # comprobación al principio: cada python importa lo que necesita
        assert '"$VPY" -c "import vllm, numpy, huggingface_hub"' in ud
        assert "import torch, transformers, numpy, huggingface_hub, sentencepiece" in ud and "nlp.fail" in ud
        assert ud.index('"$VPY" -c "import vllm, numpy, huggingface_hub"') < ud.index("# --- 2. entradas") < ud.index("bajar Qwen/Qwen3-4B-Instruct-2507")
        # el venv de transformers arranca antes de instalar vLLM (se instalan a la vez)
        assert ud.index('"$p" -m venv /opt/nlp') < ud.index('"$PY" -m venv /opt/vllm')
        assert len(re.findall(r"__[A-Z0-9_]+__", ud)) >= 4


def test_pythons_del_ud_de_c7i():
    ud = (AWS / "r10_t3a.sh").read_text(encoding="utf-8")
    assert "import torch, transformers, numpy, huggingface_hub, sentencepiece" in ud
    assert ud.index("pip install") < ud.index("# --- 2. entradas")
