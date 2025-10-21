# app.py (à la racine du repo)
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))  # rend 'models/' importable partout

# On importe ton vrai script Streamlit :
import importlib.util

script_path = ROOT / "views" / "main-national-view.py"
spec = importlib.util.spec_from_file_location("main_national_view", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
