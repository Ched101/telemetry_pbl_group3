
import subprocess
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DASHBOARD = PROJECT_ROOT / "src" / "monitoring_lead" / "dashboard.py"

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
     "--server.headless", "false"],
    check=True,
)
