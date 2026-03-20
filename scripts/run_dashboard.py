"""
run_dashboard.py — Student 5: Monitoring Lead

Launches the Streamlit monitoring dashboard.

Usage (from project root):
    python scripts/run_dashboard.py
"""

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
