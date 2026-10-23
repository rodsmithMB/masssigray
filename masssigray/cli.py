"""Console entry point: launches the Streamlit app."""
import subprocess
import sys
from pathlib import Path


def run_app():
    app = Path(__file__).with_name("app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))
