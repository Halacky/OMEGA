"""Launch the Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", "8501"],
    )


if __name__ == "__main__":
    main()
