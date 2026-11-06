import sys
from pathlib import Path

c.ServerProxy.servers = {
    "gesture-lab": {
        "command": [sys.executable, "-m", "streamlit", "run", str(Path.home() / "app.py"),
                    "--server.address=127.0.0.1", "--server.port={port}",
                    "--server.headless=true", "--browser.gatherUsageStats=false"],
        "timeout": 120,
        "launcher_entry": {"title": "Gesture Lab"},
    }
}
