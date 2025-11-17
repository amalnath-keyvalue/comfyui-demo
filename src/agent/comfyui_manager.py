import atexit
import os
import signal
import subprocess
import time

import requests


class ComfyUIManager:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.base_url = f"http://{server_address}"
        self.process = None
        self.comfyui_dir = os.path.join(os.path.dirname(__file__), "../..", "ComfyUI")

    def is_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def start_server(self):
        if self.is_running():
            print("ComfyUI server is already running")
            return True

        if not os.path.exists(self.comfyui_dir):
            print("ComfyUI not found. Run: bash run.sh")
            return False

        print("Starting ComfyUI server...")
        venv_python = os.path.join(self.comfyui_dir, "venv/bin/python")
        main_py = os.path.join(self.comfyui_dir, "main.py")

        self.process = subprocess.Popen(
            [venv_python, main_py, "--listen", "127.0.0.1", "--port", "8188"],
            cwd=self.comfyui_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        atexit.register(self.stop_server)

        for _ in range(30):
            time.sleep(1)
            if self.is_running():
                print("ComfyUI server started successfully!")
                return True

        print("Failed to start ComfyUI server")
        return False

    def stop_server(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
                print("ComfyUI server stopped")
            except Exception:
                pass
