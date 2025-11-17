import asyncio
import json
import os
from typing import Any, Callable, Dict

import requests
import websockets


class ComfyUIClient:
    def __init__(
        self,
        server_address: str = None,
    ):
        self.server_address = server_address or os.getenv(
            "COMFYUI_SERVER", "127.0.0.1:8188"
        )
        self.base_url = f"http://{self.server_address}"
        self.ws_url = f"ws://{self.server_address}/ws"
        self.client_id = "langgraph-agent"

    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        data = {"prompt": prompt, "client_id": self.client_id}
        response = requests.post(f"{self.base_url}/prompt", json=data, timeout=30)
        return response.json()["prompt_id"]

    def get_image(
        self,
        filename: str,
        subfolder: str,
        folder_type: str,
    ) -> bytes:
        url = f"{self.base_url}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = requests.get(url, params=params, timeout=30)
        return response.content

    def get_history(
        self,
        prompt_id: str,
    ) -> dict:
        response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
        return response.json()

    async def wait_for_completion(
        self, prompt_id: str, progress_callback: Callable[[str], None] = None
    ) -> dict:
        ws_url_with_client = f"{self.ws_url}?clientId={self.client_id}"
        async with websockets.connect(ws_url_with_client) as websocket:
            while True:
                out = await websocket.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    msg_type = message.get("type")

                    if msg_type == "status":
                        if progress_callback:
                            status_data = message.get("data", {})
                            exec_info = status_data.get("status", {}).get(
                                "exec_info", {}
                            )
                            queue_remaining = exec_info.get("queue_remaining", 0)
                            if queue_remaining > 0:
                                progress_callback(f"Queue position: {queue_remaining}")

                    elif msg_type == "progress":
                        if progress_callback:
                            value = message["data"]["value"]
                            max_val = message["data"]["max"]
                            progress_callback(f"Progress: {value}/{max_val}")

                    elif msg_type == "executing":
                        data = message["data"]
                        if data.get("prompt_id") == prompt_id:
                            node = data.get("node")
                            if node is None:
                                if progress_callback:
                                    progress_callback("Complete!")
                                break
                            elif progress_callback:
                                progress_callback(f"Executing: {node}")

        history = self.get_history(prompt_id)
        return history[prompt_id]

    def generate_simple_text_image(
        self,
        prompt_text: str,
        progress_callback: Callable[[str], None] = None,
    ) -> bytes:
        workflow = {
            "3": {
                "inputs": {
                    "seed": 1068304733757895,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"},
            },
            "4": {
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"},
            },
            "5": {
                "inputs": {"width": 512, "height": 512, "batch_size": 1},
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"},
            },
            "6": {
                "inputs": {
                    "text": prompt_text,
                    "clip": ["4", 1],
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
            },
            "7": {
                "inputs": {"text": "text, watermark", "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
            },
            "8": {
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"},
            },
            "9": {
                "inputs": {"filename_prefix": "SD1.5", "images": ["8", 0]},
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"},
            },
        }

        prompt_id = self.queue_prompt(workflow)
        history = asyncio.run(self.wait_for_completion(prompt_id, progress_callback))

        output = history["outputs"]["9"]
        image_info = output["images"][0]
        image_data = self.get_image(
            image_info["filename"], image_info["subfolder"], image_info["type"]
        )

        return image_data
