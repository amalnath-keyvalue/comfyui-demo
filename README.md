# ComfyUI Agent

LangGraph agent with ComfyUI integration. Automatically routes between image generation and normal chat using Groq (Llama 3.3 70B).

## Setup & Run

```bash
bash run.sh
```

First run: downloads ComfyUI, SD 1.5 model (4GB), and prompts for Groq API key.

## Features

- Smart routing (image generation vs chat)
- Real-time progress updates via WebSocket
- Auto-starts/stops ComfyUI server
- Opens generated images automatically

## Examples

```
You: generate a cat on a skateboard
# → generates and opens image

You: what's the weather?
# → normal chat response
```
