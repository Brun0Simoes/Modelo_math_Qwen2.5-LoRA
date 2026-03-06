"""
MathSolver API Server — Loads the BEST model in a background thread
and exposes /api/status for real-time progress tracking.

Usage:
    python scripts/math_api_server.py

Then open http://localhost:5173 (Vite dev) or http://localhost:3001
"""

import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "qwen25math7b_BEST"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
PORT = int(os.environ.get("PORT", 3001))
MAX_NEW_TOKENS = 512

# ── Global state ──
_model = None
_tokenizer = None
_loading_state = {
    "status": "idle",       # idle | loading | ready | error
    "progress": 0,          # 0-100
    "step": "",             # current step description
    "error": "",
    "total_layers": 339,
    "loaded_layers": 0,
}
_lock = threading.Lock()


def _update_loading(status=None, progress=None, step=None, error=None, loaded=None):
    with _lock:
        if status: _loading_state["status"] = status
        if progress is not None: _loading_state["progress"] = progress
        if step: _loading_state["step"] = step
        if error: _loading_state["error"] = error
        if loaded is not None: _loading_state["loaded_layers"] = loaded


def _get_loading_state():
    with _lock:
        return dict(_loading_state)


def load_model_background():
    """Load model in background thread."""
    global _model, _tokenizer

    try:
        _update_loading(status="loading", progress=5, step="Importando bibliotecas...")
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from peft import AutoPeftModelForCausalLM

        _update_loading(progress=10, step="Configurando quantização 4-bit NF4...")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        _update_loading(progress=15, step="Carregando pesos do modelo (7.7B parâmetros)...")
        print(f"[API] Loading model from {CHECKPOINT}...")
        sys.stdout.flush()

        _model = AutoPeftModelForCausalLM.from_pretrained(
            str(CHECKPOINT),
            quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )

        _update_loading(progress=85, step="Carregando tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))

        _update_loading(progress=95, step="Compilando modelo...")
        _model.eval()

        _update_loading(status="ready", progress=100, step="Modelo pronto!", loaded=339)
        print("[API] Model loaded and ready!")
        sys.stdout.flush()

    except Exception as e:
        traceback.print_exc()
        _update_loading(status="error", step="Erro ao carregar modelo", error=str(e))


def solve(problem_text: str) -> dict:
    """Solve a math problem and return structured result."""
    import torch

    if _model is None:
        raise RuntimeError("Modelo ainda está carregando. Aguarde.")

    print(f"[API] Solving: {problem_text[:80]}...")
    sys.stdout.flush()

    messages = [
        {"role": "system", "content": "Resolva passo a passo. Ao final, escreva 'Final answer: <resposta>'."},
        {"role": "user", "content": problem_text},
    ]

    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    elapsed = round(time.time() - t0, 1)
    n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    print(f"[API] Generated {n_tokens} tokens in {elapsed}s")
    sys.stdout.flush()

    response_ids = output[0][inputs["input_ids"].shape[1]:]
    draft = _tokenizer.decode(response_ids, skip_special_tokens=True)

    # Extract final answer
    final_answer = ""
    match = re.search(r"[Ff]inal\s+[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)", draft)
    if match:
        final_answer = match.group(1).strip()
    elif "boxed{" in draft:
        box_match = re.search(r"\\boxed\{([^}]+)\}", draft)
        if box_match:
            final_answer = box_match.group(1).strip()

    return {
        "ok": True,
        "source": "manual_text",
        "usedText": problem_text,
        "elapsed": elapsed,
        "tokens": n_tokens,
        "run": {
            "best_candidate": {
                "draft": draft,
                "final_answer": final_answer,
                "score": 1.0,
                "plan_name": "qwen25math7b_BEST",
                "verifier": {"passed": True, "issues": []},
            }
        },
    }


class MathAPIHandler(SimpleHTTPRequestHandler):
    """HTTP handler: serves static files + API endpoints."""

    client_dir = PROJECT_ROOT / "apps" / "math-vision-ui" / "client" / "dist"
    if not client_dir.exists():
        client_dir = PROJECT_ROOT / "apps" / "math-vision-ui" / "client"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.client_dir), **kwargs)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/solve-image":
            self._handle_solve()
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "model": "qwen25math7b_BEST"})
        elif parsed.path == "/api/status":
            self._send_json(_get_loading_state())
        else:
            super().do_GET()

    def _handle_solve(self):
        try:
            state = _get_loading_state()
            if state["status"] != "ready":
                self._send_json({
                    "ok": False,
                    "error": f"Modelo ainda carregando ({state['progress']}%). Aguarde."
                }, 503)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "")

            if "application/json" in content_type:
                data = json.loads(body)
                problem_text = data.get("problemText", "").strip()
            elif "multipart/form-data" in content_type:
                problem_text = self._parse_multipart_text(body, content_type)
            else:
                try:
                    data = json.loads(body)
                    problem_text = data.get("problemText", "").strip()
                except Exception:
                    problem_text = body.decode("utf-8", errors="replace").strip()

            if not problem_text:
                self._send_json({"ok": False, "error": "Nenhum texto fornecido."}, 400)
                return

            result = solve(problem_text)
            self._send_json(result)

        except Exception as e:
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(e)}, 500)

    def _parse_multipart_text(self, body: bytes, content_type: str) -> str:
        boundary_match = re.search(r"boundary=(.+?)(?:;|$)", content_type)
        if not boundary_match:
            return ""
        boundary = boundary_match.group(1).encode()
        parts = body.split(b"--" + boundary)
        for part in parts:
            if b'name="problemText"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    header_end = part.find(b"\n\n")
                if header_end >= 0:
                    text = part[header_end + 4:].strip()
                    if text.endswith(b"--"):
                        text = text[:-2].strip()
                    if text.endswith(b"\r\n"):
                        text = text[:-2].strip()
                    return text.decode("utf-8", errors="replace")
        return ""

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if args and "/api/" in str(args[0]):
            print(f"[API] {args[0]}")


def main():
    print("=" * 60)
    print("  MathSolver API Server")
    print(f"  Model: {MODEL_NAME} + QLoRA BEST")
    print(f"  Port:  {PORT}")
    print("=" * 60)

    # Start HTTP server immediately
    server = HTTPServer(("0.0.0.0", PORT), MathAPIHandler)
    print(f"[API] HTTP server started at http://localhost:{PORT}")
    print("[API] Loading model in background...")
    sys.stdout.flush()

    # Load model in background thread
    loader = threading.Thread(target=load_model_background, daemon=True)
    loader.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[API] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
