"""Compare OCR models hosted on Hugging Face.

Backends:
  - prithivMLmods/Multimodal-OCR Space (ZeroGPU): 5 models, accepts a prompt
  - nielsr/MiniCPM-OCR-vs-Unlimited-OCR-2 Space (ZeroGPU): MiniCPM-OCR + baidu/Unlimited-OCR-2, no prompt
  - Qwen/Qwen3.8-27B via HF Inference Providers (needs HF credits)
  - Gemini API (gemini-3.8-flash, gemini-pro-latest)

Usage:
    python ocr_compare.py image.png
    python ocr_compare.py image.png --models RolmOCR-7B Unlimited-OCR-2 Qwen3.8-27B
    python ocr_compare.py image.png --prompt "Extract the table as markdown"
Secrets come from the environment only, never from flags:
    HF_TOKEN        HF router (Qwen3.8); sent to the third-party Spaces only with --send-token-to-spaces
    GEMINI_API_KEY  Gemini models
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from gradio_client import Client, handle_file

MULTIMODAL_SPACE = "prithivMLmods/Multimodal-OCR"
COMPARE_SPACE = "nielsr/MiniCPM-OCR-vs-Unlimited-OCR-2"
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

MULTIMODAL_MODELS = ["Nanonets-OCR2-3B", "olmOCR-7B-0725", "RolmOCR-7B", "Aya-Vision-8B", "Qwen2-VL-OCR-2B"]
COMPARE_MODELS = {"MiniCPM-OCR-1.0": "minicpm", "Unlimited-OCR-2": "uocr"}
ROUTER_MODELS = {"Qwen3.8-27B": "Qwen/Qwen3.8-27B"}
GEMINI_MODELS = {"Gemini-3.8-Flash": "gemini-3.8-flash", "Gemini-Pro": "gemini-pro-latest"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
MODELS = MULTIMODAL_MODELS + list(COMPARE_MODELS) + list(ROUTER_MODELS) + list(GEMINI_MODELS)

DEFAULT_PROMPT = "Extract all the text in this image exactly as it appears."

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024      # under Gemini's 20 MB inline-request limit
MAX_RESPONSE_BYTES = 10 * 1024 * 1024   # a page of text is a few KB; refuse anything absurd


def check_image(path: Path) -> str:
    """Refuse anything that is not a reasonably sized image before it leaves this machine."""
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in ALLOWED_IMAGE_TYPES:
        raise SystemExit(f"Not a PNG/JPEG/WebP image: {path} ({mime})")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise SystemExit(f"Image larger than {MAX_IMAGE_BYTES // 2**20} MB: {path}")
    return mime


def redact(text: str) -> str:
    """Remove any secret value from text that gets printed or saved."""
    for name in ("HF_TOKEN", "GEMINI_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, f"<{name}>")
    return text


def read_json(req) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read(2000).decode(errors='replace')[:500]}") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Response too large")
    return json.loads(raw)


def to_data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def run_multimodal(client, model, image_url, prompt, max_tokens):
    return client.predict(
        model_name=model, text=prompt, image_b64=image_url,
        max_new_tokens_v=max_tokens, temperature_v=0.1, top_p_v=0.9, top_k_v=50,
        repetition_penalty_v=1.1, gpu_timeout_v=60, api_name="/run_ocr",
    )


def blocks_to_text(entry: dict) -> str:
    if entry.get("status") == "error":
        raise RuntimeError(entry.get("error", "unknown error"))
    return "\n\n".join(b["md"] for b in entry.get("blocks", []) if b.get("md") and not b.get("aux"))


def run_compare(client, image: Path) -> dict:
    """One call returns both MiniCPM-OCR and Unlimited-OCR-2 results."""
    return client.predict(handle_file(str(image)), True, "auto", api_name="/compare")


def run_router(model_id, image_url, prompt, max_tokens, token):
    body = {
        "model": model_id, "max_tokens": max_tokens, "temperature": 0.1,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]}],
    }
    req = urllib.request.Request(ROUTER_URL, json.dumps(body).encode(), {
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    return read_json(req)["choices"][0]["message"]["content"]


def run_gemini(model_id, image: Path, mime, prompt, key):
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(image.read_bytes()).decode()}},
        {"text": prompt},
    ]}], "generationConfig": {"temperature": 0.1}}
    req = urllib.request.Request(GEMINI_URL.format(model_id), json.dumps(body).encode(), {
        "x-goog-api-key": key, "Content-Type": "application/json"})
    parts = read_json(req)["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, metavar="MODEL")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--send-token-to-spaces", action="store_true",
                    help="also send $HF_TOKEN to the third-party Spaces (more ZeroGPU quota). "
                         "Their owners' code can read it: use a fine-grained token with no write access.")
    args = ap.parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    space_token = hf_token if args.send_token_to_spaces else None

    img = Path(args.image)
    mime = check_image(img)
    image_url = to_data_url(img, mime)
    out_dir = Path("results") / img.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    clients, compare_result = {}, None

    def client_for(space):
        if space not in clients:
            clients[space] = Client(space, token=space_token, verbose=False)
        return clients[space]

    for model in args.models:
        print(f"\n{'=' * 60}\n{model}\n{'=' * 60}")
        t0 = time.time()
        try:
            if model in MULTIMODAL_MODELS:
                text = run_multimodal(client_for(MULTIMODAL_SPACE), model, image_url, args.prompt, args.max_tokens)
            elif model in COMPARE_MODELS:
                if compare_result is None:  # first of the pair runs both
                    compare_result = run_compare(client_for(COMPARE_SPACE), img)
                text = blocks_to_text(compare_result[COMPARE_MODELS[model]])
            elif model in GEMINI_MODELS:
                key = os.environ.get("GEMINI_API_KEY")
                if not key:
                    raise RuntimeError("Gemini needs $GEMINI_API_KEY")
                text = run_gemini(GEMINI_MODELS[model], img, mime, args.prompt, key)
            else:
                if not hf_token:
                    raise RuntimeError("Qwen3.8 needs $HF_TOKEN")
                text = run_router(ROUTER_MODELS[model], image_url, args.prompt, args.max_tokens, hf_token)
        except Exception as e:
            text = f"[ERROR] {e}"
        text = redact(str(text))
        dt = time.time() - t0
        print(text)
        print(f"--- {dt:.1f}s")
        (out_dir / f"{model}.txt").write_text(str(text), encoding="utf-8")

    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
