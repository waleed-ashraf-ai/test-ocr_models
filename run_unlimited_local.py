"""Run baidu/Unlimited-OCR-2 locally with transformers and save in the same layout as ocr_compare.py.

    .venv/Scripts/python run_unlimited_local.py pages/page_4.png pages/page_5.png

Writes results/<image stem>/Unlimited-OCR-2.txt, so score_ocr.py picks it up.

Runs on CUDA when a GPU with enough memory is present, otherwise on CPU in bf16
(~7 GB RAM, minutes per page).

Security:
  - trust_remote_code runs Python from the model repo. REVISION pins the exact commit that was
    reviewed, so a later push to the repo cannot change the code that runs here.
  - The repo's save_results=True path calls eval() on model output (drawing boxes). This script
    always passes save_results=False, so that path never runs.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "baidu/Unlimited-OCR-2"
REVISION = "0480966938291445591f69f08a13b041c073a96a"  # reviewed 2026-11-08
MIN_GPU_BYTES = 10 * 1024**3  # weights alone are 6.7 GB


def pick_device() -> str:
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= MIN_GPU_BYTES:
        return "cuda"
    # The repo hardcodes .cuda() in two places (modeling_unlimitedocr.py:785, :789).
    # On CPU, make it a no-op so those tensors stay on the CPU.
    torch.Tensor.cuda = lambda self, *a, **k: self
    return "cpu"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}, threads: {torch.get_num_threads()}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True,
                                      use_safetensors=True, torch_dtype=torch.bfloat16).eval().to(device)
    print(f"loaded in {time.time() - t0:.0f}s")

    for image in map(Path, args.images):
        if not image.is_file():
            print(f"skip, not a file: {image}")
            continue
        t0 = time.time()
        try:
            with torch.inference_mode():
                text = model.infer(tok, image_file=str(image), save_results=False, eval_mode=True,
                                   max_new_tokens=args.max_new_tokens)
        except Exception as e:
            text = f"[ERROR] {type(e).__name__}: {e}"
        out = Path("results") / image.stem / "Unlimited-OCR-2.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(str(text), encoding="utf-8")
        print(f"\n=== {image.name} ({time.time() - t0:.0f}s) → {out}\n{text}")


if __name__ == "__main__":
    main()
