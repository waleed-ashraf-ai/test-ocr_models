# OCR model comparison (Hugging Face + Gemini)

Which OCR model reads our Arabic children's books best? This folder runs several models hosted
on Hugging Face, plus Gemini, on real book pages and scores each one against a hand-typed
transcript.

Spec, plan and open tasks: [`specs/035-hf-ocr-model-comparison/`](specs/035-hf-ocr-model-comparison/).

## Results so far (pages 4–5 of «وظيفة يوم العطلة», 2026-11-08)

| # | Model | Reads correctly (chars) | Pages |
|---|---|---|---|
| 1 | Gemini-Pro | 99.7% | 4, 5 |
| 2 | RolmOCR-7B | 99.5% | 4, 5 |
| 3 | Nanonets-OCR2-3B | 99.1% | 4 only |
| 4 | olmOCR-7B-0725 | 97.8% | 4, 5 |
| 5 | Unlimited-OCR-2 | 97.0% | 4, 5 (local, CPU, ~15 min/page) |
| 6 | Qwen2-VL-OCR-2B | 89.6% | 4 only |
| 7 | Gemini-3.8-Flash | 35.2% | 4, 5 (added a translation and image description around the text) |
| 8 | Aya-Vision-8B | 20.4% | 4 only (made up text) |
| — | MiniCPM-OCR-1.0 | did not run | ZeroGPU quota |
| — | Qwen3.8-27B | did not run | needs HF credits |

Open `results.html` in a browser for the full report with per-page tables.

## How to run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...         # required for the Gemini models
export HF_TOKEN=hf_...            # required for Qwen3.8-27B (see Security before using it with Spaces)

# 1. run models on a page (writes results/page_4/<model>.txt)
python ocr_compare.py pages/page_4.png
python ocr_compare.py pages/page_5.png --models Gemini-Pro RolmOCR-7B --max-tokens 4096
python ocr_compare.py pages/page_5.png --send-token-to-spaces   # more ZeroGPU quota, see Security

# 2. score every saved output
python score_ocr.py
```

Unlimited-OCR-2 can also run locally (6.7 GB of weights; GPU with ≥ 10 GB, else CPU in bf16 with ~7 GB RAM):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install transformers==4.52.1 huggingface-hub==0.36.0 tokenizers==0.21.4 addict==2.4.0 einops==0.8.2 easydict==1.13 pillow requests
.venv/Scripts/python -m pip install triton-windows   # Windows only; the model code imports triton
.venv/Scripts/python run_unlimited_local.py pages/page_4.png pages/page_5.png
```

It pins the model repo to a reviewed commit (it runs with `trust_remote_code`) and never calls the repo's `save_results` path, which runs `eval()` on model output.

On Windows, set the variables once under *Environment Variables → User variables*.

## Files

| Path | What it is |
|---|---|
| `ocr_compare.py` | sends one image to each model, saves raw text; failures saved as `[ERROR] …` |
| `score_ocr.py` | reference transcripts (`GT` dict) and the scoring table |
| `results.html` | the report; numbers are typed in by hand, update them after a new run |
| `pages/` | page images rendered from `الأمانة.pdf` (file page 4 = printed page 2) |
| `results/<page>/` | one `.txt` per model; `results/sample/` is an early smoke test whose image is not here |
| `requirements.txt` | pinned `gradio_client` version |
| `.env.example` | names of the two secrets, empty |
| `specs/035-…/` | spec kit: spec, plan, tasks |
| `الأمانة.pdf` | source book the page images were rendered from |
| `الأمانة/`, `الأمانة-راجل/`, `الأنانية/`, `العند/` | story folders (page JSON + audio); not used by these scripts |

## How the score works

Before comparing, both texts are normalized: diacritics, tatweel, punctuation, the page footer
and `<img>` tags are removed; أ/إ/آ become ا and ى becomes ي. Then
`accuracy = 1 − edit distance / reference length`, on characters and on words.

Things to know when reading the numbers:

- Diacritics are not scored. A model that drops all tashkeel can still get 100%.
- The reference follows the printed page, typos included (page 4 prints «صدقين»). A model that corrects it loses a little.
- Commentary around the text counts as errors. That is why Gemini-3.8-Flash scores low even though its transcription was almost right.

## Security

- **Secrets live only in environment variables.** There is no `--hf-token` flag on purpose: a token typed on the command line ends up in shell history and the process list. `.gitignore` keeps `.env` files out of git.
- **Secrets are redacted.** If an error message ever contains `HF_TOKEN` or `GEMINI_API_KEY`, it is replaced with `<HF_TOKEN>` / `<GEMINI_API_KEY>` before it is printed or saved to `results/`.
- **Third-party Spaces don't get your token by default.** `prithivMLmods/Multimodal-OCR` and `nielsr/MiniCPM-OCR-vs-Unlimited-OCR-2` run code owned by other people, and a token sent to them can be read by that code. Without `--send-token-to-spaces` they are called anonymously, with a smaller ZeroGPU quota. If you pass the flag, use a fine-grained HF token with no write permissions.
- **Page images leave this machine.** Every run uploads the image to Google (Gemini), Hugging Face, and those two third-party Spaces. Don't run it on books we are not allowed to share.
- **Only images are sent.** The script refuses anything that isn't a PNG/JPEG/WebP file up to 15 MB, so a whole PDF can't be uploaded by mistake. Responses over 10 MB are refused.
- **Gemini key shared in chat.** A Gemini key was pasted into a chat on 2026-11-08. Rotate it in Google AI Studio if it is still active.

## Where to continue

See [`tasks.md`](specs/035-hf-ocr-model-comparison/tasks.md). The next useful steps are:

1. Re-run page 5 for the models that hit the ZeroGPU quota, and run pages 6–7.
2. Re-run Gemini-3.8-Flash with a "text only" prompt.
3. Move the reference transcripts out of `score_ocr.py` into files and add a new page's transcript there.
