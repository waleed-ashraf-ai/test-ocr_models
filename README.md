# OCR model comparison (Hugging Face + Gemini)

Which OCR model reads our Arabic children's books best? This folder runs several models hosted
on Hugging Face, plus Gemini, on real book pages and scores each one against a hand-typed
transcript.

Spec, plan and open tasks: [`specs/035-hf-ocr-model-comparison/`](specs/035-hf-ocr-model-comparison/).

## Results so far (2026-11-10)

7 pages from 3 children's books: «الديناصورات» (file pages 5, 8, 11) and «المومياوات» (file pages 5, 14), both National Geographic,
and «وظيفة يوم العطلة» (pages 4–5). Each model was asked to copy the text **with tashkeel exactly as printed** (`--tashkeel`).
Dashboard for management: `results.html`, published at https://claude.ai/artifact/XaaaUcVFkkaXF7mNKR8sca

Overall = average of word accuracy and tashkeel accuracy. Tashkeel must match the book: a missing, wrong or **added** mark
(the model guessing tashkeel the book does not print) counts as an error. † = some pages fell back to the regular-prompt run
because credit or quota ran out before the tashkeel run.

| # | Model | Overall | Words | Tashkeel | Pages |
|---|---|---|---|---|---|
| 1 | Gemini-3.8-Flash | 94.8% | 100.0% | 89.6% | 7/7 |
| 2 | Kimi-K3 † | 92.8% | 98.4% | 87.2% | 2/7 |
| 3 | Gemini-Pro | 87.6% | 99.9% | 75.3% | 7/7 |
| 4 | Qwen2.5-VL-72B | 79.0% | 95.7% | 62.2% | 5/7 |
| 5 | Llama-4-Maverick | 78.4% | 95.4% | 61.5% | 4/7 |
| 6 | GPT-6.1-Sol † | 76.8% | 98.0% | 55.6% | 2/7 |
| 7 | DeepSeek-V4-Flash-Vision † | 76.4% | 98.6% | 54.2% | 3/7 |
| 8 | Unlimited-OCR-2 † | 69.8% | 96.7% | 43.0% | 4/7 |
| 9 | Qwen3-VL-235B | 67.2% | 81.9% | 52.5% | 5/7 |
| 10 | GPT-4.1 † | 56.3% | 98.3% | 14.3% | 3/7 |

- Recommendation: **Gemini-3.8-Flash** — best overall on all 7 pages, never invented words, ≈ $461 per 1000 books (30 pages each), ≈ half with Batch.
- Gemini-Pro reads perfectly but adds tashkeel the book does not print, so it loses on tashkeel; it also costs ≈ 3.6× more (thinking tokens).
- Kimi-K3 ran on 2 pages only and invented words in a heading («عظام مفترسة وعظام ضخمة!»).
- RolmOCR, olmOCR, Nanonets, MiniCPM read letters very well but drop almost all tashkeel.
- Missing runs: OpenRouter credit at $0, free ZeroGPU quota exhausted, Qwen3.8-27B needs HF credits, free Gemma rate-limited.
- Claude models were run but are left out of the dashboard and this table on request.

## How to run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...         # required for the Gemini models
export HF_TOKEN=hf_...            # required for Qwen3.8-27B (see Security before using it with Spaces)

# 1. run models on a page (writes results/page_4/<model>.txt)
python ocr_compare.py pages/page_4.png
python ocr_compare.py pages/page_5.png --models Gemini-Pro RolmOCR-7B --max-tokens 4096
python ocr_compare.py pages/page_5.png --send-token-to-spaces   # more ZeroGPU quota, see Security

# 2. score every saved output (letters, words, tashkeel, added tashkeel, full)
python score_ocr.py
python score_ocr.py --dir results_tashkeel --csv scores.csv

# 3. same run, asking the models to keep tashkeel exactly as printed (saves to results_tashkeel/)
python ocr_compare.py pages/dino_5.jpg --tashkeel
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
