# Implementation Plan: Hugging Face + Gemini OCR model comparison

**Feature:** `035-hf-ocr-model-comparison` · **Date:** 2026-11-08 · **Spec:** [spec.md](spec.md)

## Summary

Two standalone scripts and one static report in this folder. `ocr_compare.py` sends one page image to each model and saves the raw text;
`score_ocr.py` compares the saved texts with a hand-typed reference using Levenshtein distance
on characters and on words. `results.html` presents the numbers.

## Technical context

| | |
|---|---|
| Language / version | Python 3.10+ |
| Primary dependencies | `gradio_client` (HF Spaces); stdlib `urllib` for HF router and Gemini REST |
| Storage | plain files: `pages/*.png` in, `results/<page>/<model>.txt` out |
| Testing | none yet (see tasks T030) |
| Target platform | developer machine, Windows or Linux |
| Performance goals | none; ~10–40 s per model per page |
| Constraints (cost, quota) | ZeroGPU Spaces: small free daily quota per HF account, PRO raises it. HF router (Qwen3.8-27B) needs prepaid credits. Gemini: paid per call, ~1 call per model per page |
| Scale | a handful of pages |

## Models

| Name in script | Backend | Model id |
|---|---|---|
| Nanonets-OCR2-3B, olmOCR-7B-0725, RolmOCR-7B, Aya-Vision-8B, Qwen2-VL-OCR-2B | Space `prithivMLmods/Multimodal-OCR` `/run_ocr` | (chosen by name in the Space) |
| MiniCPM-OCR-1.0, Unlimited-OCR-2 | Space `nielsr/MiniCPM-OCR-vs-Unlimited-OCR-2` `/compare` (one call returns both) | — |
| Qwen3.8-27B | HF router `chat/completions` | `Qwen/Qwen3.8-27B` |
| Gemini-3.8-Flash, Gemini-Pro | Gemini `generateContent` | `gemini-3.8-flash`, `gemini-pro-latest` |

Prompt for all promptable models: `Extract all the text in this image exactly as it appears.`

## Scoring

```
norm(text):  drop <img>…</img>, the "N | وظيفة يوم العطلة" footer line,
             harakat U+064B–U+0652, U+0670, tatweel U+0640,
             أ إ آ ٱ → ا,  ى → ي,  everything not an Arabic letter → space, collapse spaces
char acc  =  max(0, 1 − lev(chars(norm(out)), chars(norm(ref))) / len(ref))
word acc  =  same on word lists
overall   =  1 − Σ edits / Σ ref length over the pages the model ran on
```

## Results so far (2026-11-08)

| Model | Overall char acc | Page 4 char / word | Page 5 char / word |
|---|---|---|---|
| Gemini-Pro | 99.7% | 99.5 / 97.3 | 100 / 100 |
| RolmOCR-7B | 99.5% | 99.1 / 97.3 | 100 / 100 |
| Nanonets-OCR2-3B | 99.1% (p4 only) | 99.1 / 94.6 | error |
| olmOCR-7B-0725 | 97.8% | 96.7 / 83.8 | 99.4 / 96.3 |
| Unlimited-OCR-2 (local, CPU) | 97.0% | 97.6 / 86.5 | 96.2 / 81.5 |
| Qwen2-VL-OCR-2B | 89.6% (p4 only) | 89.6 / 73.0 | error |
| Gemini-3.8-Flash | 35.2% | 0 / 0 (added translation + image description) | 82.3 / 81.5 |
| Aya-Vision-8B | 20.4% (p4 only) | 20.4 / 0 (hallucinated text) | error |
| MiniCPM-OCR-1.0 | — | ZeroGPU quota | — |
| Qwen3.8-27B | — | HTTP 402, no HF credits | — |

## Constitution check

| Principle | Status | Note |
|---|---|---|
| I Correct before diacritize | n/a | evaluation only |
| II Stable token anchoring | n/a | |
| III Uncertain ⇒ keep original | n/a | |
| IV Stage proves its lane | n/a | |
| V Per-page isolation & resume | ✅ | per-model try/except; per-page output folder |
| VI Prefer free exact path | ✅ | free Spaces first; Gemini only on request |
| VII Cost is a design input | ⚠️ | Gemini calls are not metered (025 not wired in) |
| VIII Test without paying | ❌ | no hermetic tests; every run calls live APIs |
| IX Layering | ✅ | standalone folder |
| X Honest reporting | ✅ | failures and one-page-only scores are shown, not hidden |

## Project structure

```
./
  README.md          how to run, where things are, security, what is left
  requirements.txt   pinned gradio_client
  .env.example       the two env vars, empty
  .gitignore         keeps .env and caches out of git
  ocr_compare.py     run models on one image → results/<stem>/<model>.txt
  score_ocr.py       reference transcripts (GT dict) + scoring table
  results.html       static report (numbers typed in, not generated)
  pages/             page_4..page_7.png rendered from الأمانة.pdf
  results/           page_4/, page_5/, sample/ (early smoke test; its image is not in the repo)
specs/035-hf-ocr-model-comparison/
```

## Complexity tracking

| Violation | Why needed | Simpler alternative rejected because |
|---|---|---|
| Reference text hardcoded in `score_ocr.py` | 2 pages, fast to start | Moving to files is task T020; should reuse 019's transcript format |
