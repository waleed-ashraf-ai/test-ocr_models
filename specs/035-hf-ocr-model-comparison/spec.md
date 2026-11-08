# Feature Specification: Hugging Face + Gemini OCR model comparison

**Feature:** `035-hf-ocr-model-comparison` · **Created:** 2026-11-08 · **Status:** Partially implemented
**Input:** "Try open OCR models hosted on Hugging Face (and Gemini) on real pages of our children's
books and tell me what percentage each one reads correctly."
**Code:** project root (`ocr_compare.py`, `score_ocr.py`, `results.html`) · **Related:** 019 (reference transcripts), 026 (tier benchmark)

## Problem

We pay for OCR on every book. Several free or cheap OCR models exist on Hugging Face, and we
had no numbers on how they read our books: Arabic, fully diacritized, decorative title fonts,
text over illustrations. Picking an engine needs a measured accuracy per model on our own pages,
not vendor claims.

## User Scenarios & Testing

### User Story 1 — Run every model on a page image (P1)

A developer gives one page image and gets each model's raw transcription saved to disk,
including the models that failed and why.

**Why this priority:** without outputs there is nothing to score.
**Independent test:** `python ocr_compare.py pages/page_4.png` writes one `.txt` per model to
`results/page_4/`; a failed model's file starts with `[ERROR]`.

**Acceptance scenarios:**
1. **Given** a PNG page, **When** the script runs with defaults, **Then** every model in `MODELS` has a file in `results/<image stem>/`.
2. **Given** `--models Gemini-Pro RolmOCR-7B`, **When** it runs, **Then** only those two are called and other files are left untouched.
3. **Given** a model that hits a quota or HTTP error, **When** it runs, **Then** the error text is saved and the remaining models still run.

### User Story 2 — One accuracy number per model (P1)

A developer or product owner sees, per page and overall, what percentage of the text each model
read correctly.

**Independent test:** `python score_ocr.py` prints a table per page with character and word accuracy, errors listed as `error`.

**Acceptance scenarios:**
1. **Given** a hand-made reference transcript for a page, **When** scoring runs, **Then** each model gets char accuracy = 1 − CER and word accuracy = 1 − WER after normalization, clamped at 0.
2. **Given** a model output with an extra page footer, `<img>` tags or markdown, **When** scoring runs, **Then** those are stripped and not counted as errors.
3. **Given** a model that wraps the text in commentary or a translation, **When** scoring runs, **Then** the extra text counts against it (an OCR engine must return only the text).

### User Story 3 — A shareable report (P2)

A non-developer opens one HTML page and sees the ranking, per-page tables and caveats.

**Independent test:** open `results.html` in a browser.

### Edge cases

- ZeroGPU daily quota exhausted mid-run → model saved as `[ERROR]`, scoring skips it, report says "did not run".
- Book typo in the source (page 4 prints «صدقين»); a model that "fixes" it to «صديقين» loses points. The reference follows the printed page.
- A model scored on one page only is not directly comparable with one scored on two; the report must say which pages each number covers.

## Requirements

### Functional requirements

- **FR-001**: `ocr_compare.py` MUST support four backends: the `prithivMLmods/Multimodal-OCR` Space, the `nielsr/MiniCPM-OCR-vs-Unlimited-OCR-2` Space, HF Inference Providers (router), and the Gemini API.
- **FR-002**: Secrets MUST come from the environment (`HF_TOKEN`, `GEMINI_API_KEY`), never from code, command-line flags or committed files, and MUST be redacted from anything printed or saved.
- **FR-002a**: `HF_TOKEN` MUST NOT be sent to third-party Spaces unless the user passes `--send-token-to-spaces`.
- **FR-002b**: Only PNG/JPEG/WebP files up to 15 MB may be sent; API responses are capped at 10 MB.
- **FR-003**: One model failing MUST NOT stop the others (per-model isolation, constitution V).
- **FR-004**: `score_ocr.py` MUST normalize before comparing: drop harakat and tatweel, unify أ/إ/آ/ٱ → ا and ى → ي, drop punctuation, the page footer and `<img>` blocks.
- **FR-005**: Reported numbers MUST include failed models and the pages each number is based on (constitution X).
- **FR-006**: [NEEDS CLARIFICATION: should diacritics accuracy (DER) be scored too? Today diacritics are stripped, so a model that drops all tashkeel can still score 100%.]

### Key entities

- **Page image**: `pages/page_N.png`, rendered from `الأمانة.pdf` (story «وظيفة يوم العطلة»; printed page numbers 2, 3 = file pages 4, 5).
- **Model output**: `results/<page>/<model>.txt`, raw text or `[ERROR] …`.
- **Reference transcript**: the printed text of a page, typed by a person. Today: the `GT` dict in `score_ocr.py`.

## Success Criteria

- **SC-001**: Every model in `MODELS` has a scored result on at least 4 pages (today: 2 pages, some models on 1).
- **SC-002**: The report states a recommended engine and the reason, written by a person.
- **SC-003**: Re-running the whole comparison needs only the two env vars and one command per page.

## Assumptions & Out of scope

- Pages are single-column story text; tables, multi-column layouts and handwriting are out of scope.
- Speed and $/page are recorded informally (seconds printed per call), not benchmarked; that belongs to 026.
- Production integration of any of these models is out of scope; this is an evaluation.
