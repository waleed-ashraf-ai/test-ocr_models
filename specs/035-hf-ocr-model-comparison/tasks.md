# Tasks: Hugging Face + Gemini OCR model comparison

**Input:** `specs/035-hf-ocr-model-comparison/` (spec.md, plan.md)

Format: `- [ ] T### [P] [US#] description — path`

## Phase 1 — Done (2026-11-08)

- [x] T001 [US1] Multimodal-OCR Space backend, 5 models — `ocr_compare.py`
- [x] T002 [US1] MiniCPM / Unlimited-OCR-2 Space backend — `ocr_compare.py` (`run_compare`)
- [x] T003 [US1] HF router backend for Qwen3.8-27B — `ocr_compare.py` (`run_router`)
- [x] T004 [US1] Gemini backend, key from `$GEMINI_API_KEY` — `ocr_compare.py` (`run_gemini`)
- [x] T005 [US1] Pages 4–7 rendered from `الأمانة.pdf` — `pages/`
- [x] T006 [US1] Runs on page 4 and 5 — `results/page_4/`, `results/page_5/`
- [x] T007 [US2] Reference transcripts for pages 4, 5 + char/word scorer — `score_ocr.py`
- [x] T008 [US3] HTML report — `results.html`
- [x] T009 Security hardening: secrets env-only (no `--hf-token` flag), HF token kept from third-party Spaces unless `--send-token-to-spaces`, secret redaction in output, image type/size check, response size cap, pinned `requirements.txt`, `.gitignore`, `.env.example` — `ocr_compare.py`

## Phase 2 — Complete the runs (blocked on quota / credits)

- [ ] T010 [US1] Re-run page 5 for Nanonets-OCR2-3B, Qwen2-VL-OCR-2B, Aya-Vision-8B once ZeroGPU quota resets (or with an HF PRO token and `--send-token-to-spaces`)
- [ ] T011 [US1] Run MiniCPM-OCR-1.0 on pages 4, 5
- [x] T011a [US1] Unlimited-OCR-2 run locally on pages 4, 5 (CPU, 1087 s and 742 s) — `run_unlimited_local.py`, `results/page_*/Unlimited-OCR-2.txt`
- [ ] T012 [US1] Qwen3.8-27B: needs HF prepaid credits — ask tech lead, or drop it from `MODELS`
- [ ] T013 [P] [US1] Re-run Gemini-3.8-Flash with a strict prompt ("return only the text, no commentary, no translation") and record both scores
- [ ] T014 [US1] Run all models on pages 6 and 7 (images already in `pages/`)

## Phase 3 — Make the scoring reusable

- [ ] T020 [US2] Move `GT` out of `score_ocr.py` into one file per page (reuse 019's reference transcript format)
- [ ] T021 [US2] Generate `results.html` from the scores instead of typing numbers in by hand
- [ ] T022 [US2] Decide FR-006 (diacritics accuracy) and, if yes, add DER next to CER/WER
- [ ] T023 [P] [US2] Record seconds per call and $ per page per model in the output

## Phase 4 — Tests & polish

- [ ] T030 [P] Hermetic tests for `norm` and `lev` and for one canned model output per backend (no network) — constitution VIII
- [ ] T031 Fill the recommendation section of the report (a person, not generated) — SC-002

## Dependencies

T010–T014 before T031. T020 before T021. T030 is independent.
