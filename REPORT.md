# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `IA-Group` |
| Members | Daniella Campos, Rebeca Michelle Castro Pineda, Rodrigo Marroquin, Aracely Verenice Varela |
| Repository URL | https://github.com/DaniellaCampos/proyecto-qa |
| `submission` commit hash | The commit tagged `submission` (a commit cannot contain its own hash; we send the hash together with the repository URL) |
| Public service URL used at the dry run | https://hallowedly-miasmatical-leda.ngrok-free.dev (ngrok, free plan) |
| AI tools used (which, and for what) | **Training (Daniella Campos):** Claude Code, to understand the brief, generate `training/train.py` and fix library version errors. **Evaluation and threshold (Rebeca Castro):** Gemini, for technical support setting up Google Colab with a T4 GPU, checking dependencies and running `evaluation/choose_threshold.py`. **API and tests (Rodrigo Marroquin):** Claude Code, as an assistant to run `tests.check_api`, test edge cases (invalid JSON, empty fields, contexts over 5000 characters, lowercase `bearer`), measure CPU latency and draft report sections. **Tunnel, report and submission (Aracely Varela):** Claude, to review the repository, guide the ngrok setup and dry run, fix the extra-fields validation in `/predict` and adapt this report to the template. The model runs locally; no AI API is called by the service. |

## 1. Model choice and justification

- **Base checkpoint** (exact Hugging Face id): `distilbert/distilbert-base-uncased` (66,364,418 parameters with the QA head).
- **Alternatives considered:** BERT-base (about 2× slower on CPU); RoBERTa-base and DeBERTa-v3-base (usually more accurate on SQuAD 2.0, but slower on CPU and slower to fine-tune in the 1.5-week window).
- **Trade-off:** serving is CPU-only with a 5 s timeout per request and a 150 s cap. DistilBERT is ~40% smaller and ~60% faster than BERT-base, so even multi-window contexts stay far below the timeout (Section 6). We trade a few accuracy points for a safe latency margin through the tunnel.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Random subset of 50,000 SQuAD 2.0 train questions (seed 42) → 50,528 training windows |
| Validation data (what you validated on, how many examples) | Random subset of 2,000 SQuAD 2.0 dev questions (seed 0), 48.95% unanswerable, never seen in training |
| Epochs | 1 |
| Learning rate and schedule (warmup?) | 3e-5, linear decay, 10% warmup, weight decay 0.01 |
| Batch size | 16 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), fp16, ~6.3 minutes (3,158 steps) |

Unanswerable questions point at `[CLS]` (position 0). Script: `training/train.py`.

- **Loss curve:** [`training/logs/loss_curve.png`](training/logs/loss_curve.png) (log: `training/logs/log_history.json`); loss 5.94 → 1.52.
- **Validation results** (`evaluation/choose_threshold.py`, saved in `evaluation/threshold.json`):

| Setting | Exact Match | F1 |
|---|---|---|
| Tuned threshold (−1.224) | **0.5865** | **0.6095** |
| Threshold = 0 | 0.5590 | 0.5933 |
| Always-null baseline | 0.4895 | — |

We did not compute EM/F1 separately for answerable and unanswerable questions.

## 3. No-answer threshold

- **Decision rule:** for each question, `null_score` is the start + end logit of `[CLS]` (minimum over windows), and `best_span_score` is the best start + end logit among valid spans inside the context (end ≥ start, at most 30 tokens, top-20 starts and ends). If `null_score − best_span_score > threshold`, the service returns `null`; otherwise it returns the span. The rule is in `qa_core.py`, shared by evaluation and the API.
- **How we chose it:** on the 2,000 validation questions we tried every observed difference as a threshold and kept the highest Exact Match (F1 as tie-breaker), counting a correct `null` as correct. **Final value: −1.224** (EM 0.5590 → 0.5865).
- **Too low or too high:** too low → more `null` answers, at the extreme the always-null baseline (0.4895 here, ~0.25 on the hidden set). Too high → almost always a span, so unanswerable questions fail. Our validation set is 49% unanswerable versus ~25% in the hidden set, so the threshold may lean slightly towards `null`.

## 4. Serving architecture

- **Framework and server:** FastAPI (`api/app.py`) on uvicorn, **1 worker**, port 8000; a lock serves one request at a time, like the harness sends them.
- **Model loading:** at startup a background thread loads the tokenizer, `models/qa-distilbert` and `evaluation/threshold.json`, then runs a warm-up prediction. Until then `/health` returns 503 `loading`, then 200 `ok`.
- **Request flow:** token check → JSON parsing and Pydantic validation (`id` 1–64, `context` 1–5000, `question` 1–500 characters, strict strings, extra fields rejected) → tokenization of question + context; contexts longer than 384 tokens are split into overlapping windows (stride 128) → best span over all windows → no-answer rule (Section 3) → token positions mapped back to character offsets, so `context[start:start+len(answer)] == answer` always holds → `{"id", "answer", "start", "score"}`. Invalid input → 400 `bad_request`; unexpected errors → 500 `internal_error`.
- **Access control:** `/info` and `/predict` compare the `Authorization` header with `"Bearer " + QA_API_TOKEN` exactly (`secrets.compare_digest`), returning 401 `unauthorized` otherwise; `/health` is public. The token only comes from the environment variable, and the app refuses to start without it.
- **Tunnel:** ngrok (free plan, one fixed domain per account): `ngrok http 8000`. Backup: `cloudflared tunnel --url http://localhost:8000`.
- **How to run** from a clean machine (Windows PowerShell, CPU only; the model folder is shared by Google Drive):

```powershell
git clone https://github.com/DaniellaCampos/proyecto-qa.git
cd proyecto-qa
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch
pip install -r requirements.txt
Expand-Archive <zip-downloaded-from-Drive>.zip -DestinationPath models   # gives models/qa-distilbert/config.json
$env:QA_API_TOKEN = "<token>"
$env:QA_TEAM_NAME = "IA-Group"
uvicorn api.app:app --host 0.0.0.0 --port 8000
# second terminal:
ngrok http 8000
# third terminal (same QA_API_TOKEN): contract checks + public sample through the tunnel
python -m tests.check_api https://<public-url>
```

## 5. Results on the public sample set

Dry run through the public tunnel URL (`python -m tests.check_api <public-url>`): **12/12 checks OK**, 4/5 correct answers. Screenshots of curl tests 1–4 (`/health`, `/info` without token, with a wrong token and with the valid token) through the public URL are delivered as a separate document with the submission.

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | "16" / "16 attention heads" | `null` | `null` | No |
| sample-02 | "3.5 days" | "3.5 days" | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `null` | `null` | Yes |
| sample-05 | "50,257" / "50,257 tokens" | "50,257" | 3157 | Yes |

**sample-01 failed:** the best span was `"24 layers, a hidden size of 1024 and 16"`; start and end are scored independently, so the model merged several numbers from one list. That span scored just below the no-answer option (difference −0.92 > −1.224), so the service returned `null`. We did not retune the threshold for one public item.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 254 MB (`model.safetensors`, fp32) |
| Mean latency per `/predict` request on your laptop (CPU) | Laptop A: 76 ms local (sample-05: 238 ms; 5,000-character context: 0.33 s). Laptop B: 0.16–0.22 s local (sample-05: 1.38 s); through ngrok ≈1.0 s (sample-05: 2.08 s) |
| Laptop CPU and RAM | Laptop A: AMD Ryzen 7 7435HS, 32 GB RAM. Laptop B: slower CPU-only laptop used for the dry run |

The tunnel adds about 0.7–0.9 s per request. Laptop A will serve on presentation day; laptop B is the backup.

## 7. Problems encountered and how you fixed them

1. **Problem:** the API could not start on the CPU laptop because `models/qa-distilbert` did not exist. **Cause:** weights are excluded from git (as required) and Google Drive downloads the shared folder as an automatically named zip (`qa-distilbert-<date>-1-001.zip`), not the `qa-distilbert.zip` our manual expected. **Fix:** extract the Drive zip into `models/` and check `models/qa-distilbert/config.json` before starting uvicorn. **Commit:** `d227cd3`
2. **Problem:** `POST /predict` accepted a body with extra fields (e.g. `"foo": 1`) and answered 200, but the contract says `additionalProperties: false`. **Cause:** Pydantic ignores unknown fields by default. **Fix:** `ConfigDict(strict=True, extra="forbid")` in `PredictRequest`; extra fields now return 400 `bad_request` ("Extra inputs are not permitted"). **Commit:** `d75295f`
3. **Problem:** in the dry run, the long-context item (sample-05) took 2.08 s through the tunnel on the second laptop. **Cause:** that laptop's CPU is about 5× slower for this model, and the tunnel adds ~0.8 s. **Fix:** we measured both laptops and chose the faster one to serve on presentation day. **Commit:** `12660e8`

## 8. Decoder investigation

1. **Distractor spans (sample-01).** *Problem:* a wrong merged span, then abstention. *Decoder:* reads question and context jointly and generates just "16". *Cost:* it may write "sixteen" or "It has 16 heads", failing Exact Match and the `start` rule unless the output is searched back in the context or decoding is constrained to copy.
2. **Span boundary errors (validation).** *Problem:* F1 (0.6095) > EM (0.5865): answers that add or cut an article, unit or word. *Encoder–decoder (e.g. T5):* generates a short answer with natural boundaries. *Cost:* paraphrases are also punished by EM.
3. **Non-span answers** (yes/no, counts). *Problem:* we can only copy one span of ≤30 tokens. *Decoder:* generates "yes" or "3". *Cost:* such answers have no `start`, so they break the contract.
4. **Long contexts (sample-05).** *Problem:* 384-token windows never combine evidence, and latency grows per window (1.38 s on laptop B). *Decoder:* a long context window sees the whole text. *Cost:* even a 1B decoder is ~15× DistilBERT and generates token by token, risking the 5 s timeout on CPU; it may also hallucinate instead of abstaining.

## 9. Known limitations

- **Moderate accuracy:** EM 0.5865 / F1 0.6095, after 1 epoch on 50,000 of the ~130,000 training questions.
- **Only extractive answers:** one continuous span of at most 30 tokens; yes/no, counting or multi-sentence questions get a wrong span or `null`.
- **One global threshold:** tuned on a set that is 49% unanswerable while the hidden set is ~25%; questions near the threshold (like sample-01) can flip between `null` and a span.
- **English only:** other languages give poor answers.
- **Input limits:** `context` over 5,000 characters, `question` over 500, empty fields, non-string types or extra fields → 400 `bad_request`.
- **One request at a time:** concurrent requests wait in line.
- **Laptop and tunnel dependent:** sleep, power or network loss, or a new tunnel URL make the service unreachable.
- **Single shared token:** no rate limiting and no per-user keys.
