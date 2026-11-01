# REPORT

## Team

- Team name: IA-Group
- Members:
  - Daniella Campos — Persona 1: model training
  - Rebeca Michelle Castro Pineda — Persona 2: evaluation and no-answer threshold
  - Rodrigo Marroquin — Persona 3: API and tests
  - Aracely Verenice Varela — Persona 4: tunnel, final report and submission

## Model choice and justification

We fine-tuned **DistilBERT** (`distilbert/distilbert-base-uncased`, ~66M parameters), a classic encoder.
Why DistilBERT:
- It is about 40% smaller and 60% faster than BERT-base while keeping most of its accuracy.
- The service runs on a laptop **CPU** with a 5-second timeout per request and a 150-second global cap, so inference speed matters more than a few extra points of accuracy.
- Bigger encoders (RoBERTa-base, DeBERTa-v3-base) could be more accurate, but they are slower on CPU and take longer to fine-tune, and we had very limited time.
- Decoders (GPT-style) are not allowed and do not fit exact-span answers.

## Training setup, loss curve and validation results

**Training setup** (script: `training/train.py`, config saved in `training/logs/train_config.json`):

| Setting | Value |
|---|---|
| Base checkpoint | distilbert/distilbert-base-uncased |
| Training data | SQuAD 2.0 train, random subset of 50,000 questions (seed 42) |
| Training windows (features) | 50,528 |
| Epochs | 1 |
| Batch size | 16 |
| Learning rate | 3e-5 (linear decay, 10% warmup) |
| Weight decay | 0.01 |
| max_seq_length / doc_stride | 384 / 128 |
| Precision | fp16 |
| Hardware | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB) |
| Training time | ~6.3 minutes (3,158 steps) |

Unanswerable questions are trained to point at the `[CLS]` token (position 0). Long contexts are split into overlapping windows of 384 tokens with a stride of 128.

**Loss curve:** the training loss went down from 5.94 at the start to 1.52 at the end (average 1.90), so the model learned the task. Full log: `training/logs/log_history.json`.

**Validation results** (script: evaluation/choose_threshold.py, results saved in evaluation/threshold.json):

We evaluated on a random subset of 2000 questions from the SQuAD 2.0 validation set (seed 0), which the model never saw during training. About 48.95% of them are unanswerable.

| Metric | Value |
|---|---|
| Exact Match (tuned threshold) | 0.5865 |
| F1 (tuned threshold) | 0.6095 |
| Exact Match (threshold = 0) | 0.5590 |
| F1 (threshold = 0) | 0.5933 |
| Always-null baseline (Exact Match) | 0.4895 |

The model clearly beats the always-null baseline, and tuning the no-answer threshold improved Exact Match from 0.5590 to 0.5865.

![Loss curve](training/logs/loss_curve.png)

## How we chose the no-answer threshold

For every question, the model produces two scores:
- **null score**: start logit + end logit of the [CLS] token (the "no answer" option). For long contexts split into several windows, we take the minimum null score across windows.
- **best span score**: the highest start logit + end logit among valid spans inside the context (end after start, at most 30 tokens, searching the top 20 start and end positions).

We compute diff = null_score - best_span_score. If diff > threshold, the service answers `null`; otherwise it returns the best span.

To choose the threshold, we ran the model once on 2000 SQuAD 2.0 validation questions, stored each question's diff, and then tried every observed diff value as a candidate threshold. We kept the one with the highest Exact Match (F1 as a tie-breaker), where a correct `null` on an unanswerable question counts as a correct answer. This is the same way the instructor scores each item.

The chosen threshold is **-1.224** (Exact Match 0.5865, F1 0.6095), compared to Exact Match 0.5590 with the default threshold of 0. The value is saved in evaluation/threshold.json, and the API reads it at startup, so evaluation and serving use exactly the same decision rule (shared code in qa_core.py).

## Serving architecture and tunnel

<!-- Persona 3: architecture. Persona 4: tunnel (at the end of this section). -->

**Framework and server.** The service is a FastAPI app (`api/app.py`) served by uvicorn with **1 worker** on port 8000. One worker keeps a single copy of the model in RAM; a lock (`predict_lock`) makes the model answer one question at a time, which is enough because the instructor's harness sends one request at a time with a 5 s timeout.

**Model loading.** At startup (FastAPI `lifespan`), the app first checks that `QA_API_TOKEN` exists and refuses to start without it. Then it loads the tokenizer, the fine-tuned model from `models/qa-distilbert` (`QA_MODEL_DIR`) and the threshold from `evaluation/threshold.json` in a **background thread**, and runs one warm-up prediction. While this happens, `GET /health` returns `503 {"status": "loading"}`; it only returns `200 {"status": "ok"}` once the model can answer `/predict`. If loading fails, the error is printed in the console.

**Request flow for `POST /predict`:**
1. `check_token` compares the `Authorization` header with `"Bearer " + QA_API_TOKEN` exactly (case-sensitive, `secrets.compare_digest`). Missing or wrong token → `401 {"error": "unauthorized"}`.
2. The body is parsed as JSON and validated with a Pydantic model (`id` 1–64 chars, `context` 1–5000, `question` 1–500, strict string types). Invalid JSON or a missing/wrong field → `400 {"error": "bad_request", "detail": ...}`.
3. `qa_core.predict` tokenizes question + context. Contexts longer than 384 tokens are split into **overlapping windows** (`max_seq_length` 384, `doc_stride` 128), so long texts such as sample-05 are fully covered.
4. For every window it takes the top-20 start and end logits, keeps only valid spans inside the context (end ≥ start, at most 30 tokens) and keeps the best one overall. The no-answer score is the `[CLS]` score (minimum over windows).
5. If `null_score − best_span_score > −1.224` (our threshold), it returns `answer: null, start: null`. Otherwise it maps the tokens back to **character offsets** using the tokenizer's offset mapping, so `context[start:start+len(answer)] == answer` always holds.
6. The response is `{"id", "answer", "start", "score"}`, where `score` is a 0–1 confidence (sigmoid of the margin to the threshold). Any unexpected exception → `500 {"error": "internal_error"}`.

`GET /info` (protected) returns `team`, `model` (`distilbert/distilbert-base-uncased`), `params`, `max_seq_length` and `doc_stride`.

**Access control.** The token is checked inside `/info` and `/predict` (not in `/health`, which is public). Its value only comes from the environment variable `QA_API_TOKEN`; it is never written in the code or committed (`.env` is in `.gitignore`).

**How to run from a clean machine (Windows PowerShell, CPU only):**

```powershell
git clone https://github.com/DaniellaCampos/proyecto-qa.git
cd proyecto-qa
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch
pip install -r requirements.txt
Expand-Archive <zip-downloaded-from-Drive>.zip -DestinationPath models   # must give models/qa-distilbert/config.json
$env:QA_API_TOKEN = "<token>"
$env:QA_TEAM_NAME = "IA-Group"
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Then, in a second terminal (with the same `QA_API_TOKEN`): `python -m tests.check_api` (runs the 7 contract checks plus the public sample set; `public_sample.json` must be in the project folder).

**Tunnel (Persona 4).** The laptop has no public IP, so we expose the local API with **ngrok** (`ngrok http 8000`). ngrok opens an outbound connection from the laptop to ngrok's servers and gives a public HTTPS URL (for example `https://<name>.ngrok-free.dev`) that forwards every request to `localhost:8000`. Nothing about the API changes: the same token check, JSON validation and model run on the laptop. Backup if ngrok fails: `cloudflared tunnel --url http://localhost:8000` (Cloudflare quick tunnel, new URL each time it starts).

Because the URL is public, anyone who learns it could call the service. That is why `/info` and `/predict` require `Authorization: Bearer <token>`; only `/health` is open. `tests/check_api.py` sends the header `ngrok-skip-browser-warning` so ngrok's free-tier browser warning page never replaces a JSON response.

During evaluation the laptop stays plugged in with sleep disabled, with uvicorn and the tunnel running. If the tunnel restarts and the URL changes, we send the new URL to the instructor. On presentation day we set the instructor's new token in `QA_API_TOKEN` and restart uvicorn.

- Public URL used in our tests (Persona 4's laptop): `https://hallowedly-miasmatical-leda.ngrok-free.dev`. If the API runs on another laptop on presentation day, we send that laptop's ngrok URL to the instructor.
- Tunnel used: **ngrok** v3.39 (free plan, region United States), with `cloudflared` as backup.

## Results on the public sample set

<!-- Persona 4 -->

Measured through the public tunnel URL with `python -m tests.check_api <public-url>` (same harness as the local run, but every request goes through the internet and the tunnel).

| id | Our answer | Expected | Correct? | Latency (s) |
|---|---|---|---|---|
| sample-01 | `null` | "16" / "16 attention heads" | No | 1.11 |
| sample-02 | "3.5 days" (start 437) | "3.5 days" | Yes | 1.01 |
| sample-03 | `null` | `null` (unanswerable) | Yes | 1.08 |
| sample-04 | `null` | `null` (unanswerable) | Yes | 1.04 |
| sample-05 | "50,257" (start 3157) | "50,257" | Yes | 2.08 |

- Tests passed: **12/12** (the 7 contract checks plus the 5 sample items, all with a valid format: `context[start:start+len(answer)] == answer`, and `start` is null only when `answer` is null). An extra check with an unknown field in the body (`"foo": 1`) returns `400 bad_request` ("Extra inputs are not permitted").
- Exact Match on the public sample: **4/5** (same answers as the local run without the tunnel)
- Slowest request through the tunnel: **2.08 s** (sample-05, the long context with several windows). All requests are below the 5 s limit.

Through the tunnel every request takes about **1 s**, while the same requests on the same laptop take 0.16–0.22 s locally (1.38 s for sample-05; see *Second laptop* in the next section). The difference, about 0.7–0.9 s per request, is network time: each request goes from the client to ngrok's server in the United States and back to the laptop, and the test script opens a new HTTPS connection for each request. sample-05 is slower in both cases because its long context is split into several 384-token windows.

The only error is sample-01, explained in *Decoder investigation*: the best span (`"24 layers, a hidden size of 1024 and 16"`) was too long and its score fell just on the no-answer side of the threshold.

## Model size and self-measured latency

<!-- Persona 3 / Persona 4 -->

Measured locally (`http://127.0.0.1:8000`, no tunnel) on Persona 3's laptop, with the service started by `uvicorn api.app:app --port 8000`.

| Item | Value |
|---|---|
| Parameters (from `GET /info`) | 66,364,418 |
| Size on disk (`models/qa-distilbert`) | 254 MB (`model.safetensors` 265,470,032 bytes, fp32) |
| Model load time at startup | ~10 s until `/health` returns `ok` (includes warm-up prediction) |
| Mean latency per `/predict` request | **76 ms** (average over the 5 public sample items, 10 requests each) |
| Short contexts (sample-01 to sample-04, 1 window) | 33–39 ms |
| Long context (sample-05, >512 tokens, several windows) | 238 ms |
| Worst case: 5000-character context | 0.33 s |
| Laptop CPU and RAM | AMD Ryzen 7 7435HS, 32 GB RAM, CPU only (PyTorch 2.15 CPU build) |

Every request is far below the 5 s timeout, so the main risk on presentation day is the extra network latency of the tunnel, not the model. Latency grows with the number of 384-token windows, which is why the long-context item is about 6× slower than the short ones.

**API test results** (`python -m tests.check_api`, local): **12/12 OK**. `/health` without token → 200 `ok`; `/info` without token and with a wrong token → 401; `/info` with the valid token → 200 with `team`, `model`, `params`; `/predict` with the valid token → 200; `/predict` without token → 401; incomplete body → 400. Extra checks: invalid JSON, a non-string `id`, an empty `question` and a 5001-character `context` all return 400 `bad_request`; a lowercase `bearer` scheme returns 401 (exact comparison). All 5 public sample items returned a valid format (`context[start:start+len(answer)] == answer`, `start` is null only when `answer` is null). Four of the five answers are correct; sample-01 returned `null` instead of `16`. The best span the model found was `"24 layers, a hidden size of 1024 and 16"`, which is wrong and too long, and its score was only slightly worse than the no-answer score (diff −0.92 > threshold −1.224). We did not lower the threshold to fix one public item, because it was tuned on 2000 validation questions.

**Second laptop (Persona 4, CPU only, no GPU).** We repeated the same tests on a second, slower laptop, locally and through the ngrok tunnel (12/12 OK both times):

| Request | Local (`127.0.0.1`) | Through ngrok tunnel |
|---|---|---|
| Short contexts (sample-01 to sample-04) | 0.16–0.22 s | 1.01–1.11 s |
| Long context (sample-05) | 1.38 s | 2.08 s |

This laptop's CPU is about 5× slower than the first one for this model, and the tunnel adds about 0.7–0.9 s of network time per request. Plugging in the charger and using the best-performance power mode did not change the result. Everything stays below the 5 s limit, but the long context goes over our 2 s safety margin, so on presentation day the API and the tunnel run on the faster laptop, and the second laptop stays ready as a backup.

## Problems encountered and how we fixed them

<!-- Everyone adds their own -->

1. **Problem:** the API could not start on the CPU laptop because `models/qa-distilbert` did not exist: the team manual expected a file `qa-distilbert.zip`, but the trained model was shared on Google Drive as a folder of 5 files. **Cause:** the model is not in GitHub (it is ignored by `.gitignore`, as the rules require), so every member has to download it separately, and Drive serves a folder as an automatically generated zip with a different name (`qa-distilbert-<date>-1-001.zip`). **Fix:** download the whole Drive folder, extract it into `models/` (the zip already contains the `qa-distilbert/` folder) and check that `models/qa-distilbert/config.json` exists before running uvicorn. The "How to run" steps above describe this. **Commit:** `d227cd3`
2. **Problem:** in the threshold section of this report, the word `null` was shown broken across two lines as "ull". **Cause:** the text was written with `\n` before "ull", which some tools convert into a line break. **Fix:** we wrote it as inline code (`` `null` ``). **Commit:** `d227cd3`

3. **Problem:** `POST /predict` accepted a body with extra fields (for example `{"id", "context", "question", "foo"}`) and answered 200, but the contract says `additionalProperties: false`. **Cause:** Pydantic models ignore unknown fields by default. **Fix:** we set `ConfigDict(strict=True, extra="forbid")` in `PredictRequest`, so any extra field now returns `400 {"error": "bad_request"}` (verified: `"field 'foo': Extra inputs are not permitted"`). **Commit:** `d75295f`

## Decoder investigation

<!-- Persona 4 -->

Our model is an **extractive encoder**: it can only return a continuous span copied from the context, or `null`. A decoder (GPT-style) model **generates** the answer token by token. We did not use one (decoders are not allowed in the service), but we looked at where a decoder would behave differently on our errors.

**Errors where a decoder could help**

1. **Distractor spans and near-threshold decisions (public sample-01).** The answer was `16`, but the best span our model found was `"24 layers, a hidden size of 1024 and 16"`, and its score was so close to the no-answer score that the threshold turned it into `null`. The encoder scores start and end positions independently, so it can merge several numbers from one list into one wrong span. A decoder reads the whole question and context together and writes only the part that answers the question, so it would probably say "16".
2. **Answers that are not a span of the text.** Yes/no questions ("Is X larger than Y?"), counts ("How many...?" when the items are listed but the number is not written) or answers that combine two sentences cannot be extracted at all. A decoder can write "yes", "3" or a short synthesis.
3. **Long contexts.** We split long texts into 384-token windows; the answer can be cut at a window border, and evidence spread across windows is never seen together. Modern decoders have much longer context windows and see the whole text at once.

4. **Span boundary errors (validation set).** On the 2000 validation questions, F1 (0.6095) is higher than Exact Match (0.5865). The gap means that in several questions the model finds almost the right answer but adds or cuts tokens, such as an extra article, a unit or a neighbouring word. An extractive model only chooses start and end positions, so a small boundary mistake counts as a full EM error. A decoder would write a short answer with natural boundaries, but it could also paraphrase it, which EM punishes as well.

**Why a decoder is not automatically better for this task**

- **Exact Match penalises paraphrase.** A decoder may answer "sixteen" or "It has 16 heads" instead of the exact span `16`. That is wrong under EM, and the contract also requires `start` with `context[start:start+len(answer)] == answer`, which a free-form answer does not guarantee. We would need to force it to copy text or search the generated string back in the context.
- **Hallucination.** A decoder can answer from its general knowledge instead of the given context, which is exactly what SQuAD 2.0's unanswerable questions punish. Our encoder cannot invent text that is not in the context.
- **No calibrated "no answer".** We tune a numeric threshold on 2000 validation questions. A decoder decides to say "no answer" through the prompt, which is harder to measure and tune.
- **Cost on CPU.** Even a small decoder (1B+ parameters) is 15× or more the size of DistilBERT (66M) and generates token by token, which puts the 5 s limit per request at risk on a laptop CPU.

**Conclusion:** for SQuAD-style extractive QA with an exact-match metric and a CPU latency budget, a fine-tuned encoder is the better fit. A decoder would mainly help on non-span answers (yes/no, counting, synthesis), which are outside this task's contract.

## Known limitations

<!-- Persona 4 -->

- **Moderate accuracy.** Exact Match 0.5865 / F1 0.6095 on 2000 validation questions, because we trained only 1 epoch on 50,000 of the ~130,000 SQuAD 2.0 training questions. More epochs, the full dataset or a bigger encoder would score higher.
- **Only extractive answers.** The model can only return a continuous span of the context (max 30 tokens). Yes/no, counting or multi-sentence answers are impossible.
- **Single global threshold.** One no-answer threshold (−1.224) is used for every question. Questions close to the threshold (like public sample-01) can flip between `null` and a span.
- **English only.** DistilBERT-base-uncased and SQuAD 2.0 are English; questions in other languages will perform poorly.
- **Long contexts.** Texts are split into 384-token windows; evidence spread across windows is not combined, and latency grows with the number of windows. Context is capped at 5000 characters.
- **One request at a time.** One uvicorn worker plus a lock: concurrent requests wait in line. Fine for the instructor's sequential harness, not for real traffic.
- **Depends on the laptop and the tunnel.** If the laptop sleeps, loses power or internet, or the tunnel restarts with a new URL, the service is unavailable until we fix it. The free ngrok tier adds network latency.
- **Single shared token.** Access control is one static Bearer token, with no rate limiting and no per-user keys.
