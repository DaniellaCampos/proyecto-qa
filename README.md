# Extractive QA Service — IA-Group

Fine-tuned **DistilBERT** for extractive question answering with unanswerable questions (SQuAD 2.0 style), served as an **access-controlled REST API** that runs on a laptop CPU and is exposed through a public HTTPS tunnel.

Given a context paragraph and a question, the service returns the answer as an exact span of the context, or `null` when the context does not contain the answer.

> Course project — Ingeniería de la IA · Servicios NLP expuestos por medio de API

---

## Team

| Member | Role |
|---|---|
| Daniella Campos | Model training |
| Rebeca Michelle Castro Pineda | Evaluation and no-answer threshold |
| Rodrigo Marroquin | API and tests |
| Aracely Verenice Varela | Tunnel, final report and submission |

---

## Results

| Metric | Value |
|---|---|
| Base checkpoint | `distilbert/distilbert-base-uncased` (66M parameters) |
| Training data | 50,000 SQuAD 2.0 questions, 1 epoch (~6 min on an RTX 5060 Laptop GPU) |
| Training loss | 5.94 → 1.52 |
| Exact Match (SQuAD 2.0 validation, 2,000 questions) | 0.587 |
| F1 | 0.610 |
| Always-null baseline (Exact Match) | 0.490 |
| Mean latency per request (CPU) | ~76 ms |

Full details are in [REPORT.md](REPORT.md).

---

## Repository structure

```
├── training/
│   ├── train.py              # Fine-tuning on SQuAD 2.0
│   └── logs/                 # Training logs, config and loss curve
├── evaluation/
│   ├── choose_threshold.py   # Validation metrics and no-answer threshold
│   └── threshold.json        # Chosen threshold (read by the API)
├── api/
│   └── app.py                # FastAPI service (/health, /info, /predict)
├── tests/
│   └── check_api.py          # Automated checks against the API contract
├── qa_core.py                # Shared inference logic (API + evaluation)
├── requirements.txt
└── REPORT.md
```

---

## How it works

1. **Training** — `distilbert-base-uncased` is fine-tuned on SQuAD 2.0. Unanswerable questions point to the `[CLS]` token. Long contexts are split into 384-token windows with a 128-token stride.
2. **No-answer threshold** — for each question we compare the "no answer" score with the best span score. The threshold that maximizes Exact Match on validation data is saved in `evaluation/threshold.json`.
3. **Serving** — FastAPI loads the model in the background (`/health` returns `loading` until it is ready), checks the bearer token, validates the JSON body, runs the model on CPU and returns `id`, `answer`, `start` and `score`.
4. **Tunnel** — ngrok exposes `localhost:8000` through a public HTTPS URL so the instructor's harness can reach the laptop.

---

## Setup

Requirements: Python 3.12+, Git. An NVIDIA GPU is only needed for training.

```powershell
git clone https://github.com/DaniellaCampos/proyecto-qa.git
cd proyecto-qa
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch:

```powershell
# With an NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch
```

Install the rest:

```powershell
pip install -r requirements.txt
```

### Model weights

Model weights are **not** stored in Git. Download `qa-distilbert.zip` from the team's shared folder and extract it so that `models/qa-distilbert/config.json` exists:

```powershell
Expand-Archive qa-distilbert.zip -DestinationPath models
```

---

## Usage

All commands run from the project root.

### Train (GPU)

```powershell
python -m training.train
```

Optional: `$env:TRAIN_SAMPLES = "20000"` to train on fewer questions.

### Choose the no-answer threshold

```powershell
python -m evaluation.choose_threshold
```

Optional: `$env:EVAL_SAMPLES = "500"` on CPU-only machines.

### Run the API

```powershell
$env:QA_API_TOKEN = '<token>'
$env:QA_TEAM_NAME = 'IA-Group'
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Wait until the console prints `Modelo cargado`.

### Test the API

In a second terminal (with the same token):

```powershell
$env:QA_API_TOKEN = '<token>'
python -m tests.check_api                                  # local
python -m tests.check_api https://<your-url>.ngrok-free.dev # through the tunnel
```

### Expose it with a tunnel

```powershell
ngrok http 8000
```

Backup: `cloudflared tunnel --url http://localhost:8000`

---

## API

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | Public | `200 {"status": "ok"}` when ready, `503 {"status": "loading"}` while loading |
| `/info` | GET | Bearer token | Team name, base model, parameters, `max_seq_length`, `doc_stride` |
| `/predict` | POST | Bearer token | Answers a question about a context |

Example request:

```json
{
  "id": "q-017",
  "context": "The Lempa River is the longest river in El Salvador, flowing about 422 km from Guatemala to the Pacific Ocean.",
  "question": "How long is the Lempa River?"
}
```

Example response:

```json
{"id": "q-017", "answer": "422 km", "start": 67, "score": 0.74}
```

`answer` is always copied verbatim from the context (`context[start:start+len(answer)] == answer`), or `null` with `start: null` when there is no answer.

Errors use `{"error": "<code>", "detail": "<optional>"}` with codes `unauthorized` (401), `bad_request` (400) and `internal_error` (500).

---

## Rules we follow

- The access token is read from `QA_API_TOKEN` and is **never** written in the code or committed.
- Model weights are excluded from Git (`.gitignore`).
- No commercial API calls: the model runs locally.
- Git history is preserved (no squashing, no force pushes).
