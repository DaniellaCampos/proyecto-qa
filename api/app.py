"""Persona 3: la API que sigue el contrato del profesor.

Correr (desde la carpeta del proyecto):
    uvicorn api.app:app --host 0.0.0.0 --port 8000
Variables de entorno:
    QA_API_TOKEN  (obligatoria) token que da el profesor
    QA_TEAM_NAME  nombre del equipo registrado, sin espacios
    QA_MODEL_DIR  carpeta del modelo (default models/qa-distilbert)
"""
import json
import os
import secrets
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from qa_core import BASE_MODEL, DOC_STRIDE, MAX_SEQ_LENGTH, predict

MODEL_DIR = os.environ.get("QA_MODEL_DIR", "models/qa-distilbert")
TEAM_NAME = os.environ.get("QA_TEAM_NAME", "team-XX")
THRESHOLD_FILE = "evaluation/threshold.json"

state = {"model": None, "tokenizer": None, "threshold": 0.0, "params": 0, "error": None}
predict_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, status, code, detail=None):
        self.status, self.code, self.detail = status, code, detail


class PredictRequest(BaseModel):
    # extra="forbid": campos extra -> 400 (contrato: additionalProperties: false)
    model_config = ConfigDict(strict=True, extra="forbid")
    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


def load_model():
    """Se ejecuta en segundo plano: mientras tanto /health responde 'loading'."""
    try:
        torch.set_grad_enabled(False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).eval()
        if os.path.exists(THRESHOLD_FILE):
            with open(THRESHOLD_FILE) as f:
                state["threshold"] = float(json.load(f)["threshold"])
        # Primera prediccion de calentamiento para que la primera peticion real sea rapida.
        predict(model, tokenizer, "Warm up text.", "What?", state["threshold"])
        state["params"] = sum(p.numel() for p in model.parameters())
        state["tokenizer"], state["model"] = tokenizer, model
        print(f"Modelo cargado. Umbral = {state['threshold']}")
    except Exception as exc:  # se reporta en /health y en la consola
        state["error"] = repr(exc)
        print(f"ERROR cargando el modelo: {exc!r}")


@asynccontextmanager
async def lifespan(app):
    if not os.environ.get("QA_API_TOKEN"):
        raise RuntimeError("Falta la variable de entorno QA_API_TOKEN")
    threading.Thread(target=load_model, daemon=True).start()
    yield


app = FastAPI(title="QA API", lifespan=lifespan)


@app.exception_handler(ApiError)
async def api_error_handler(request, exc):
    body = {"error": exc.code}
    if exc.detail:
        body["detail"] = exc.detail
    return JSONResponse(status_code=exc.status, content=body)


@app.exception_handler(Exception)
async def unexpected_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "internal_error"})


def check_token(request):
    """Exige 'Authorization: Bearer <token>' exacto (distingue mayusculas)."""
    expected = "Bearer " + os.environ["QA_API_TOKEN"]
    received = request.headers.get("authorization", "")
    if not secrets.compare_digest(received.encode(), expected.encode()):
        raise ApiError(401, "unauthorized")


def run_prediction(context, question):
    with predict_lock:
        return predict(state["model"], state["tokenizer"], context, question, state["threshold"])


@app.get("/health")
def health():
    if state["model"] is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.get("/info")
def info(request: Request):
    check_token(request)
    return {
        "team": TEAM_NAME,
        "model": BASE_MODEL,
        "params": state["params"],
        "max_seq_length": MAX_SEQ_LENGTH,
        "doc_stride": DOC_STRIDE,
    }


@app.post("/predict")
async def predict_endpoint(request: Request):
    check_token(request)
    try:
        body = json.loads(await request.body())
    except (ValueError, UnicodeDecodeError):
        raise ApiError(400, "bad_request", "body must be valid JSON")
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", "body must be a JSON object")
    try:
        req = PredictRequest.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first["loc"]) or "body"
        raise ApiError(400, "bad_request", f"field '{field}': {first['msg']}")

    if state["model"] is None:
        raise ApiError(500, "internal_error", "model not loaded yet")
    answer, start, score = await run_in_threadpool(run_prediction, req.context, req.question)
    return {"id": req.id, "answer": answer, "start": start, "score": score}
