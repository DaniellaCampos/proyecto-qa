"""Persona 2: medir el modelo y elegir el umbral de "no respuesta".

Uso (desde la carpeta del proyecto):
    python -m evaluation.choose_threshold
Variables opcionales:
    EVAL_SAMPLES (default 2000)  cuantas preguntas de validacion usar
"""
import collections
import json
import os
import re
import string

import torch
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from qa_core import score_example

MODEL_DIR = os.environ.get("QA_MODEL_DIR", "models/qa-distilbert")
EVAL_SAMPLES = int(os.environ.get("EVAL_SAMPLES", 2000))
OUT_FILE = "evaluation/threshold.json"


def normalize(text):
    """Normalizacion oficial de SQuAD: minusculas, sin puntuacion ni articulos."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def item_scores(pred, golds):
    """(exact, f1) de una prediccion; golds vacio = pregunta sin respuesta."""
    if not golds:
        ok = float(pred is None)
        return ok, ok
    if pred is None:
        return 0.0, 0.0
    return (
        max(float(normalize(pred) == normalize(g)) for g in golds),
        max(f1(pred, g) for g in golds),
    )


def evaluate_at(rows, threshold):
    exact = f1_sum = 0.0
    for text, diff, golds in rows:
        pred = None if text is None or diff > threshold else text
        e, f = item_scores(pred, golds)
        exact += e
        f1_sum += f
    return exact / len(rows), f1_sum / len(rows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).to(device).eval()

    val = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(seed=0)
    val = val.select(range(min(EVAL_SAMPLES, len(val))))
    print(f"Evaluando {len(val)} preguntas en {device}...")

    rows = []
    for i, ex in enumerate(val):
        text, _, diff = score_example(model, tokenizer, ex["context"], ex["question"], device)
        rows.append((text, diff, ex["answers"]["text"]))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(val)}")

    # Probar como umbral cada diferencia observada y quedarse con el mejor Exact Match.
    candidates = sorted({0.0} | {round(d, 3) for _, d, _ in rows if d != float("inf")})
    results = [(t, *evaluate_at(rows, t)) for t in candidates]
    best_t, best_em, best_f1 = max(results, key=lambda r: (r[1], r[2]))
    em0, f10 = evaluate_at(rows, 0.0)
    null_em, null_f1 = evaluate_at([(None, 0, g) for _, _, g in rows], 0.0)

    summary = {
        "threshold": best_t,
        "eval_samples": len(rows),
        "exact_match": round(best_em, 4),
        "f1": round(best_f1, 4),
        "exact_match_threshold_0": round(em0, 4),
        "f1_threshold_0": round(f10, 4),
        "always_null_exact_match": round(null_em, 4),
        "unanswerable_ratio": round(sum(1 for *_, g in rows if not g) / len(rows), 4),
    }
    with open(OUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Guardado en {OUT_FILE}")


if __name__ == "__main__":
    main()
