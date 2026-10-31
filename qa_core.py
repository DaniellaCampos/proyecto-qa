"""Logica compartida de prediccion.

La usan evaluation/choose_threshold.py (Persona 2) y api/app.py (Persona 3),
asi el umbral que se elige es exactamente el que usa la API.
"""
import math

import torch

BASE_MODEL = "distilbert/distilbert-base-uncased"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 30
N_BEST = 20


def score_example(model, tokenizer, context, question, device="cpu"):
    """Devuelve el mejor fragmento y la diferencia de puntaje contra 'sin respuesta'.

    Resultado: (texto, start, score_diff), donde score_diff = null_score - best_span_score.
    Si score_diff > umbral, la respuesta debe ser null.
    """
    # Los textos largos se parten en ventanas de 384 tokens que se solapan 128.
    enc = tokenizer(
        question.strip(),
        context,
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping").tolist()
    enc.pop("overflow_to_sample_mapping", None)
    inputs = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    with torch.inference_mode():
        out = model(**inputs)
    start_logits = out.start_logits.float().cpu()
    end_logits = out.end_logits.float().cpu()

    null_score = math.inf
    best = (None, None, -math.inf)  # (texto, start_char, puntaje)
    for w in range(start_logits.shape[0]):
        seq_ids = enc.sequence_ids(w)
        s_log, e_log = start_logits[w], end_logits[w]
        # Puntaje de "no hay respuesta" = token [CLS] en la posicion 0.
        null_score = min(null_score, (s_log[0] + e_log[0]).item())

        starts = torch.topk(s_log, min(N_BEST, len(s_log))).indices.tolist()
        ends = torch.topk(e_log, min(N_BEST, len(e_log))).indices.tolist()
        for s in starts:
            for e in ends:
                # Solo fragmentos dentro del contexto, en orden y no muy largos.
                if seq_ids[s] != 1 or seq_ids[e] != 1:
                    continue
                if e < s or e - s + 1 > MAX_ANSWER_TOKENS:
                    continue
                score = (s_log[s] + e_log[e]).item()
                if score > best[2]:
                    start_char, end_char = offsets[w][s][0], offsets[w][e][1]
                    best = (context[start_char:end_char], start_char, score)

    if best[0] is None:
        return None, None, math.inf
    return best[0], best[1], null_score - best[2]


def predict(model, tokenizer, context, question, threshold, device="cpu"):
    """Aplica el umbral y devuelve (answer, start, score) listo para la API."""
    text, start, diff = score_example(model, tokenizer, context, question, device)
    if text is None or diff > threshold or not text.strip():
        confidence = 1 / (1 + math.exp(-min(diff - threshold, 50))) if text else 1.0
        return None, None, round(confidence, 4)

    # Quitar espacios de las orillas sin romper la regla context[start:start+len] == answer.
    stripped = text.lstrip()
    start += len(text) - len(stripped)
    text = stripped.rstrip()
    confidence = 1 / (1 + math.exp(-min(threshold - diff, 50)))
    return text, start, round(confidence, 4)
