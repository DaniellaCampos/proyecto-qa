"""Persona 1: fine-tuning de DistilBERT en SQuAD 2.0.

Uso (desde la carpeta del proyecto):
    python -m training.train
Variables opcionales:
    TRAIN_SAMPLES (default 50000)  cuantas preguntas de SQuAD usar
    EPOCHS        (default 1)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from qa_core import BASE_MODEL, DOC_STRIDE, MAX_SEQ_LENGTH

TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", 50000))
EPOCHS = float(os.environ.get("EPOCHS", 1))
OUTPUT_DIR = "models/qa-distilbert"
LOG_DIR = "training/logs"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def prepare_train_features(examples):
    """Convierte texto + respuesta en posiciones de tokens (inicio y fin)."""
    enc = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets_all = enc.pop("offset_mapping")
    starts, ends = [], []
    for i, offsets in enumerate(offsets_all):
        answers = examples["answers"][sample_map[i]]
        # Pregunta sin respuesta -> apuntar al token [CLS] (posicion 0).
        if len(answers["answer_start"]) == 0:
            starts.append(0)
            ends.append(0)
            continue
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        seq_ids = enc.sequence_ids(i)
        ctx_start = seq_ids.index(1)
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)
        # La respuesta no cae en esta ventana -> tambien cuenta como "sin respuesta".
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            starts.append(0)
            ends.append(0)
            continue
        idx = ctx_start
        while idx <= ctx_end and offsets[idx][0] <= start_char:
            idx += 1
        starts.append(idx - 1)
        idx = ctx_end
        while idx >= ctx_start and offsets[idx][1] >= end_char:
            idx -= 1
        ends.append(idx + 1)
    enc["start_positions"] = starts
    enc["end_positions"] = ends
    return enc


def save_loss_curve(log_history):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(f"{LOG_DIR}/log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)
    points = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    plt.figure(figsize=(7, 4))
    plt.plot([p[0] for p in points], [p[1] for p in points])
    plt.xlabel("Paso de entrenamiento")
    plt.ylabel("Loss")
    plt.title(f"Fine-tuning {BASE_MODEL} en SQuAD 2.0")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{LOG_DIR}/loss_curve.png", dpi=150)


def main():
    print(f"GPU disponible: {torch.cuda.is_available()}")
    raw = load_dataset("rajpurkar/squad_v2", split="train").shuffle(seed=42)
    raw = raw.select(range(min(TRAIN_SAMPLES, len(raw))))
    print(f"Preguntas de entrenamiento: {len(raw)}")
    train_ds = raw.map(prepare_train_features, batched=True, remove_columns=raw.column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL)
    args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=16,
        learning_rate=3e-5,
        num_train_epochs=EPOCHS,
        warmup_steps=0.1,  # 10% de los pasos
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    save_loss_curve(trainer.state.log_history)
    with open(f"{LOG_DIR}/train_config.json", "w") as f:
        json.dump(
            {
                "base_model": BASE_MODEL,
                "train_samples": len(raw),
                "train_features": len(train_ds),
                "epochs": EPOCHS,
                "batch_size": 16,
                "learning_rate": 3e-5,
                "max_seq_length": MAX_SEQ_LENGTH,
                "doc_stride": DOC_STRIDE,
            },
            f,
            indent=2,
        )
    print(f"Listo. Modelo guardado en {OUTPUT_DIR}, curva en {LOG_DIR}/loss_curve.png")


if __name__ == "__main__":
    main()
