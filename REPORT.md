# REPORT

> Si el profesor entrego REPORT_TEMPLATE.md, copien aqui sus encabezados exactos.
> Ninguna seccion puede quedar vacia: un informe incompleto cuenta como no entregado.

## Team

- Team name:
- Members:

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

**Validation results:** (Persona 2)

![Loss curve](training/logs/loss_curve.png)

## How we chose the no-answer threshold

<!-- Persona 2 -->

## Serving architecture and tunnel

<!-- Persona 3: architecture. Persona 4: tunnel. -->

## Results on the public sample set

<!-- Persona 4 -->

## Model size and self-measured latency

<!-- Persona 3 / Persona 4 -->

## Problems encountered and how we fixed them

<!-- Everyone adds their own -->

## Decoder investigation

<!-- Persona 4 -->

## Known limitations

<!-- Persona 4 -->
