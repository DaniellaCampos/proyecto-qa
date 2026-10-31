"""Persona 4: revisa la API igual que lo hara el profesor.

Uso (desde la carpeta del proyecto, con QA_API_TOKEN definido):
    python -m tests.check_api http://127.0.0.1:8000
    python -m tests.check_api https://<tu-url>.ngrok-free.app
Opcional: segundo argumento con la ruta de public_sample.json (default public_sample.json).
"""
import json
import os
import sys
import time

import requests

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else "public_sample.json"
TOKEN = os.environ.get("QA_API_TOKEN", "")
GOOD = {"Authorization": f"Bearer {TOKEN}"}
HEADERS = {"ngrok-skip-browser-warning": "1"}
results = []


def check(name, ok, extra=""):
    results.append(ok)
    print(f"[{'OK ' if ok else 'MAL'}] {name} {extra}")


def get(path, headers=None):
    return requests.get(BASE + path, headers={**HEADERS, **(headers or {})}, timeout=10)


def post_predict(item, headers=GOOD):
    body = {k: item[k] for k in ("id", "context", "question")}
    t0 = time.time()
    r = requests.post(BASE + "/predict", json=body, headers={**HEADERS, **headers}, timeout=10)
    return r, time.time() - t0


def reference_answers(item):
    """Acepta varios formatos posibles de respuestas de referencia."""
    for key in ("answers", "reference_answers", "references", "answer"):
        if key in item:
            refs = item[key]
            break
    else:
        return None
    if refs is None:
        return []
    if isinstance(refs, dict):
        refs = refs.get("text", [])
    if not isinstance(refs, list):
        refs = [refs]
    return [r["text"] if isinstance(r, dict) else r for r in refs if r is not None]


def main():
    if not TOKEN:
        sys.exit("Falta QA_API_TOKEN en esta terminal.")
    print(f"Probando {BASE}\n")

    r = get("/health")
    check("1. /health sin token -> 200 ok", r.status_code == 200 and r.json().get("status") == "ok", r.text)
    r = get("/info")
    check("2. /info sin token -> 401", r.status_code == 401, r.text)
    r = get("/info", {"Authorization": "Bearer wrong"})
    check("3. /info token malo -> 401", r.status_code == 401, r.text)
    r = get("/info", GOOD)
    info = r.json() if r.status_code == 200 else {}
    check("4. /info token bueno -> 200", r.status_code == 200 and {"team", "model", "params"} <= info.keys(), r.text)

    demo = {
        "id": "q-017",
        "context": "The Lempa River is the longest river in El Salvador, flowing about 422 km "
        "from Guatemala to the Pacific Ocean.",
        "question": "How long is the Lempa River?",
    }
    r, secs = post_predict(demo)
    check("5. /predict token bueno -> 200", r.status_code == 200, f"{r.text} ({secs:.2f}s)")
    r, _ = post_predict(demo, headers={})
    check("6. /predict sin token -> 401", r.status_code == 401, r.text)
    r = requests.post(BASE + "/predict", json={"id": "x"}, headers={**HEADERS, **GOOD}, timeout=10)
    check("7. /predict cuerpo incompleto -> 400/422", r.status_code in (400, 422), r.text)

    if os.path.exists(SAMPLE):
        with open(SAMPLE, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
        print(f"\nMuestra publica ({len(items)} preguntas):")
        for item in items:
            r, secs = post_predict(item)
            if r.status_code != 200:
                check(f"  {item['id']}", False, f"HTTP {r.status_code} {r.text}")
                continue
            out = r.json()
            answer, start = out.get("answer"), out.get("start")
            valid = out.get("id") == item["id"] and "score" in out and (answer is None) == (start is None)
            if answer is not None:
                valid = valid and item["context"][start:start + len(answer)] == answer
            refs = reference_answers(item)
            if refs is None:
                verdict = "sin referencia"
            elif not refs:
                verdict = "correcto (null)" if answer is None else "deberia ser null"
            else:
                verdict = "coincide" if answer and answer.lower() in [x.lower() for x in refs] else f"esperado {refs}"
            check(f"  {item['id']} formato valido y < 5s", valid and secs < 5,
                  f"-> {answer!r} | {verdict} | {secs:.2f}s")
    else:
        print(f"\n(No encontre {SAMPLE}; ponlo en la carpeta del proyecto para probarlo.)")

    print(f"\nResultado: {sum(results)}/{len(results)} pruebas OK")


if __name__ == "__main__":
    main()
