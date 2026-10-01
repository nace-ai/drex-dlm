"""Score a System One request with Drex DLM.

    pip install -r requirements.txt
    python inference.py
    python inference.py --request examples/request.json

The model returns a probability for every option in one forward pass.
A state or question past the context limit returns an error, so the score covers the whole document.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from kev.api import SystemOneRequest, output_tokens, to_answers, to_record  # noqa: E402
from kev.checkpoint import load  # noqa: E402
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE  # noqa: E402

EXAMPLE = {
    "state": {
        "ticket": "I was charged twice for the same order. Please refund the extra payment.",
    },
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, charges, and refunds",
                "technical": "Bugs and outages",
                "other": "Anything else",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for a refund?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["Routine", "Soon", "Urgent"],
        },
    },
}


def decide(tok, model, request):
    rec, meta = to_record(SystemOneRequest(state=request["state"], questions=request["questions"]))
    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
    started = time.perf_counter()
    with torch.no_grad():
        probs = [p.float().tolist() for p in model.probs(enc)]
    answers = to_answers(probs, meta)
    return {
        "answers": answers,
        "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(tok, answers)},
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Score one System One request with Drex DLM.")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent), help="Local checkpoint directory")
    parser.add_argument("--request", help="JSON file with one System One request")
    args = parser.parse_args()
    tok, model = load(args.model)
    request = json.loads(Path(args.request).read_text()) if args.request else EXAMPLE
    print(json.dumps(decide(tok, model, request), indent=2))


if __name__ == "__main__":
    main()
