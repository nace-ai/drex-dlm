"""Check the release's one-decision contract against one ready runner.

Run separately for Python, native and Ollama. Compare the resulting JSON reports
rather than claiming exact BF16/GGUF probability equality.
"""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/request.json"


def post(url, payload, timeout=240):
    request = urllib.request.Request(
        url + "/v1/systemone", json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def check_answer(value):
    if not isinstance(value, dict):
        raise AssertionError(f"missing answer: {value!r}")
    if value.get("type") == "noul":
        numbers = [value.get("noul")]
    else:
        numbers = list(value.get("probabilities", {}).values())
        if not numbers or not isinstance(value.get("confidence"), (int, float)):
            raise AssertionError(f"incomplete probabilities: {value!r}")
        numbers.append(value["confidence"])
        if value.get("type") == "score":
            numbers.append(value.get("score"))
    if not all(isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) for p in numbers):
        raise AssertionError(f"non-finite answer: {value!r}")
    bounded = numbers[:-1] if value.get("type") == "score" else numbers
    if not all(0 <= p <= 1 for p in bounded):
        raise AssertionError(f"invalid probability: {value!r}")


def score(url, name, payload):
    started = time.monotonic()
    status, body = post(url, payload)
    if status != 200:
        raise AssertionError(f"{name}: HTTP {status}: {str(body)[:400]}")
    expected = set(payload["questions"])
    answers = body.get("answers", {})
    if set(answers) != expected:
        raise AssertionError(f"{name}: answer keys {sorted(answers)} != {sorted(expected)}")
    for value in answers.values():
        check_answer(value)
    usage = body.get("usage", {})
    if not isinstance(usage.get("input_tokens"), int) or usage["input_tokens"] <= 0:
        raise AssertionError(f"{name}: invalid input usage: {usage}")
    if not isinstance(usage.get("output_tokens"), int) or usage["output_tokens"] <= 0:
        raise AssertionError(f"{name}: invalid output usage: {usage}")
    print(f"{name}: {usage['input_tokens']} tokens, {time.monotonic() - started:.1f}s", flush=True)
    return {"answers": answers, "usage": usage}


def long_request(position):
    # Stable filler and a single marker near one of three locations in a long state.
    # Input length is recorded from the actual tokenizer, not estimated here.
    chunk = "Record 4827 is routine customer-support background information. "
    blocks = [chunk * 400 for _ in range(3)]
    blocks[position] += " Unique reference: the support code for this case is COBALT. "
    return {"model": "drex-dlm", "state": "\n".join(blocks), "questions": {
        "code": {"type": "choice", "instructions": "Which support code appears in the case?",
                 "criteria": {"COBALT": "code word", "AMBER": "code word"}}
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long", action="store_true", help="Run repeated long retrieval cases")
    args = parser.parse_args()
    health_path = "/api/version" if args.runner == "ollama" else "/health"
    with urllib.request.urlopen(args.url + health_path, timeout=5) as health:
        body = json.load(health)
        if health.status != 200 or (args.runner != "ollama" and body.get("status") != "ok"):
            raise AssertionError("runner is not ready")
    example = json.loads(EXAMPLE.read_text())
    report = {"runner": args.runner, "cases": {}, "rejections": {}}
    report["cases"]["sample"] = score(args.url, "sample", example)
    if report["cases"]["sample"]["answers"]["team"].get("choice") != "billing":
        raise AssertionError("sample did not select billing")
    options = {str(i): str(i) for i in range(255)}
    boundary = {"model": "drex-dlm", "state": "Choose option 9.", "questions": {
        "q": {"type": "choice", "criteria": options}
    }}
    report["cases"]["255 options"] = score(args.url, "255 options", boundary)
    if len(report["cases"]["255 options"]["answers"]["q"]["probabilities"]) != 255:
        raise AssertionError("255-option boundary returned too few options")
    invalid = {
        "256 options": {**boundary, "questions": {"q": {"type": "choice", "criteria": {**options, "255": "255"}}}},
        "empty questions": {"model": "drex-dlm", "state": "x", "questions": {}},
        "mixed batch": {**example, "requests": [example]},
        "overlong state": {"model": "drex-dlm", "state": "Record 4827 is routine customer-support background information. " * 3000,
                           "questions": {"q": {"type": "noul"}}},
    }
    for name, payload in invalid.items():
        status, body = post(args.url, payload)
        report["rejections"][name] = status
        print(f"{name}: HTTP {status}", flush=True)
        if status != 400:
            raise AssertionError(f"{name}: expected HTTP 400, got {status}: {str(body)[:400]}")
    if args.runner != "python":
        status, _ = post(args.url, {"model": "drex-dlm", "requests": [example]})
        report["rejections"]["unsupported batch"] = status
        if status != 400:
            raise AssertionError(f"unsupported batch: expected HTTP 400, got {status}")
    if args.long:
        for position in (0, 2):
            name = f"long marker {position}"
            report["cases"][name] = score(args.url, name, long_request(position))
            selected = report["cases"][name]["answers"]["code"].get("choice")
            report["cases"][name]["retrieval_correct"] = selected == "COBALT"
            print(f"{name}: selected {selected}, expected COBALT", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
