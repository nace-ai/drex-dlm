---
license: mit
base_model: nvidia/Efficient-DLM-8B
tags:
  - decision-model
  - system-one
library_name: transformers
pipeline_tag: text-classification
---

# Drex DLM

We introduce Drex DLM from [Nace.AI](https://www.nace.ai/), a decision model that answers typed questions about a given context. Pass that context as `state`, provide one or more named questions, and get a probability for every option. One forward pass of an 8B diffusion language model plus a pointer head produces that distribution.

The backbone is [NVIDIA Efficient-DLM-8B](https://huggingface.co/nvidia/Efficient-DLM-8B), a diffusion language model with a decision adapter merged into its weights. The block tensor names match Qwen3. The shared context (`state`) uses bidirectional attention. Each question branch attends to that context and uses causal attention within the branch; state tokens cannot attend to questions, and question branches cannot attend to one another.

A shared pointer head projects the final-layer hidden state at the decision marker into a query and the final-layer hidden state at each option-ending marker into a key. Scaled dot products, temperature scaling, and a softmax over each question's options produce the probabilities.

![Drex DLM architecture: packed context and questions pass through the Efficient-DLM-8B diffusion backbone. Final hidden states are grouped by question. A shared pointer readout projects option-ending states into keys and decision states into queries, then scores and normalizes each question's options into probabilities.](assets/drex-dlm-architecture.png)

The diagram uses `<decide>` and `</opt>` as readable aliases for the decision and option-ending markers. Packed requests within the token budget use one forward pass; larger requests are split across question rows.

Weights: [nace-ai/drex-dlm](https://huggingface.co/nace-ai/drex-dlm)

Code: [nace-ai/drex-dlm](https://github.com/nace-ai/drex-dlm)

Server: [nace-ai/llama.cpp](https://github.com/nace-ai/llama.cpp), branch `edlm`

Context length is 32,768 tokens. The recommended default is 16,384. [Context length](#context-length) shows how to use the full window.

Python 3.12 is the tested interpreter. The first load is about 16 GB of bfloat16 weights. A 24 GB GPU is a comfortable fit. CPU runs it too, more slowly.

## Decision Index 0.2

Scores reported as of October 2026.

| Model | Index | Knowledge & Reasoning | Language Understanding | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Drex DLM | 52.31 | 50.71 | 57.77 | 54.25 | 48.59 | 50.25 |
| Decider chat · Gemma-4-31B | 51.93 | 44.54 | 58.91 | 50.34 | 66.99 | 38.89 |
| Jev | 51.67 | 50.54 | 59.74 | 43.35 | 66.86 | 37.86 |
| AutoJev-27B | 50.94 | 40.93 | 61.90 | 42.00 | 69.98 | 39.88 |
| Jebadiah 27B | 50.32 | 38.82 | 58.97 | 45.48 | 69.30 | 39.02 |
| simple-jev · Qwen3.8-27B | 50.21 | 36.61 | 60.16 | 50.22 | 66.95 | 37.10 |

The other five are the top of the [Decision Index 0.2](https://huggingface.co/spaces/multimodalart/jev-decision-index) board.

## Quick start

```bash
git clone https://github.com/nace-ai/drex-dlm.git
cd drex-dlm
hf download nace-ai/drex-dlm --local-dir .
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference.py --request examples/request.json
```

That prints the answers for the sample ticket. To run the inference server instead:

```bash
python serve.py --port 8000
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

`GET /health` returns `{"status": "ok"}` once the weights are loaded. The same request also runs through `llama-server` and Ollama (see [Serving](#serving)).

## Request format

Two fields matter:

| Field | What to put there |
|---|---|
| `state` | The document: a string, object, or list. Nested objects render as indented text. |
| `questions` | A dictionary of named questions. The names come back as the keys of `answers`. |

```json
{
  "state": {
    "ticket": "I was charged twice for the same order. Please refund the extra payment."
  },
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, charges, and refunds",
        "technical": "Bugs and outages",
        "other": "Anything else"
      }
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the customer explicitly ask for a refund?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["Routine", "Soon", "Urgent"]
    }
  }
}
```

| Type | You supply | You get back |
|---|---|---|
| `choice` | Named options, each with a short description. Object key order is option order. | `choice`, a probability per name, and `confidence` |
| `noul` | A yes-or-no question, with optional `criteria.true` / `criteria.false` descriptions. | `noul`, the probability of "yes," from 0 to 1 |
| `score` | An ordered scale, lowest first. | A probability-weighted `score`, plus `legend`, per-level `probabilities`, and `confidence` |

`confidence` measures how far the distribution sits from a tie — high means one option carries most of the probability.

A local bfloat16 run of `examples/request.json` returns:

```json
{
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9461,
      "probabilities": { "billing": 0.9641, "technical": 0.001, "other": 0.0349 }
    },
    "refund": { "type": "noul", "noul": 0.8548 },
    "urgency": {
      "type": "score",
      "score": 1.4078,
      "legend": { "0": "Routine", "1": "Soon", "2": "Urgent" },
      "probabilities": { "0": 0.1876, "1": 0.2169, "2": 0.5955 },
      "confidence": 0.7039
    }
  }
}
```

`usage.input_tokens` is the encoded document plus questions (87 for this ticket). `usage.output_tokens` counts the serialized answers, not generated text. `latency_ms` is the scoring time.

**Batching:** send `requests`, a list of `{state, questions}` objects, to score several decisions in one forward pass. The response is `results` in the same order, each with its own `answers`, `usage`, and `latency_ms`. That `latency_ms` is the shared forward for the group, not a separate score for the item. A single object (no `requests` wrapper) always scores one decision.

## Serving

| Runner | Weights | Listens on |
|---|---|---|
| Python | this directory, including `head.pt` | `POST /v1/systemone`, port 8000 |
| llama-server | `drex-dlm.gguf` | `POST /v1/systemone`, port 8097 |
| Ollama | the same GGUF | `POST /v1/systemone`, port 11434 |

Each runner scores up to the context length it was started with; longer requests fall back to scoring one question at a time, with identical probabilities either way. See [Context length](#context-length) for sizing details.

### Python

```bash
pip install -r requirements.txt
python serve.py --host 127.0.0.1 --port 8000
```

The server loads the safetensors shards and `head.pt` from this directory. `--model` points at another checkout of the same files; `--name` sets the `model` field in the response.

### llama-server

The `edlm` architecture, its converter, and `POST /v1/systemone` live on branch `edlm` of [nace-ai/llama.cpp](https://github.com/nace-ai/llama.cpp).

```bash
git clone --branch edlm https://github.com/nace-ai/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --target llama-server -j
```

For NVIDIA, add `-DGGML_CUDA=ON` to the `cmake -B build` step.

Convert this directory with that checkout — the converter copies the pointer head into the GGUF:

```bash
python convert_hf_to_gguf.py /path/to/drex-dlm \
  --outfile /path/to/drex-dlm/drex-dlm.gguf \
  --outtype f16
```

Start the server:

```bash
./build/bin/llama-server \
  -m /path/to/drex-dlm/drex-dlm.gguf \
  --host 127.0.0.1 --port 8097 \
  --embedding --pooling none \
  -c 16384 -b 16384 -ub 16384 -np 1 \
  --no-warmup
curl http://127.0.0.1:8097/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

The response matches Python's `answers` fields, plus `latency_ms` and an `x-typesafe-request-id` header. If the server is started with a smaller `-c`/`-b`/`-ub` than the batch it's asked to score, the request is rejected rather than split.

### Ollama

[nace-ai/ollama](https://github.com/nace-ai/ollama) branch `nace-edlm` launches that same `llama-server` and forwards `POST /v1/systemone`. Build the server from [nace-ai/llama.cpp](https://github.com/nace-ai/llama.cpp) branch `edlm` first:

```bash
export OLLAMA_LLAMA_CPP_SOURCE=/path/to/llama.cpp
cmake -S llama/server --preset darwin      # Apple Silicon
cmake --build build/llama-server-darwin --target llama-server --parallel 8
```

Use `llama_cuda_v12_linux` / `build/llama-server-cuda_v12` for NVIDIA Linux, or `cpu` / `build/llama-server-cpu` for CPU. Set `OLLAMA_LLAMA_CPP_SOURCE` before `cmake -S`. Then create the model from `drex-dlm.gguf` and post the same JSON to `http://127.0.0.1:11434/v1/systemone`.

## Context length

The weights support up to **32,768 tokens**; the recommended default is **16,384**. A `state`, and any one question together with it, must fit inside whichever context you select — several questions share one forward pass up to that limit, and the server falls back to scoring one question at a time beyond it. Requests above 32,768 are ignored and the 16,384 default stays in force. Each question may be `choice`, `noul`, or `score`, with up to 255 options; option text is the option name plus its description. A `<|name|>` span in user text is rewritten to `<¦name¦>` before tokenization, keeping the document separate from the five delimiter tokens.

Python already uses the recommended default. To use the full window instead:

```bash
KEV_CONTEXT=32768 python serve.py --port 8000
```

For llama-server, the encode cap and the slot size must move together — rebuild from the `edlm` branch after pulling this change, then:

```bash
SYSTEMONE_CONTEXT=32768 ./build/bin/llama-server \
  -m /path/to/drex-dlm/drex-dlm.gguf \
  --host 127.0.0.1 --port 8097 \
  --embedding --pooling none \
  -c 32768 -b 32768 -ub 32768 -np 1 \
  --no-warmup
```

For Ollama, set `PARAMETER num_ctx 32768` in the Modelfile, run `ollama create` again, and start Ollama with `SYSTEMONE_CONTEXT=32768` — `num_ctx` sizes the slot, the environment variable raises the encode cap.

`KEV_SERVE_MAX_STATE`, `KEV_SERVE_MAX_BRANCH`, and `KEV_SERVE_MAX_PACKED` (Python), and their `SYSTEMONE_*` equivalents (llama-server), override individual caps if you need finer control — none of them are required to select the full window.

## Files

| File | Role |
|---|---|
| `model-*-of-00004.safetensors` | Merged backbone, bfloat16, ~16 GB |
| `head.pt` | Pointer head: 256-dim query/key, temperature 1.0 |
| `inference.py` | Score one JSON file and print the answers |
| `serve.py` | `POST /v1/systemone` |
| `examples/request.json` | The ticket used in the sample response |

## License

MIT, including these weights. The backbone is derived from NVIDIA Efficient-DLM-8B — cite that model when citing Drex DLM.
