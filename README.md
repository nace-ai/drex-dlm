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

The backbone is [NVIDIA Efficient-DLM-8B](https://huggingface.co/nvidia/Efficient-DLM-8B), with a decision adapter merged into the weights. The block tensor names match Qwen3. Attention is the diffusion encoder: the document is bidirectional, and each question is its own causal branch, so one question's options stay invisible to the next question. The pointer head reads the hidden state at the decision marker and at each option marker, then turns those two vectors into a score.

![Drex DLM architecture: bidirectional shared context feeds isolated causal question branches. A shared pointer head projects decision and option hidden states into queries and keys, then converts scaled dot-product scores into option probabilities.](assets/drex-dlm-architecture.png)

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

That prints the answers for the sample ticket. To start the inference server:

```bash
python serve.py --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

`GET /health` returns `{"status": "ok"}` once the weights are loaded.

The same request also runs through [llama-server](#llama-server) and [Ollama](#ollama).

## A request

Two fields matter.

| Field | What to put there |
| --- | --- |
| `state` | The document: a string, an object, or a list. Nested objects are rendered as indented text. |
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
| --- | --- | --- |
| `choice` | Named options, each with a short description. Object key order is option order. | `choice`, a probability for every name, and `confidence` |
| `noul` | A yes-or-no question. Optional `criteria.true` and `criteria.false` descriptions. | `noul`, the probability that the answer is yes, from 0 to 1 |
| `score` | An ordered scale, lowest first | `score`, a probability-weighted level, plus `legend`, per-level probabilities, and `confidence` |

`confidence` measures how far the distribution sits from a tie. A high value means one option carries most of the probability.

A local bfloat16 run of `examples/request.json` returns:

```json
{
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9461,
      "probabilities": {
        "billing": 0.9641,
        "technical": 0.001,
        "other": 0.0349
      }
    },
    "refund": {
      "type": "noul",
      "noul": 0.8548
    },
    "urgency": {
      "type": "score",
      "score": 1.4078,
      "legend": {
        "0": "Routine",
        "1": "Soon",
        "2": "Urgent"
      },
      "probabilities": {
        "0": 0.1876,
        "1": 0.2169,
        "2": 0.5955
      },
      "confidence": 0.7039
    }
  }
}
```

`usage.input_tokens` is the encoded document plus questions. For this ticket that is 87. `usage.output_tokens` counts the serialized answers, not generated text. `latency_ms` is the scoring time.

Several decisions can share one forward. Send `requests`, a list of objects with `state` and `questions`. The response is `results` in that order. Each result has its own `answers`, `usage`, and `latency_ms`. That `latency_ms` is the shared forward, not a separate score for the item. One object, as above, stays one decision. A list cannot also contain `state` or `questions`.

## Serving

| Runner | Weights | Listen |
| --- | --- | --- |
| Python | this directory, including `head.pt` | `POST /v1/systemone` on port 8000 |
| llama-server | `drex-dlm.gguf` | `POST /v1/systemone` on port 8097 |
| Ollama | the same GGUF | `POST /v1/systemone` on port 11434 |

### Python

```bash
pip install -r requirements.txt
python serve.py --host 127.0.0.1 --port 8000
```

The server loads the safetensors shards and `head.pt` from this directory. `--model` points at another checkout of the same files. `--name` sets the `model` field in the response.

The recommended context is 16,384 tokens. A state longer than the selected context comes back as an error, so the score always covers the whole document. One question, including the state, fits in that same limit. Several questions share one forward pass up to that limit. Past that, Python scores one question at a time. The probabilities are the same either way.

### llama-server

The `edlm` architecture, the converter, and `POST /v1/systemone` are on branch `edlm` of [nace-ai/llama.cpp](https://github.com/nace-ai/llama.cpp).

```bash
git clone --branch edlm https://github.com/nace-ai/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --target llama-server -j
```

NVIDIA:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-server -j
```

Convert this directory with that checkout. The converter copies the pointer head into the GGUF.

```bash
python convert_hf_to_gguf.py /path/to/drex-dlm \
  --outfile /path/to/drex-dlm/drex-dlm.gguf \
  --outtype f16
```

Start the server at the recommended 16,384-token context. One request is encoded in a single batch.

```bash
./build/bin/llama-server \
  -m /path/to/drex-dlm/drex-dlm.gguf \
  --host 127.0.0.1 --port 8097 \
  --embedding --pooling none \
  -c 16384 -b 16384 -ub 16384 -np 1 \
  --no-warmup
```

```bash
curl http://127.0.0.1:8097/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

The response uses the same `answers` fields as Python, plus `latency_ms` and an `x-typesafe-request-id` header. One forward pass covers the selected context. A longer request is scored one question at a time. If the server was started with a smaller `-c`, `-b`, or `-ub` than the batch it is asked to score, the request is rejected instead of split. F16 is the converted type used with this server.

### Ollama

[nace-ai/ollama](https://github.com/nace-ai/ollama) branch `nace-edlm` launches that same `llama-server` and forwards `POST /v1/systemone`. Build the server from [nace-ai/llama.cpp](https://github.com/nace-ai/llama.cpp) branch `edlm` before configuring Ollama:

```bash
export OLLAMA_LLAMA_CPP_SOURCE=/path/to/llama.cpp
cmake -S llama/server --preset darwin
cmake --build build/llama-server-darwin --target llama-server --parallel 8
```

Apple Silicon uses `darwin`. NVIDIA Linux uses `llama_cuda_v12_linux` and `build/llama-server-cuda_v12`. CPU uses `cpu` and `build/llama-server-cpu`. Set the source variable before `cmake -S`. Then create the model from `drex-dlm.gguf` and post the same JSON to `http://127.0.0.1:11434/v1/systemone`. `PARAMETER num_ctx 16384` in `Modelfile` is the recommended context.

## Context length

The context length is 32,768 tokens. That is the position window in the weights. The recommended default is 16,384. A state, and one question including the state, must fit in the selected context. Several questions share one forward up to that same limit. Past that, the server scores one question at a time.

| | Tokens |
| --- | --- |
| Context length | 32,768 |
| Recommended default | 16,384 |
| Options in one question | 255 |

A value above 32,768 is ignored, and the recommended 16,384 stays in force.

Python already uses the recommended default. For the full window:

```bash
KEV_CONTEXT=32768 python serve.py --port 8000
```

llama-server needs the encode cap and the slot to move together. Rebuild `llama-server` from the `edlm` branch after pulling this change:

```bash
SYSTEMONE_CONTEXT=32768 ./build/bin/llama-server \
  -m /path/to/drex-dlm/drex-dlm.gguf \
  --host 127.0.0.1 --port 8097 \
  --embedding --pooling none \
  -c 32768 -b 32768 -ub 32768 -np 1 \
  --no-warmup
```

Ollama: set `PARAMETER num_ctx 32768` in `Modelfile`, run `ollama create` again, and start Ollama with `SYSTEMONE_CONTEXT=32768`. `num_ctx` sizes the slot. The environment variable raises the encode cap.

`KEV_SERVE_MAX_STATE`, `KEV_SERVE_MAX_BRANCH`, and `KEV_SERVE_MAX_PACKED` override one Python cap. `SYSTEMONE_MAX_STATE`, `SYSTEMONE_MAX_BRANCH`, and `SYSTEMONE_MAX_PACKED` do the same for llama-server. You do not need them to select the full window.

A question may be `choice`, `noul`, or `score`. Option text is the option name plus its description. A `<|name|>` span in user text is rewritten to `<¦name¦>` before tokenization, which keeps the document separate from the five delimiter tokens.

## Files

| File | Role |
| --- | --- |
| `model-*-of-00004.safetensors` | Merged backbone, bfloat16, about 16 GB |
| `head.pt` | Pointer head: 256-dimensional query and key, temperature 1.0 |
| `inference.py` | Score one JSON file and print the answers |
| `serve.py` | `POST /v1/systemone` |
| `examples/request.json` | The ticket used in the sample response |

## License

MIT, including these weights. The backbone was derived from NVIDIA Efficient-DLM-8B. Cite that model when you cite Drex DLM.
