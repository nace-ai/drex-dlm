---
license: cc-by-nc-4.0
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

The model supports a maximum context window of **32,768 tokens**, but the local Python, llama-server, and Ollama runners use **16,384 tokens by default**. The 32K window is not enabled automatically; [Context length](#context-length) explains how to configure it for each runner.

Python 3.12 is the validated interpreter. Local inference was tested on an Apple M5 Max with 128 GiB of unified memory; CUDA and CPU-only inference have not yet been validated. BF16 weights occupy about 16 GB, but actual memory use grows with context length and batching. Long-context quality is experimental.

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
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf download nace-ai/drex-dlm --local-dir ../drex-dlm-weights
python inference.py --model ../drex-dlm-weights --request examples/request.json
```

The checkpoint lives in a sibling directory so its code cannot overwrite the GitHub checkout. The `hf` command in these steps requires `huggingface_hub>=0.34`, installed by `requirements.txt` ([CLI rename in v0.34.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.34.0)). While private, authenticate with `hf auth login` before downloading. That prints answers for the sample ticket. To run the inference server instead:

```bash
python serve.py --model ../drex-dlm-weights --port 8000
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

`GET /health` returns `{"status": "ok"}` once the weights are loaded. The same request also runs through `llama-server` and Ollama (see [Serving](#serving)). `python onboard.py` prints commands using the downloaded sibling `../drex-dlm-weights` when present; use `python onboard.py --weights /path/to/checkpoint` for another location. The helper puts its generated Modelfile beside the GGUF, not in the GitHub checkout, and does not replace an existing one; it rejects linked Modelfiles rather than following them. Set `OLLAMA_SOURCE=/path/to/nace-edlm-checkout` if the Ollama fork is not at the default sibling `../ollama`; the printed create command uses that checkout's executable.

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

`choice.confidence` rescales the winning probability above a uniform baseline. `score.confidence` measures concentration near the modal level; neither is a calibrated probability of correctness. The local score-confidence formula approximates the hosted API's statistic.

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
      "legend": ["Routine", "Soon", "Urgent"],
      "probabilities": { "0": 0.1876, "1": 0.2169, "2": 0.5955 },
      "confidence": 0.7039
    }
  }
}
```

`usage.input_tokens` is the encoded document plus questions (87 for this ticket). `usage.output_tokens` counts the serialized answers, not generated text. `latency_ms` is the scoring time.

**One decision per request is the release contract.** Experimental multi-decision `requests` work remains in a separate local development checkout and is not part of this release. Published Python, native, and Ollama builds do not support that wrapper; make separate calls on all three runners.

## Serving

| Runner | Weights | Listens on |
|---|---|---|
| Python | `drex-dlm-weights`, including `head.pt` | `POST /v1/systemone`, port 8000 |
| llama-server | `drex-dlm-f16.gguf` | `POST /v1/systemone`, port 8097 |
| Ollama | the same GGUF | `POST /v1/systemone`, port 11434 |

When combined questions exceed the packed-token cap, the runners can score separate question rows, provided the state plus each question fits the configured row limit. Floating-point probabilities can differ slightly between packed and row execution. Native context, batch, and microbatch capacities must each fit an encoded span. See [Context length](#context-length) for details.

### Python

```bash
source .venv/bin/activate
python serve.py --model ../drex-dlm-weights --host 127.0.0.1 --port 8000
```

The server loads the safetensors shards and `head.pt` from `--model`; `--name` sets the fallback response model name. Run it from `drex-dlm` after the quick start.

### llama-server

The `edlm` architecture, its converter, and `POST /v1/systemone` live on branch `edlm` of [nace-ai/llama.cpp](https://github.com/nace-ai/llama.cpp). From `drex-dlm`, clone a sibling checkout and build the server (CMake and a C/C++ toolchain required; Apple Silicon also needs the Xcode Metal toolchain):

```bash
cd ..
git clone --branch edlm --single-branch https://github.com/nace-ai/llama.cpp.git llama.cpp
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --target llama-server --parallel 8
```

For NVIDIA, add `-DGGML_CUDA=ON` when configuring; CUDA has not been validated for this release. Use a separate conversion environment because converter dependencies differ from the inference requirements:

```bash
python3.12 -m venv llama.cpp/.venv-convert
llama.cpp/.venv-convert/bin/python -m pip install \
  -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
llama.cpp/.venv-convert/bin/python llama.cpp/convert_hf_to_gguf.py drex-dlm-weights \
  --outfile drex-dlm-weights/drex-dlm-f16.gguf --outtype f16
cd drex-dlm
```

From `drex-dlm`, start the server, then send the sample from another terminal:

```bash
GGML_METAL_TENSOR_DISABLE=1 ../llama.cpp/build/bin/llama-server \
  -m ../drex-dlm-weights/drex-dlm-f16.gguf \
  --host 127.0.0.1 --port 8097 \
  --embedding --pooling none \
  -c 16384 -b 16384 -ub 16384 -np 1 --no-warmup
```

```bash
curl http://127.0.0.1:8097/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

Keep `GGML_METAL_TENSOR_DISABLE=1` on Apple Silicon: the Metal tensor matmul path produced incorrect long-input results in validation. If an encoded span exceeds the native `-c`, `-b`, or `-ub` capacity, it is rejected; the server does not adjust rows to smaller launch capacities. The response includes Python's `answers` schema plus `latency_ms` and an `x-typesafe-request-id` header.

### Ollama

[nace-ai/ollama](https://github.com/nace-ai/ollama) branch `nace-edlm` launches a custom `llama-server` and forwards `POST /v1/systemone`. Complete the native checkout and conversion above first. From `drex-dlm`, clone a sibling checkout and build the Apple Silicon runner and Go daemon (Go 1.26 with automatic toolchain download):

```bash
cd ..
git clone --branch nace-edlm --single-branch https://github.com/nace-ai/ollama.git ollama
cd ollama
export OLLAMA_LLAMA_CPP_SOURCE="$PWD/../llama.cpp"
cmake -S llama/server --preset darwin
cmake --build build/llama-server-darwin --target llama-server --parallel 8
GOTOOLCHAIN=auto go build -trimpath -o ollama .
cd ../drex-dlm
```

The Linux CUDA and CPU presets have not been validated for this release. Build from the matching native fork, not the stock Ollama binary. Write a Modelfile next to the converted GGUF:

```bash
cat > ../drex-dlm-weights/Modelfile <<'EOF'
FROM ./drex-dlm-f16.gguf
CAPABILITY decision
PARAMETER num_ctx 16384
EOF
```

Start the daemon from `drex-dlm` (leave it running) and, in another terminal, import and call the model:

```bash
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_LLAMA_SERVER="$PWD/../ollama/build/llama-server-darwin/bin/llama-server" \
  ../ollama/ollama serve
```

```bash
OLLAMA_HOST=127.0.0.1:11434 ../ollama/ollama create drex-dlm \
  -f ../drex-dlm-weights/Modelfile
curl http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

`GET /api/version` checks readiness. Current fork builds align native context, batch, and microbatch capacities; on macOS, the Ollama fork disables the problematic Metal tensor path for eDLM child runners.

## Context length

**Model capacity: 32,768 tokens. Local runner default: 16,384 tokens.** The Python server defaults to 16K, the llama-server command above sets `-c -b -ub` to 16K, and the Ollama Modelfile sets `num_ctx` to 16K.

Python already uses the recommended default. To use the full window instead:

```bash
KEV_CONTEXT=32768 python serve.py --model ../drex-dlm-weights --port 8000
```

For llama-server, the encode cap and the slot size must move together — rebuild from the `edlm` branch after pulling this change, then:

```bash
GGML_METAL_TENSOR_DISABLE=1 SYSTEMONE_CONTEXT=32768 ../llama.cpp/build/bin/llama-server \
  -m ../drex-dlm-weights/drex-dlm-f16.gguf \
  --host 127.0.0.1 --port 8097 \
  --embedding --pooling none \
  -c 32768 -b 32768 -ub 32768 -np 1 \
  --no-warmup
```

For Ollama, set `PARAMETER num_ctx 32768` in `../drex-dlm-weights/Modelfile`, repeat the fork's `ollama create` command, and restart its daemon with `SYSTEMONE_CONTEXT=32768`. `num_ctx` sizes the slot (the patched fork also sizes its decision batch and microbatch); the environment variable raises the encode cap. Keep them aligned. A successful 32K capacity check is not a correctness guarantee.

`KEV_SERVE_MAX_STATE`, `KEV_SERVE_MAX_BRANCH`, and `KEV_SERVE_MAX_PACKED` (Python), and their `SYSTEMONE_*` equivalents (llama-server), override individual caps if you need finer control — none of them are required to select the full window.

## Validation scope

An October 2026 Apple M5 Max smoke suite of 26 requests and 53 questions (50 independently labeled) was answered correctly by Python, the original native fork, and the original Ollama fork. That synthetic suite does not reproduce Decision Index 0.2, demonstrate calibration or broad quality, or establish equivalence to hosted Drex. The published native fork already widens Metal matrix batch offsets; proposed row-offset casts are not a release gate. A later local 15,644-token singleton retrieval check failed on the first-position marker in native and Ollama and crashed the Python BF16 server; see `validation/RESULTS.md`. The older local GGUF used in that check embeds `general.license = mit`; it is **not distributed in this GitHub or Hugging Face release**. Convert afresh from the updated Hub model card and verify `general.license = cc-by-nc-4.0` before distributing any GGUF. Model weights remain CC BY-NC 4.0. Final-runner 16K sign-off remains open; multi-decision batching is outside this release's scope. Latency depends on the hardware, context, dtype, workload, and concurrency; no universal latency or sustained serving rate is claimed.

## Files

| File | Role |
|---|---|
| `model-*-of-00004.safetensors` | Merged backbone, bfloat16, ~16 GB |
| `head.pt` | Pointer head: 256-dim query/key, temperature 1.0 |
| `inference.py` | Score one JSON file and print the answers |
| `serve.py` | `POST /v1/systemone` |
| `examples/request.json` | The ticket used in the sample response |

## License

The **model weights are released under CC BY-NC 4.0**, not MIT. They derive from [NVIDIA Efficient-DLM-8B](https://huggingface.co/nvidia/Efficient-DLM-8B), whose published [model card](https://huggingface.co/nvidia/Efficient-DLM-8B/raw/main/README.md) specifies CC BY-NC 4.0. The backbone was further trained/merged with a decision adapter and paired with a pointer head; these are modifications of the upstream model. Attribute both the original authors and Nace.AI, link the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/), and indicate the modifications when redistributing. **Commercial use of the weights is not licensed by this release.**

The original code written by Nace.AI in this repository is under the [MIT License](LICENSE). Third-party dependencies and inherited model code retain their respective licenses. This separation does not relicense the upstream weights or code under MIT; review upstream notices before redistribution.
