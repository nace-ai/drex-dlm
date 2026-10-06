# Singleton cross-runner validation (2026-10-06)

**Result: not signed off.** One decision can contain multiple named questions sharing a state; none of the accepted test requests uses the experimental multi-decision `requests` wrapper. Tests ran locally on an Apple M5 Pro (48 GiB RAM) using the BF16 checkpoint (`drex-dlm`), native `cdcf65d3` with `GGML_METAL_TENSOR_DISABLE=1`, and Ollama `2164070` with a freshly built native runner from `cdcf65d3` and the same local F16 GGUF. Both GGUF runners had a 16,384-token context, batch, and microbatch. The Ollama import was isolated under `/tmp/drex-crossrunner-models`; nothing was uploaded or pushed. Its loader logged `general.license = mit` for this **older local GGUF**, despite the corrected CC BY-NC 4.0 source metadata; do not distribute this conversion without fixing and verifying its embedded license label.

| Case | Python BF16 | Native GGUF | Ollama GGUF |
|---|---|---|---|
| Sample: one state, three questions (`team`, `refund`, `urgency`), 87 input tokens | 200; `team=billing` | 200; `team=billing` | 200; `team=billing` |
| Choice with 255 options, 2,338 input tokens | 200; option `9` | 200; option `9` | 200; option `9` |
| 256 options, empty questions, mixed single/batch body | 400 each | 400 each | 400 each (explicit `model`) |
| State beyond configured context | 400 in a later Python run before it crashed | 400 | 400 |
| `requests` wrapper alone | Experimental Python support, outside release scope | 400 | 400 |
| 15,644-token state, marker in first third, expected `COBALT` | **Server SIGSEGV; no answer**, including on isolated retry | **`AMBER` (wrong)** | **`AMBER` (wrong)** in isolated run |
| 15,644-token state, marker in final third, expected `COBALT` | No answer after crash | `COBALT` | `COBALT` in isolated run |

For the sample, input-token counts matched (87) and maximum absolute per-option BF16-vs-GGUF difference was **0.0050**. For 255 options the counts matched (2,338), both selected `9`, and the largest difference was **0.0094**. Native and Ollama matched exactly on the sample, the 255-option case, and both *isolated* long cases. Output-token counts are billing-style counts of serialized answers and need not match when rounded probabilities differ.

The long-state fixture places the same literal marker at two positions in otherwise identical repetitive filler; both requests are 15,644 encoded tokens including the question, below the configured 16,384. The first-position wrong answer is a **retrieval-correctness failure**, not a capacity rejection and not evidence that the proposed row-offset casts would fix it. The Python process crashed with SIGSEGV on the first long request both when another native model was resident and after both GGUF runners were unloaded; its cause was not established.

**Concurrency caveat:** an earlier overlapping native/Ollama run returned a spurious `0.5/0.5` distribution for the final-position case and later HTTP 400. Ollama's native logs explicitly reported `kIOGPUCommandBufferCallbackErrorOutOfMemory`. That run (`ollama.json`) is *not* reliable parity evidence. After unloading the native process and restarting Ollama's model, the separate run (`ollama-isolated.json`) returned 0.5952 for `COBALT` in the final-position case, identical to native. Avoid running these 16K GGUF contexts concurrently on this 48 GiB machine.

This is a targeted contract and long-retrieval check, **not** broad accuracy or release approval. Packed-vs-row-split behavior with enough questions to exceed the packed limit, a 1 MB request-body boundary, 32K retrieval (out of scope), CUDA, and CPU-only inference were not tested. Reproduce and diagnose the wrong first-position choice and BF16 crash before declaring 16K cross-runner sign-off. The reusable request harness is `tests/cross_runner_single.py`; raw outputs are `python.json`, `native.json`, `ollama.json`, `ollama-isolated.json`, and `python-isolated.json` in this directory.
