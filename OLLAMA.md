# Ollama

Drex DLM requires the `nace-edlm` branch of [nace-ai/ollama](https://github.com/nace-ai/ollama) and a native runner built from the `edlm` branch of [nace-ai/llama.cpp](https://github.com/nace-ai/llama.cpp).

Follow the [build, conversion, daemon, and import instructions](https://github.com/nace-ai/drex-dlm#ollama). The main README is the canonical serving guide.

The checkpoint and any converted GGUF weights are CC BY-NC 4.0, not MIT; see [MODEL_LICENSE.md](MODEL_LICENSE.md). The included `Modelfile` expects `drex-dlm-f16.gguf` beside it; convert weights first. With the custom daemon running, import with the **fork's** binary:

```bash
/path/to/nace-ollama/ollama create drex-dlm -f Modelfile
```

Send a single `state` and named `questions` to `POST /v1/systemone` with `"model": "drex-dlm"`. The `requests` multi-decision wrapper is outside this release's scope and is not available in published native/Ollama builds; send separate calls instead. The model supports up to **32,768 tokens**, but this local Ollama runner defaults to **16,384 tokens** (`num_ctx 16384` in the Modelfile). To enable 32K, change `num_ctx` to 32768, re-import, and restart the daemon with `SYSTEMONE_CONTEXT=32768`; long-context correctness is not yet validated.
