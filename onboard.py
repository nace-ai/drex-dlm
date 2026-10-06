#!/usr/bin/env python3
"""Print the Drex DLM runner that matches this machine.

The same JSON runs on three runners:

  python   the downloaded checkpoint's safetensors and head.pt
  llama    drex-dlm-f16.gguf, scored by the edlm llama.cpp checkout
  ollama   that GGUF, scored by the nace-edlm fork's native runner

    python onboard.py
    python onboard.py --run python
    python onboard.py --convert
    python onboard.py --run llama

Pass --convert or --run to convert or start a server.
"""
from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GGUF = ROOT / "drex-dlm-f16.gguf"
MODELFILE = ROOT / "Modelfile"


def default_weights_dir() -> Path:
    sibling = ROOT.parent / "drex-dlm-weights"
    if (sibling / "head.pt").is_file() or (sibling / GGUF.name).is_file():
        return sibling
    return ROOT


def llama_cpp_dir() -> Path | None:
    env = os.environ.get("LLAMA_CPP_SOURCE", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT.parent / "llama.cpp")
    for path in candidates:
        if (path / "convert_hf_to_gguf.py").is_file() and (path / "src" / "models" / "edlm.cpp").is_file():
            return path
    return None


def backend() -> tuple[str, str, str]:
    """Return preset, build directory, and a short reason."""
    system = platform.system()
    if system == "Darwin":
        return "darwin", "build/llama-server-darwin", "Apple Silicon Metal"
    if system == "Linux" and shutil.which("nvidia-smi"):
        return "llama_cuda_v12_linux", "build/llama-server-cuda_v12", "NVIDIA CUDA"
    if system == "Windows" and shutil.which("nvidia-smi"):
        return "llama_cuda_v12_windows", "build/llama-server-cuda_v12", "NVIDIA CUDA"
    return "cpu", "build/llama-server-cpu", "CPU"


def ollama_source_dir() -> Path:
    source = os.environ.get("OLLAMA_SOURCE", "").strip()
    return Path(source) if source else ROOT.parent / "ollama"


def llama_server_bin(fork: Path | None) -> Path | None:
    names = ["llama-server.exe", "llama-server"] if platform.system() == "Windows" else ["llama-server"]
    candidates = []
    if fork is not None:
        preset, build_dir, _ = backend()
        del preset
        candidates.append(fork / "build" / "bin" / names[0])
        candidates.append(fork / build_dir / "bin" / names[0])
    ollama_root = ollama_source_dir()
    _, build_dir, _ = backend()
    for name in names:
        candidates.append(ollama_root / build_dir / "bin" / name)
        candidates.append(ollama_root / "build" / "llama-server-cpu" / "bin" / name)
    for path in candidates:
        if path.is_file():
            return path
    return None


def convert_command(fork: Path, weights: Path = ROOT) -> list[str]:
    python = fork / ".venv-convert" / "bin" / "python"
    if not python.is_file():
        python = ROOT.parent / ".venv-convert" / "bin" / "python"
    return [
        str(python) if python.is_file() else sys.executable,
        str(fork / "convert_hf_to_gguf.py"),
        str(weights),
        "--outfile",
        str(weights / GGUF.name),
        "--outtype",
        "f16",
    ]


def llama_environment() -> dict[str, str]:
    env = os.environ.copy()
    if platform.system() == "Darwin":
        env["GGML_METAL_TENSOR_DISABLE"] = "1"
    return env


def llama_serve_command(binary: Path, weights: Path = ROOT) -> list[str]:
    return [
        str(binary),
        "-m",
        str(weights / GGUF.name),
        "--host",
        "127.0.0.1",
        "--port",
        "8097",
        "--embedding",
        "--pooling",
        "none",
        "-c",
        "16384",
        "-b",
        "16384",
        "-ub",
        "16384",
        "-np",
        "1",
        "--no-warmup",
    ]


def ollama_build_lines(fork: Path | None) -> list[str]:
    preset, build_dir, reason = backend()
    source = str(fork) if fork else "/path/to/llama.cpp"
    return [
        f"# {reason}",
        f"export OLLAMA_LLAMA_CPP_SOURCE={shlex.quote(source)}",
        f"cmake -S llama/server --preset {preset}",
        f"cmake --build {build_dir} --target llama-server --parallel 8",
    ]


def plan(weights: Path | None = None) -> str:
    weights = weights or default_weights_dir()
    modelfile = weights / MODELFILE.name
    gguf = weights / GGUF.name
    fork = llama_cpp_dir()
    preset, build_dir, reason = backend()
    binary = llama_server_bin(fork)
    lines = [
        "Drex DLM onboarding",
        "",
        f"Python loads checkpoint weights from {weights}.",
        f"Ollama loads {gguf} through {modelfile}.",
        "",
        f"Machine backend: {reason} (Ollama preset {preset}, {build_dir}).",
        "",
        "1. Python",
        f"   python serve.py --model {shlex.quote(str(weights))} --port 8000",
        "   curl http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json",
        "",
        "2. Conversion (use a separate converter environment)",
        "   Follow the complete build and install commands in README.md before running --convert.",
        "",
    ]
    if fork is None:
        lines.append("3. edlm checkout: set LLAMA_CPP_SOURCE to a llama.cpp tree that contains src/models/edlm.cpp.")
    else:
        lines.append(f"3. edlm checkout: {fork}")
        if gguf.is_file():
            lines.append(f"   GGUF already present: {gguf}")
        else:
            lines.append("   Convert once:")
            lines.append("   " + shlex.join(convert_command(fork, weights)))
        if binary is None:
            lines.append("   llama-server is not built yet. From the Ollama repository root:")
            lines.extend("   " + line for line in ollama_build_lines(fork))
        else:
            lines.append(f"   llama-server: {binary}")
            lines.append("   " + shlex.join(llama_serve_command(binary, weights)))
            lines.append("   Then POST the same JSON to http://127.0.0.1:8097/v1/systemone")
            lines.append("   Recommended context is 16384. For the full 32768 window, set SYSTEMONE_CONTEXT=32768 and use -c -b -ub 32768.")
    ollama_root = ollama_source_dir()
    lines.extend([
        "",
        "4. Ollama from the GGUF",
        f"   Modelfile is {modelfile}. The edlm architecture selects the pointer head.",
        "   Use the nace-edlm Ollama fork, not the stock ollama command.",
        f"   Build its native runner and Go daemon in {ollama_root}; see README.md.",
        f"   {shlex.join([str(ollama_root / 'ollama'), 'create', 'drex-dlm', '-f', str(modelfile)])}",
        "   curl http://127.0.0.1:11434/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json",
    ])
    return "\n".join(lines)


def write_modelfile(path: Path = MODELFILE) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing linked Modelfile: {path}")
    if path.exists():
        return
    path.write_text(
        "# Model weights: CC BY-NC 4.0, not MIT. See MODEL_LICENSE.md.\n"
        "# Model maximum: 32768 tokens; this local runner defaults to 16384 tokens.\n"
        "# For 32768, set num_ctx to 32768 and start Ollama with SYSTEMONE_CONTEXT=32768.\n"
        "FROM ./drex-dlm-f16.gguf\n"
        "CAPABILITY decision\n"
        "PARAMETER num_ctx 16384\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Choose a Drex DLM runner for this machine.")
    parser.add_argument("--run", choices=("python", "llama"), help="Start that local runner")
    parser.add_argument("--convert", action="store_true", help="Write drex-dlm-f16.gguf with the edlm fork")
    parser.add_argument("--weights", type=Path, help="Checkpoint directory (defaults to sibling drex-dlm-weights if downloaded)")
    args = parser.parse_args()
    weights = args.weights.resolve() if args.weights else default_weights_dir()
    if not weights.is_dir():
        print(f"missing checkpoint directory: {weights}", file=sys.stderr)
        return 2
    gguf = weights / GGUF.name
    modelfile = weights / MODELFILE.name
    if modelfile.is_symlink():
        print(f"refusing linked Modelfile: {modelfile}", file=sys.stderr)
        return 2
    if args.run == "python" and not (weights / "head.pt").is_file():
        print(f"missing checkpoint at {weights}; download weights first (see README.md)", file=sys.stderr)
        return 2

    if args.convert or args.run == "llama":
        fork = llama_cpp_dir()
        if fork is None:
            print("Set LLAMA_CPP_SOURCE to a llama.cpp checkout that contains src/models/edlm.cpp.", file=sys.stderr)
            return 2
        if args.convert or not gguf.is_file():
            if not (weights / "head.pt").is_file():
                print(f"missing checkpoint at {weights}; download weights first (see README.md)", file=sys.stderr)
                return 2
            cmd = convert_command(fork, weights)
            print(shlex.join(cmd))
            subprocess.check_call(cmd, cwd=fork)

    binary = None
    if args.run == "llama":
        binary = llama_server_bin(llama_cpp_dir())
        if binary is None:
            print("\n".join(ollama_build_lines(llama_cpp_dir())), file=sys.stderr)
            return 2
        if not gguf.is_file():
            print(f"missing {gguf}", file=sys.stderr)
            return 2
    try:
        write_modelfile(modelfile)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.run == "python":
        os.execv(sys.executable, [sys.executable, str(ROOT / "serve.py"), "--model", str(weights), "--port", "8000"])
    if args.run == "llama":
        os.execve(str(binary), llama_serve_command(binary, weights), llama_environment())

    print(plan(weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
