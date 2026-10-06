"""Review small files for the private Hugging Face model repository before release.

Dry-run only by default. Publishing requires --apply, the reviewed file digest, and an exact parent revision.
Weights, GGUF artifacts, token files, and unlisted paths are never uploaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parent
REPO_ID = "nace-ai/drex-dlm"
FILES = (
    "README.md",
    "MODEL_LICENSE.md",
    "OLLAMA.md",
    "Modelfile",
    "LICENSE",
    "requirements.txt",
    "inference.py",
    "serve.py",
    "onboard.py",
    "code/kev/__init__.py",
    "code/kev/api.py",
    "code/kev/checkpoint.py",
    "code/kev/model.py",
    "examples/request.json",
    "validation/RESULTS.md",
)
MAX_FILE_BYTES = 1_000_000


def reviewed_files(root=ROOT):
    paths = []
    for name in FILES:
        path = root / name
        parent = root
        linked_parent = False
        for part in Path(name).parts[:-1]:
            parent = parent / part
            linked_parent |= parent.is_symlink()
        if linked_parent or not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"missing, linked, or oversized release file: {name}")
        paths.append((name, path))
    return paths


def sync(api, *, expected_parent, apply=False, reviewed_digest=None, root=ROOT):
    if not expected_parent or len(expected_parent) != 40 or any(c not in "0123456789abcdef" for c in expected_parent):
        raise ValueError("--expected-parent must be the full 40-character lowercase Hub commit SHA")
    files = [(name, path.read_bytes()) for name, path in reviewed_files(root)]
    for name, data in files:
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"oversized release file after reading: {name}")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if apply and reviewed_digest != digest:
        raise ValueError("--reviewed-digest must match the dry-run file manifest")
    repo = api.repo_info(repo_id=REPO_ID, repo_type="model")
    if repo.private is not True:
        raise RuntimeError(f"refusing sync: {REPO_ID} is not private")
    if repo.sha != expected_parent:
        raise RuntimeError(f"Hub revision changed: expected {expected_parent}, found {repo.sha}")
    summary = {"repo": REPO_ID, "parent": repo.sha, "files": hashes, "review_digest": digest, "applied": apply}
    if apply:
        api.create_commit(
            repo_id=REPO_ID,
            repo_type="model",
            parent_commit=expected_parent,
            commit_message="Update Drex DLM release code and documentation",
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in files],
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-parent", required=True, help="Exact current private Hub model revision")
    parser.add_argument("--apply", action="store_true", help="Commit the reviewed small files to Hugging Face")
    parser.add_argument("--reviewed-digest", help="Dry-run review_digest, required with --apply")
    args = parser.parse_args()
    print(json.dumps(sync(HfApi(), expected_parent=args.expected_parent, apply=args.apply,
                          reviewed_digest=args.reviewed_digest), indent=2))


if __name__ == "__main__":
    main()
