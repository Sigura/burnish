import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Callable

import httpx

from burnish.config import Config, load_config

_HUB = "https://huggingface.co"
_CHUNK_BYTES = 1024 * 1024


class ModelError(RuntimeError):
    pass


class ModelStore:
    def __init__(self, cfg: Config, transport: httpx.BaseTransport | None = None):
        self.models_dir = cfg.engine.models_dir
        self.sources = cfg.models
        token = os.environ.get("HF_TOKEN")
        self.http = httpx.Client(
            base_url=_HUB,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            follow_redirects=True,
            timeout=httpx.Timeout(60, connect=30),
            transport=transport,
        )

    def _source(self, name: str):
        if name not in self.sources:
            raise ModelError(f"Unknown model {name!r}. Known models: {', '.join(sorted(self.sources))}")
        return self.sources[name]

    def _dir(self, name: str) -> Path:
        return self.models_dir / name.replace(":", "-").replace("/", "-")

    def path(self, name: str) -> Path | None:
        pattern = self._source(name).file
        directory = self._dir(name)
        if not directory.is_dir():
            return None
        matches = sorted(p for p in directory.iterdir() if fnmatch.fnmatchcase(p.name, pattern))
        return matches[0] if matches else None

    def _check_status(self, response: httpx.Response, repo: str) -> None:
        if response.status_code in (401, 403):
            raise ModelError(f"{repo} refused access ({response.status_code}). It may be gated: accept its "
                             "license on Hugging Face and set HF_TOKEN.")
        if response.status_code not in (200, 206):
            raise ModelError(f"{repo}: Hugging Face returned {response.status_code}")

    def _resolve(self, name: str) -> tuple[str, int, str]:
        source = self._source(name)
        response = self.http.get(f"/api/models/{source.repo}/tree/main", params={"recursive": "true"})
        self._check_status(response, source.repo)
        files = [e for e in response.json() if e["type"] == "file"]
        matches = [e for e in files if fnmatch.fnmatchcase(PurePosixPath(e["path"]).name, source.file)]
        if len(matches) != 1:
            available = ", ".join(e["path"] for e in files if e["path"].endswith(".gguf"))
            raise ModelError(f"{source.repo}: {len(matches)} files match {source.file!r}. Available: {available}")
        entry = matches[0]
        if not entry.get("lfs"):
            raise ModelError(f"{source.repo}: {entry['path']} has no published checksum")
        return entry["path"], entry["lfs"]["size"], entry["lfs"]["oid"]

    def pull(self, name: str, on_progress: Callable[[int, int], None] | None = None) -> Path:
        existing = self.path(name)
        if existing:
            return existing
        repo = self._source(name).repo
        try:
            remote_path, size, sha256 = self._resolve(name)
            directory = self._dir(name)
            directory.mkdir(parents=True, exist_ok=True)
            dest = directory / PurePosixPath(remote_path).name
            part = dest.with_name(dest.name + ".part")
            self._download(repo, remote_path, part, size, on_progress)
        except httpx.HTTPError as e:
            raise ModelError(f"Download of {name} failed: {e}. Run the same command again to resume.") from e

        digest = hashlib.sha256()
        with open(part, "rb") as f:
            while chunk := f.read(_CHUNK_BYTES):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            part.unlink()
            raise ModelError(f"Checksum mismatch for {remote_path}; the partial file was deleted.")
        os.replace(part, dest)
        return dest

    def _download(self, repo, remote_path, part: Path, size: int, on_progress) -> None:
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            part.unlink()
            offset = 0
        if offset == size:
            return
        free = shutil.disk_usage(part.parent).free
        if free < size - offset:
            raise ModelError(f"Not enough disk space in {part.parent}: need {size - offset} bytes, have {free}.")

        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with self.http.stream("GET", f"/{repo}/resolve/main/{remote_path}", headers=headers) as response:
            self._check_status(response, repo)
            # A server may ignore the range and send the whole file.
            mode = "ab" if response.status_code == 206 else "wb"
            written = offset if response.status_code == 206 else 0
            with open(part, mode) as f:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    f.write(chunk)
                    written += len(chunk)
                    if on_progress:
                        on_progress(written, size)
        if written != size:
            raise ModelError(f"{remote_path}: got {written} of {size} bytes. Run the same command again to resume.")


def _print_progress(name: str) -> Callable[[int, int], None]:
    last = [-1]

    def show(done: int, total: int) -> None:
        percent = done * 100 // total
        if percent != last[0]:
            last[0] = percent
            print(f"\r{name}: {percent}% of {total / 1024**3:.1f} GB", end="", file=sys.stderr, flush=True)

    return show


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="burnish-models", description="Download and list model weights.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show known models and whether they are downloaded")
    pull = commands.add_parser("pull", help="download weights; re-running resumes")
    pull.add_argument("names", nargs="*")
    pull.add_argument("--all", action="store_true", help="every model used by the config")
    args = parser.parse_args(argv)

    cfg = load_config()
    store = ModelStore(cfg)
    if args.command == "list":
        for name in sorted(cfg.models):
            print(f"{name:<14} {store.path(name) or 'not downloaded'}")
        return 0

    names = cfg.all_models() if args.all else args.names
    if not names:
        parser.error("pull needs model names or --all")
    try:
        for name in names:
            path = store.pull(name, _print_progress(name))
            print(f"\n{name}: {path}", file=sys.stderr)
    except ModelError as e:
        print(f"\nburnish-models: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
