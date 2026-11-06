import hashlib
from dataclasses import replace

import httpx
import pytest

from burnish.models import ModelError, ModelStore

NAME = "qwen3:8b"
REPO = "Qwen/Qwen3-8B-GGUF"
FILE = "Qwen3-8B-Q4_K_M.gguf"
BLOB = b"weights" * 1000


class Hub:
    def __init__(self, blob=BLOB, published=None):
        self.blob = blob
        self.published = published or blob
        self.files = [FILE, "Qwen3-8B-Q8_0.gguf"]
        self.status = 200
        self.honor_range = True
        self.cut_after = None
        self.ranges = []
        self.downloads = 0

    def __call__(self, request):
        if self.status != 200:
            return httpx.Response(self.status)
        if request.url.path == f"/api/models/{REPO}/tree/main":
            entries = [
                {
                    "type": "file",
                    "path": f,
                    "size": len(self.published),
                    "lfs": {
                        "oid": hashlib.sha256(self.published).hexdigest(),
                        "size": len(self.published),
                    },
                }
                for f in self.files
            ]
            return httpx.Response(200, json=entries + [{"type": "directory", "path": "docs"}])
        assert request.url.path == f"/{REPO}/resolve/main/{FILE}"
        self.downloads += 1
        byte_range = request.headers.get("range")
        self.ranges.append(byte_range)
        if byte_range and self.honor_range:
            start = int(byte_range.removeprefix("bytes=").removesuffix("-"))
            return httpx.Response(206, content=self.blob[start:])
        return httpx.Response(200, content=self.blob[: self.cut_after])


@pytest.fixture
def store_for(cfg, tmp_path):
    def build(hub):
        custom = replace(cfg, engine=replace(cfg.engine, models_dir=tmp_path))
        return ModelStore(custom, transport=httpx.MockTransport(hub))

    return build


def test_pull_downloads_verifies_and_finds_the_file(store_for):
    store = store_for(Hub())
    assert store.path(NAME) is None
    path = store.pull(NAME)
    assert path.read_bytes() == BLOB
    assert path.name == FILE
    assert ":" not in str(path.parent.name)
    assert store.path(NAME) == path


def test_pull_skips_a_downloaded_model(store_for):
    hub = Hub()
    store = store_for(hub)
    store.pull(NAME)
    store.pull(NAME)
    assert hub.downloads == 1


def test_pull_resumes_a_partial_file(store_for, tmp_path):
    hub = Hub()
    part = tmp_path / "qwen3-8b" / f"{FILE}.part"
    part.parent.mkdir()
    part.write_bytes(BLOB[:100])
    path = store_for(hub).pull(NAME)
    assert hub.ranges == ["bytes=100-"]
    assert path.read_bytes() == BLOB
    assert not part.exists()


def test_server_ignoring_range_restarts_the_file(store_for, tmp_path):
    hub = Hub()
    hub.honor_range = False
    part = tmp_path / "qwen3-8b" / f"{FILE}.part"
    part.parent.mkdir()
    part.write_bytes(b"x" * 100)
    assert store_for(hub).pull(NAME).read_bytes() == BLOB


def test_checksum_mismatch_deletes_the_file(store_for, tmp_path):
    store = store_for(Hub(blob=b"y" * len(BLOB), published=BLOB))
    with pytest.raises(ModelError, match="Checksum mismatch"):
        store.pull(NAME)
    assert store.path(NAME) is None
    assert not (tmp_path / "qwen3-8b" / f"{FILE}.part").exists()


def test_short_download_keeps_the_part_for_resuming(store_for, tmp_path):
    hub = Hub()
    hub.cut_after = 500
    with pytest.raises(ModelError, match="resume"):
        store_for(hub).pull(NAME)
    assert (tmp_path / "qwen3-8b" / f"{FILE}.part").stat().st_size == 500


def test_pattern_without_a_single_match_lists_available_files(store_for):
    hub = Hub()
    hub.files = ["Qwen3-8B-Q8_0.gguf"]
    with pytest.raises(ModelError, match="Available: Qwen3-8B-Q8_0"):
        store_for(hub).pull(NAME)


def test_gated_repo_explains_the_token(store_for):
    hub = Hub()
    hub.status = 401
    with pytest.raises(ModelError, match="HF_TOKEN"):
        store_for(hub).pull(NAME)


def test_unknown_model_is_rejected(store_for):
    with pytest.raises(ModelError, match="Unknown model"):
        store_for(Hub()).pull("nope")


def test_every_configured_model_has_a_source(cfg):
    assert set(cfg.all_models()) <= set(cfg.models)
