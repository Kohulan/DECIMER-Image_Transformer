"""Tests for downloading and loading the models on first use instead of at import.

None of these tests needs the network or the real model files: the download function
and ``tf.saved_model.load`` are replaced by recorders, and the weights are tiny
placeholder files.
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

import DECIMER.decimer as decimer_module
import DECIMER.utils as utils
from DECIMER import load_models
from DECIMER import predict_SMILES

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE = str(Path(__file__).resolve().parent / "caffeine.png")
LAZY_NAMES = ("tokenizer", "DECIMER_V2", "DECIMER_Hand_drawn")

# The model directory names in the Zenodo zip files, by download URL.
PRINTED_URL = decimer_module.model_urls["DECIMER"]
HAND_DRAWN_URL = decimer_module.model_urls["DECIMER_HandDrawn"]
DIRECTORY_OF_URL = {
    PRINTED_URL: "DECIMER_model",
    HAND_DRAWN_URL: "DECIMER_HandDrawn_model",
}


def write_model_files(models_dir, model_directory):
    """Write the placeholder files that count as an installed model."""
    directory = Path(models_dir) / model_directory
    (directory / "assets").mkdir(parents=True, exist_ok=True)
    (directory / "saved_model.pb").write_bytes(b"placeholder")
    (directory / "assets" / "tokenizer_SMILES.pkl").write_bytes(b"placeholder")


class FakeModel:
    """Stands in for a loaded SavedModel: <start> C O <end>, confidence 0.9 each."""

    def __init__(self, path):
        self.path = path

    def __call__(self, image):
        return tf.constant([[1, 2, 3, 4]]), tf.constant([0.9, 0.9, 0.9, 0.9])


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """An empty model directory, and recorders for everything that would download or load."""
    home = tmp_path / "pystow-home"
    monkeypatch.setenv("PYSTOW_HOME", str(home))
    record = SimpleNamespace(
        home=home,
        models_dir=home / "DECIMER-V2",
        downloads=[],
        loads=[],
        tokenizer_paths=[],
        download_delay=0.0,
        download_error=None,
    )

    def fake_download(model_url, model_path, verbose=1):
        record.downloads.append(model_url)
        time.sleep(record.download_delay)
        if record.download_error is not None:
            raise record.download_error
        write_model_files(model_path, DIRECTORY_OF_URL[model_url])

    def fake_saved_model_load(path, *args, **kwargs):
        record.loads.append(os.path.basename(path))
        return FakeModel(path)

    def fake_load_tokenizer(path):
        record.tokenizer_paths.append(path)
        return SimpleNamespace(index_word={1: "<start>", 2: "C", 3: "O", 4: "<end>"})

    monkeypatch.setattr(utils, "download_trained_weights", fake_download)
    monkeypatch.setattr(tf.saved_model, "load", fake_saved_model_load)
    monkeypatch.setattr(decimer_module, "load_tokenizer", fake_load_tokenizer)

    def forget_loaded_models():
        for name in LAZY_NAMES:
            vars(decimer_module).pop(name, None)

    forget_loaded_models()
    yield record
    forget_loaded_models()


def test_import_does_no_network_and_loads_no_model(tmp_path):
    """`import DECIMER` in a fresh interpreter: no network, no model, no files."""
    home = tmp_path / "pystow-home"
    script = """
import os
import pathlib
import socket


def refuse(*args, **kwargs):
    raise AssertionError("network access while importing DECIMER")


socket.socket.connect = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse

import tensorflow as tf


def refuse_load(*args, **kwargs):
    raise AssertionError("tf.saved_model.load called while importing DECIMER")


tf.saved_model.load = refuse_load

import DECIMER
import DECIMER.decimer as decimer_module

for name in ("tokenizer", "DECIMER_V2", "DECIMER_Hand_drawn"):
    assert name not in vars(decimer_module), name
home = pathlib.Path(os.environ["PYSTOW_HOME"])
assert not (home / "DECIMER-V2").exists()
assert not list(home.rglob("saved_model.pb"))
print("import-is-passive")
"""
    env = dict(os.environ)
    env["PYSTOW_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "import-is-passive" in result.stdout


def test_first_call_downloads_and_loads_once_second_call_reuses(recorded):
    assert predict_SMILES(IMAGE) == "CO"
    assert recorded.downloads == [PRINTED_URL]
    assert recorded.loads == ["DECIMER_model"]
    assert len(recorded.tokenizer_paths) == 1

    assert predict_SMILES(IMAGE) == "CO"
    assert recorded.downloads == [PRINTED_URL]
    assert recorded.loads == ["DECIMER_model"]
    assert len(recorded.tokenizer_paths) == 1


def test_hand_drawn_model_is_fetched_and_loaded_only_when_asked_for(recorded):
    predict_SMILES(IMAGE)
    assert HAND_DRAWN_URL not in recorded.downloads
    assert "DECIMER_HandDrawn_model" not in recorded.loads

    assert predict_SMILES(IMAGE, hand_drawn=True) == "CO"
    assert recorded.downloads == [PRINTED_URL, HAND_DRAWN_URL]
    assert recorded.loads == ["DECIMER_model", "DECIMER_HandDrawn_model"]

    predict_SMILES(IMAGE, hand_drawn=True)
    assert recorded.downloads == [PRINTED_URL, HAND_DRAWN_URL]
    assert recorded.loads == ["DECIMER_model", "DECIMER_HandDrawn_model"]


def test_hand_drawn_first_does_not_load_the_printed_model(recorded):
    """The tokenizer needs the printed model's files, but not the printed model itself."""
    predict_SMILES(IMAGE, hand_drawn=True)
    assert recorded.loads == ["DECIMER_HandDrawn_model"]
    assert PRINTED_URL in recorded.downloads


def test_confidence_output_is_unchanged(recorded):
    smiles, tokens = predict_SMILES(IMAGE, confidence=True)
    assert smiles == "CO"
    assert [token for token, _ in tokens] == ["C", "O"]
    assert [float(value) for _, value in tokens] == pytest.approx([0.9, 0.9])


def test_prepopulated_model_directory_is_used_without_network(recorded):
    write_model_files(recorded.models_dir, "DECIMER_model")
    write_model_files(recorded.models_dir, "DECIMER_HandDrawn_model")
    recorded.download_error = AssertionError("downloaded although the files exist")

    assert predict_SMILES(IMAGE) == "CO"
    assert predict_SMILES(IMAGE, hand_drawn=True) == "CO"

    assert recorded.downloads == []
    assert recorded.loads == ["DECIMER_model", "DECIMER_HandDrawn_model"]
    expected = os.path.join(
        recorded.models_dir, "DECIMER_model", "assets", "tokenizer_SMILES.pkl"
    )
    assert recorded.tokenizer_paths == [expected]


def test_pystow_home_set_after_import_is_respected(recorded, tmp_path, monkeypatch):
    """The directory is looked up at first use, so a late PYSTOW_HOME still applies."""
    elsewhere = tmp_path / "baked-in-image"
    write_model_files(elsewhere / "DECIMER-V2", "DECIMER_model")
    monkeypatch.setenv("PYSTOW_HOME", str(elsewhere))
    recorded.download_error = AssertionError("downloaded although the files exist")

    assert predict_SMILES(IMAGE) == "CO"
    assert recorded.downloads == []
    assert not recorded.models_dir.exists()


def test_a_bad_image_fails_before_any_download(recorded, tmp_path):
    # decimer raises a bare Exception for an image it cannot read
    with pytest.raises(Exception):
        predict_SMILES(str(tmp_path / "missing.png"))
    assert recorded.downloads == []
    assert recorded.loads == []


def test_load_models_loads_the_printed_model_once(recorded):
    load_models()
    assert recorded.downloads == [PRINTED_URL]
    assert recorded.loads == ["DECIMER_model"]

    load_models()
    assert predict_SMILES(IMAGE) == "CO"
    assert recorded.downloads == [PRINTED_URL]
    assert recorded.loads == ["DECIMER_model"]


def test_load_models_with_hand_drawn_loads_everything_decimer_2_8_0_loaded_at_import(
    recorded,
):
    load_models(hand_drawn=True)
    assert recorded.downloads == [PRINTED_URL, HAND_DRAWN_URL]
    assert sorted(recorded.loads) == ["DECIMER_HandDrawn_model", "DECIMER_model"]

    load_models(hand_drawn=True)
    assert len(recorded.loads) == 2


def test_load_models_without_the_printed_model(recorded):
    load_models(printed=False, hand_drawn=True)
    assert recorded.loads == ["DECIMER_HandDrawn_model"]


def test_failed_download_is_not_cached_and_is_retried(recorded):
    recorded.download_error = OSError("zenodo is down")
    with pytest.raises(OSError, match="zenodo is down"):
        predict_SMILES(IMAGE)
    assert recorded.loads == []

    recorded.download_error = None
    assert predict_SMILES(IMAGE) == "CO"
    assert recorded.downloads == [PRINTED_URL, PRINTED_URL]
    assert recorded.loads == ["DECIMER_model"]


def test_concurrent_first_calls_download_and_load_once(recorded):
    recorded.download_delay = 0.3
    threads_count = 4
    barrier = threading.Barrier(threads_count)
    results = []
    errors = []

    def first_call():
        try:
            barrier.wait(timeout=30)
            results.append(predict_SMILES(IMAGE))
        except Exception as error:  # reported below, in the main thread
            errors.append(error)

    threads = [threading.Thread(target=first_call) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert results == ["CO"] * threads_count
    assert recorded.downloads == [PRINTED_URL]
    assert recorded.loads == ["DECIMER_model"]


def test_module_level_names_of_older_releases_still_resolve(recorded):
    """`from DECIMER.decimer import tokenizer, DECIMER_V2, ...` worked with decimer <= 2.8.0."""
    assert not hasattr(decimer_module, "no_such_name")
    assert recorded.loads == []

    from DECIMER.decimer import DECIMER_Hand_drawn
    from DECIMER.decimer import DECIMER_V2
    from DECIMER.decimer import tokenizer

    assert isinstance(DECIMER_V2, FakeModel)
    assert DECIMER_V2.path.endswith("DECIMER_model")
    assert isinstance(DECIMER_Hand_drawn, FakeModel)
    assert DECIMER_Hand_drawn.path.endswith("DECIMER_HandDrawn_model")
    assert tokenizer.index_word[2] == "C"
    assert decimer_module.default_path == recorded.models_dir
    assert sorted(recorded.loads) == ["DECIMER_HandDrawn_model", "DECIMER_model"]


def test_a_model_assigned_to_the_module_is_used(recorded):
    """Code that replaced decimer.DECIMER_V2 with its own model keeps working."""

    class OtherModel(FakeModel):
        def __call__(self, image):
            return tf.constant([[1, 3, 3, 4]]), tf.constant([0.5, 0.5, 0.5, 0.5])

    decimer_module.DECIMER_V2 = OtherModel("my-own-model")
    assert predict_SMILES(IMAGE) == "OO"
    assert recorded.loads == []
