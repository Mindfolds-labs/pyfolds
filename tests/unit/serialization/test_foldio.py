import mmap
import struct
from unittest import mock

import numpy as np
import pytest
import torch

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import (
    FoldReader,
    FoldWriter,
    ReedSolomonECC,
    ecc_from_protection,
    is_mind,
    load_fold_or_mind,
    peek_fold_or_mind,
    read_nuclear_arrays,
    save_fold_or_mind,
)
from pyfolds.serialization.foldio import HEADER_FMT, MAGIC, MAX_CHUNK_SIZE, FoldSecurityError, FoldWriter, crc32c_u32


try:
    import reedsolo  # noqa: F401

    HAS_REEDSOLO = True
except Exception:
    HAS_REEDSOLO = False


def _build_neuron(enable_telemetry: bool = False) -> MPJRDNeuron:
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    return MPJRDNeuron(cfg, enable_telemetry=enable_telemetry, telemetry_profile="heavy")


def test_fold_roundtrip_and_peek(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "model.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        tags={"exp": "unit"},
        include_history=False,
        include_telemetry=False,
        compress="none",
    )

    peek = peek_fold_or_mind(str(file_path), use_mmap=True)

    assert "torch_state" in peek["chunks"]
    assert "llm_manifest" in peek["chunks"]
    assert "nuclear_arrays" in peek["chunks"]
    assert peek["metadata"]["model_type"] == "MPJRDNeuron"
    assert peek["is_mind"] is False
    assert is_mind(str(file_path)) is False

    loaded = load_fold_or_mind(str(file_path), MPJRDNeuron, map_location="cpu")

    assert loaded.__class__.__name__ == "MPJRDNeuron"
    assert loaded.cfg.n_dendrites == neuron.cfg.n_dendrites
    assert loaded.cfg.n_synapses_per_dendrite == neuron.cfg.n_synapses_per_dendrite


def test_fold_roundtrip_preserves_state_dict_after_forward_steps(tmp_path):
    neuron = _build_neuron(enable_telemetry=True)

    x = torch.rand(8, neuron.cfg.n_dendrites, neuron.cfg.n_synapses_per_dendrite)
    for _ in range(5):
        neuron.forward(x, collect_stats=True)

    file_path = tmp_path / "state-roundtrip.fold"
    save_fold_or_mind(
        neuron,
        str(file_path),
        tags={"audit": "issue-026"},
        include_history=True,
        include_telemetry=True,
        include_nuclear_arrays=True,
        compress="none",
    )

    loaded = load_fold_or_mind(str(file_path), MPJRDNeuron, map_location="cpu")

    assert loaded.cfg.n_dendrites == neuron.cfg.n_dendrites
    assert loaded.cfg.n_synapses_per_dendrite == neuron.cfg.n_synapses_per_dendrite

    for key, value in neuron.state_dict().items():
        assert torch.allclose(value, loaded.state_dict()[key], atol=1e-6), f"Mismatch in param {key}"


def test_training_then_save_with_telemetry_and_history_and_nuclear_arrays(tmp_path):
    neuron = _build_neuron(enable_telemetry=True)

    for _ in range(3):
        x = torch.rand(4, neuron.cfg.n_dendrites, neuron.cfg.n_synapses_per_dendrite)
        neuron.forward(x, collect_stats=True)

    neuron.apply_plasticity(dt=1.0)
    file_path = tmp_path / "trained.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        tags={"phase": "trained"},
        include_history=True,
        include_telemetry=True,
        include_nuclear_arrays=True,
        compress="none",
    )

    info = peek_fold_or_mind(str(file_path), use_mmap=True)
    assert "metrics" in info["chunks"]
    assert "telemetry" in info["chunks"]
    assert "nuclear_arrays" in info["chunks"]
    assert info["llm_manifest"]["routing"]["resume_training"] == "torch_state"

    with FoldReader(str(file_path), use_mmap=True) as reader:
        telemetry = reader.read_json("telemetry")

    arrays = read_nuclear_arrays(str(file_path))
    assert telemetry["enabled"] is True
    assert isinstance(telemetry["events"], list)
    assert arrays["N"].shape == (neuron.cfg.n_dendrites, neuron.cfg.n_synapses_per_dendrite)
    assert np.isfinite(arrays["theta"]).all()


def test_detects_corruption(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "corrupt.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )

    with FoldReader(str(file_path), use_mmap=False) as reader:
        torch_state_chunk = next(c for c in reader.index["chunks"] if c["name"] == "torch_state")
        byte_to_flip = torch_state_chunk["offset"] + torch_state_chunk["header_len"] + 8

    with open(file_path, "r+b") as f:
        f.seek(byte_to_flip)
        original = f.read(1)
        f.seek(byte_to_flip)
        f.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(RuntimeError):
        with FoldReader(str(file_path), use_mmap=False) as reader:
            reader.read_chunk_bytes("torch_state", verify=True)


def test_hierarchical_hashes_present_in_metadata(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "hashes.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )

    with FoldReader(str(file_path), use_mmap=False) as reader:
        metadata = reader.index.get("metadata", {})

    assert "chunk_hashes" in metadata
    assert "manifest_hash" in metadata
    assert "torch_state" in metadata["chunk_hashes"]


def test_ecc_from_protection_mapping():
    assert ecc_from_protection("off").name == "none"

    if HAS_REEDSOLO:
        assert ecc_from_protection("low").name == "rs(16)"
        assert ecc_from_protection("med").name == "rs(32)"
        assert ecc_from_protection("high").name == "rs(64)"
    else:
        with pytest.raises(ModuleNotFoundError):
            ecc_from_protection("low")

    with pytest.raises(ValueError):
        ecc_from_protection("ultra")


def test_ecc_roundtrip_if_available(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "ecc.fold"

    ecc_codec = ReedSolomonECC(symbols=16) if HAS_REEDSOLO else ecc_from_protection("off")

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=True,
        compress="none",
        ecc=ecc_codec,
    )

    with FoldReader(str(file_path), use_mmap=True) as reader:
        chunks = {c["name"]: c for c in reader.index["chunks"]}

    if HAS_REEDSOLO:
        assert chunks["torch_state"]["ecc_algo"] == "rs(16)"
        assert chunks["torch_state"]["ecc_len"] > 0
    else:
        assert chunks["torch_state"]["ecc_algo"] == "none"
        assert chunks["torch_state"]["ecc_len"] == 0


def test_crc32c_matches_known_vector():
    assert crc32c_u32(b"123456789") == 0xE3069283


def test_fold_reader_bounds_validation_with_mmap(tmp_path):
    file_path = tmp_path / "bounds.fold"
    file_path.write_bytes(b"X" * 64)

    reader = FoldReader(str(file_path), use_mmap=True)
    reader._f = open(file_path, "rb")
    reader._mm = mmap.mmap(reader._f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        with pytest.raises(EOFError, match="além do arquivo"):
            reader._read_at(32, 40)
    finally:
        reader.__exit__(None, None, None)


def test_fold_reader_bounds_validation_negative_values(tmp_path):
    file_path = tmp_path / "bounds-neg.fold"
    file_path.write_bytes(b"X" * 32)

    reader = FoldReader(str(file_path), use_mmap=False)
    reader._f = open(file_path, "rb")
    try:
        with pytest.raises(ValueError, match="offset e length devem ser >= 0"):
            reader._read_at(-1, 4)
        with pytest.raises(ValueError, match="offset e length devem ser >= 0"):
            reader._read_at(0, -4)
    finally:
        reader.__exit__(None, None, None)


def test_fold_reader_index_size_validation(tmp_path):
    file_path = tmp_path / "huge-index.fold"
    fake_header = struct.pack(HEADER_FMT, MAGIC, struct.calcsize(HEADER_FMT), 28, 2_000_000_000)
    file_path.write_bytes(fake_header + (b"\x00" * 128))

    with pytest.raises(ValueError, match="Index muito grande"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_reader_index_offset_validation(tmp_path):
    file_path = tmp_path / "bad-index-offset.fold"
    header_size = struct.calcsize(HEADER_FMT)
    fake_header = struct.pack(HEADER_FMT, MAGIC, header_size, header_size - 1, 0)
    file_path.write_bytes(fake_header)

    with pytest.raises(ValueError, match="Index offset inválido"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_reader_header_len_validation(tmp_path):
    file_path = tmp_path / "bad-header-len.fold"
    header_size = struct.calcsize(HEADER_FMT)
    fake_header = struct.pack(HEADER_FMT, MAGIC, header_size + 1, header_size, 0)
    file_path.write_bytes(fake_header)

    with pytest.raises(ValueError, match="Header inconsistente"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_writer_finalize_wraps_io_failure_with_phase_context(tmp_path, monkeypatch):
    file_path = tmp_path / "finalize-fail.fold"

    def _fail_fsync(_fd):
        raise OSError("disk full")

    monkeypatch.setattr("pyfolds.serialization.foldio.os.fsync", _fail_fsync)

    with FoldWriter(str(file_path), compress="none") as writer:
        writer.add_chunk("meta", "JSON", b"{}")

        with pytest.raises(RuntimeError, match=r"persistência do arquivo fold \(index fsync\)") as exc_info:
            writer.finalize({"source": "unit-test"})

    assert isinstance(exc_info.value.__cause__, OSError)


def test_fold_reader_reports_magic_values(tmp_path):
    file_path = tmp_path / "wrong-magic.fold"
    bad_magic = b"NOTFOLD!"
    header = struct.pack(HEADER_FMT, bad_magic, struct.calcsize(HEADER_FMT), 24, 0)
    file_path.write_bytes(header)

    with pytest.raises(ValueError, match="Magic esperado"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_reader_exit_closes_file_even_if_mmap_close_fails(tmp_path):
    class FailingMM:
        def close(self):
            raise RuntimeError("forced close error")

    file_path = tmp_path / "cleanup.fold"
    file_path.write_bytes(b"abc")

    reader = FoldReader(str(file_path), use_mmap=True)
    reader._f = open(file_path, "rb")
    reader._mm = FailingMM()

    with pytest.raises(RuntimeError, match="forced close error"):
        reader.__exit__(None, None, None)

    assert reader._f is None
    assert reader._mm is None


def _build_writer_with_chunk(tmp_path):
    writer = FoldWriter(str(tmp_path / "writer.fold"), compress="none")
    writer.__enter__()
    writer.add_chunk("dummy", "DUMY", b"payload")
    return writer


@pytest.mark.parametrize(
    ("phase", "patcher", "error_message"),
    [
        (
            "write_index",
            lambda writer, monkeypatch: monkeypatch.setattr(
                writer._f,
                "flush",
                mock.Mock(side_effect=OSError("flush failed")),
            ),
            "flush failed",
        ),
        (
            "fsync_index",
            lambda writer, monkeypatch: monkeypatch.setattr(
                "pyfolds.serialization.foldio.os.fsync",
                mock.Mock(side_effect=OSError("fsync index failed")),
            ),
            "fsync index failed",
        ),
        (
            "write_header",
            lambda writer, monkeypatch: monkeypatch.setattr(
                writer._f,
                "seek",
                mock.Mock(side_effect=OSError("seek failed")),
            ),
            "seek failed",
        ),
        (
            "write_header",
            lambda writer, monkeypatch: monkeypatch.setattr(
                writer._f,
                "write",
                _failing_write_after_first_call(writer._f.write),
            ),
            "write failed",
        ),
    ],
)
def test_fold_writer_finalize_wraps_failures_with_phase(
    tmp_path,
    monkeypatch,
    phase,
    patcher,
    error_message,
):
    writer = _build_writer_with_chunk(tmp_path)
    try:
        patcher(writer, monkeypatch)
        with pytest.raises(RuntimeError, match=rf"fase '{phase}'") as excinfo:
            writer.finalize({"model_type": "dummy"})

        assert error_message in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)
    finally:
        writer._f.flush = mock.Mock()
        writer.__exit__(None, None, None)


def _failing_write_after_first_call(original_write):
    call_count = {"value": 0}

    def _write(data):
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise OSError("write failed")
        return original_write(data)

    return mock.Mock(side_effect=_write)


def test_fold_manifest_includes_governance_sections(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "governance.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        compress="none",
        dataset_manifest={"name": "mnist", "version": "1.0"},
        performance_manifest={"latency_ms": 3.2, "throughput": 1200},
        fairness_manifest={"demographic_parity_gap": 0.02},
        explainability_manifest={"method": "integrated-gradients"},
        compliance_manifest={"standards": ["IEEE-730", "ISO-15288"]},
    )

    info = peek_fold_or_mind(str(file_path), use_mmap=False)
    llm_manifest = info["llm_manifest"]
    assert llm_manifest["hyperparameters"]["n_dendrites"] == neuron.cfg.n_dendrites
    assert llm_manifest["dataset"]["name"] == "mnist"
    assert llm_manifest["performance"]["throughput"] == 1200
    assert llm_manifest["fairness"]["demographic_parity_gap"] == 0.02
    assert llm_manifest["explainability"]["method"] == "integrated-gradients"
    assert llm_manifest["compliance"]["standards"][0] == "IEEE-730"


def test_fold_signature_roundtrip_if_cryptography_available(tmp_path):
    serialization_module = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    ed25519_module = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")

    private_key = ed25519_module.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization_module.Encoding.PEM,
        format=serialization_module.PrivateFormat.PKCS8,
        encryption_algorithm=serialization_module.NoEncryption(),
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization_module.Encoding.PEM,
        format=serialization_module.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    neuron = _build_neuron()
    file_path = tmp_path / "signed.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        compress="none",
        include_history=False,
        include_telemetry=False,
        signature_private_key_pem=private_pem,
        signature_key_id="unit-test",
    )

    with FoldReader(str(file_path), use_mmap=False) as reader:
        signature = reader.index["metadata"]["signature"]
    assert signature["algorithm"] == "ed25519"
    assert signature["key_id"] == "unit-test"

    loaded = load_fold_or_mind(
        str(file_path),
        MPJRDNeuron,
        signature_public_key_pem=public_pem,
    )
    assert loaded.__class__.__name__ == "MPJRDNeuron"

    with pytest.raises(FoldSecurityError):
        load_fold_or_mind(
            str(file_path),
            MPJRDNeuron,
            signature_public_key_pem=private_pem,
        )
