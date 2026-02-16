from concurrent.futures import ThreadPoolExecutor

import pytest

from pyfolds.serialization import FoldReader, save_fold_or_mind
from tests.unit.serialization.test_foldio import _build_neuron


@pytest.mark.concurrency
def test_parallel_reads_same_fold_file(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "shared.fold"

    save_fold_or_mind(neuron, str(file_path), compress="none", include_history=False, include_telemetry=False)

    def read_once() -> tuple[set[str], int]:
        with FoldReader(str(file_path), use_mmap=True) as reader:
            chunks = set(reader.list_chunks())
            payload = reader.read_chunk_bytes("torch_state", verify=True)
            return chunks, len(payload)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: read_once(), range(10)))

    for chunks, payload_len in results:
        assert "torch_state" in chunks
        assert "llm_manifest" in chunks
        assert payload_len > 0
