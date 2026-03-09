import pytest

from pyfolds.serialization.sharding_raid import RAIDSharding


def test_split_reconstruct_with_loss():
    r = RAIDSharding(data_shards=4, parity_shards=1)
    data = b"hello-sharding" * 100
    shards = r.split(data)
    avail = [s for i, s in enumerate(shards) if i != 2]
    idx = [i for i in range(len(shards)) if i != 2]
    rec = r.reconstruct(avail, idx)
    assert rec.startswith(data)


def test_reconstruct_insufficient():
    r = RAIDSharding()
    with pytest.raises(ValueError):
        r.reconstruct([b"a"], [0])
