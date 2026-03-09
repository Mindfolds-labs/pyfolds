import pytest

from pyfolds.serialization.encryption_fast import FastEncryptor

pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead")


def test_encrypt_decrypt_and_nonce_and_aad():
    enc = FastEncryptor(b"k" * 32)
    n1, c1 = enc.encrypt(b"hello", aad=b"h")
    n2, c2 = enc.encrypt(b"hello", aad=b"h")
    assert n1 != n2
    assert enc.decrypt(n1, c1, aad=b"h") == b"hello"
    with pytest.raises(Exception):
        enc.decrypt(n1, c1, aad=b"bad")
    with pytest.raises(Exception):
        enc.decrypt(n1, c1[:-1] + b"X", aad=b"h")
    out = enc.encrypt_chunked([b"a", b"b"])
    assert len(out) == 2
