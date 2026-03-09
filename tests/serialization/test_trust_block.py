import pytest

crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
Ed25519PrivateKey = crypto.Ed25519PrivateKey

from pyfolds.serialization.trust_block import TrustBlock, verify_header


def _keys():
    pk = Ed25519PrivateKey.generate()
    priv = pk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    pub = pk.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


def test_trust_block_sign_verify_roundtrip():
    priv, pub = _keys()
    tb = TrustBlock.build({"a": 1}, {"b": 2}, private_key_pem=priv)
    raw = tb.to_bytes()
    assert TrustBlock.from_bytes(raw).verify(pub)
    assert verify_header(raw, pub)


def test_trust_block_invalid_signature():
    priv, pub = _keys()
    tb = TrustBlock.build({"a": 1}, {"b": 2}, private_key_pem=priv)
    tampered = tb.to_bytes().replace(b'"index_hash"', b'"index_hazh"')
    with pytest.raises(ValueError):
        TrustBlock.from_bytes(tampered)
    assert not TrustBlock.from_bytes(tb.to_bytes()).verify(_keys()[1])
