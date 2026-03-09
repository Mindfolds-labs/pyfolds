from pyfolds.serialization.provenance_light import LightProvenance


def test_provenance_chain_and_tamper():
    p = LightProvenance()
    p.add("create", {"x": 1})
    p.add("finalize", {"ok": True})
    assert p.verify()
    d = p.to_dict()
    assert len(d["entries"]) == 2
    p.entries[1].metadata["ok"] = False
    assert not p.verify()
