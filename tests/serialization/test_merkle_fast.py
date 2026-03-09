from pyfolds.serialization.merkle_fast import FastMerkleTree


def test_merkle_root_proof_odd():
    leaves = [b"a", b"b", b"c"]
    t = FastMerkleTree(leaves)
    assert t.root == FastMerkleTree(leaves).root
    proof = t.get_proof(1)
    assert FastMerkleTree.verify(b"b", proof, t.root, 1)
    assert not FastMerkleTree.verify(b"x", proof, t.root, 1)
