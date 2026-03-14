import string

from train_mnist_folds import _generate_short_id


def test_generate_short_id_uses_expected_alphabet_and_length():
    generated = _generate_short_id(12)

    assert len(generated) == 12
    allowed = set(string.ascii_lowercase + string.digits)
    assert set(generated).issubset(allowed)
