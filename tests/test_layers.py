from foldsnet.factory import create_foldsnet


def test_retina_neurons():
    model = create_foldsnet("4L", "mnist")
    assert len(model.retina) == 49


def test_lgn_neurons():
    model = create_foldsnet("4L", "mnist")
    assert len(model.lgn) == 49


def test_v1_neurons():
    model = create_foldsnet("4L", "mnist")
    assert len(model.v1) == 98


def test_it_neurons():
    model = create_foldsnet("4L", "mnist")
    assert len(model.it) == 49


def test_sparse_connections_shape():
    model = create_foldsnet("4L", "mnist")
    assert model.lgn_to_v1.shape == (98, 49)
    assert model.v1_to_it.shape == (49, 98)
