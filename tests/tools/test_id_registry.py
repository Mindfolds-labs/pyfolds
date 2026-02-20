from tools import id_registry


def test_next_issue_id_has_prefix():
    value = id_registry.next_issue_id()
    assert value.startswith('ISSUE-')
    assert len(value) == 9


def test_next_adr_id_has_prefix():
    value = id_registry.next_adr_id()
    assert value.startswith('ADR-')
    assert len(value) == 8
