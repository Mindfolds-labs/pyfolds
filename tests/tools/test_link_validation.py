from tools.validate_issue_format import validate_links


def test_validate_links_missing(tmp_path):
    f = tmp_path / 'ISSUE-001-a.md'
    f.write_text('[x](./nao-existe.md)', encoding='utf-8')
    errors = validate_links(f)
    assert errors
