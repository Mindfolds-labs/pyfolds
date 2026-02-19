from pathlib import Path

from tools.validate_issue_format import validate_structure


def test_validate_structure_filename(tmp_path: Path):
    f = tmp_path / 'INVALID.md'
    f.write_text('## PROMPT:EXECUTAR\n```yaml\na:1\n```', encoding='utf-8')
    errors = validate_structure(f)
    assert errors
