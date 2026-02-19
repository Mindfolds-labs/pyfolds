from tools.batch_create_issues import validate_batch_structure


def test_validate_batch_structure_duplicate_ids():
    config = {'issues': [{'issue_id': 'ISSUE-001'}, {'issue_id': 'ISSUE-001'}]}
    errors = validate_batch_structure(config)
    assert errors
