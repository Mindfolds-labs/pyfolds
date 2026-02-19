from tools.create_issue_report import generate_yaml_frontmatter, IssueData


def test_generate_yaml_frontmatter_contains_id():
    text = generate_yaml_frontmatter(IssueData('ISSUE-999', 'Tema', 'Alta', 'Core'))
    assert 'id: "ISSUE-999"' in text
