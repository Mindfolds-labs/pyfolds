from tools.sync_hub_auto import status_to_emoji


def test_status_to_emoji_done():
    assert 'Concluída' in status_to_emoji('concluida')
