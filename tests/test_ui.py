from jlens.ui import page


def test_steer_header_is_not_a_steer_specification_row():
    html = page()

    assert '<div class="steer-header">' in html
    assert 'function steerRows()' in html
    assert 'document.querySelectorAll("#steer-rows .steer-row")' in html


def test_ui_only_shows_intervention_token_ids():
    html = page()

    assert 'Token ID: ${data.answer_ids[0]}' in html
    assert 'class="token-check choice-check"' in html
    assert 'Tokenization check only' not in html
    assert 'id="mode"' not in html
    assert 'saveReport(mode === "steer" ? "Steered" : "Swapped", report.html)' in html
