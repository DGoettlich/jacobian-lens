from jlens.ui import page


def test_steer_header_is_not_a_steer_specification_row():
    html = page()

    assert '<div class="steer-header">' in html
    assert 'function steerRows()' in html
    assert 'document.querySelectorAll("#steer-rows .steer-row")' in html
