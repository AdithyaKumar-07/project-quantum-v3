from qmi.experiments.resolutions import parse_resolutions


def test_single_resolutions():
    assert parse_resolutions("16") == (16,)
    assert parse_resolutions("32") == (32,)
    assert parse_resolutions("240") == (240,)


def test_resolution_groups_do_not_eagerly_parse_group_names():
    assert parse_resolutions("both") == (16, 32)
    assert parse_resolutions("native") == (240,)
    assert parse_resolutions("all") == (16, 32, 240)


def test_resolution_parser_normalizes_case_and_whitespace():
    assert parse_resolutions(" ALL ") == (16, 32, 240)
