from afrimeet.utils.config import PROJECT_ROOT, load_config


def test_load_config_has_expected_top_level_keys():
    config = load_config()
    for key in ("paths", "data", "whisper", "training", "evaluation"):
        assert key in config


def test_paths_are_resolved_absolute_under_project_root():
    config = load_config()
    for path_str in config["paths"].values():
        assert str(PROJECT_ROOT) in path_str
