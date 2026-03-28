import pytest


def test_import_sakha():
    import sakha

    assert sakha.__version__ == "0.1.0"


def test_import_openenv_types():
    from openenv.core.env_server.types import Action, Observation, State

    assert Action is not None
    assert Observation is not None
    assert State is not None


def test_import_openenv_interface():
    from openenv.core.env_server.interfaces import Environment

    assert Environment is not None
