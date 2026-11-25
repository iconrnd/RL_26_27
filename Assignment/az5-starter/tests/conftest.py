"""Provided; not to be edited. Skips the public tests of the stages after
az5.STAGE, so that pytest passes at every submission tag."""

import pytest

import az5


STAGE_OF_FOLDER = {"stage1": 1, "stage2": 2, "stage3": 3}


def pytest_collection_modifyitems(config, items):
    for item in items:
        stage = STAGE_OF_FOLDER.get(item.path.parent.name)   # None for tests/own
        if stage is not None and stage > az5.STAGE:
            item.add_marker(pytest.mark.skip(reason=f"stage {stage} test, az5.STAGE = {az5.STAGE}"))
