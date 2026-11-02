import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from profrec import config  # noqa: E402


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """A private cache built from the real data, so tests never touch the user's cache dir."""
    d = tmp_path_factory.mktemp("cache")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "CACHE_DIR", d)
    mp.setattr(config, "CACHE_DB", d / "cache.sqlite")
    mp.setattr(config, "REQUEST_CACHE_DB", d / "requests.sqlite")
    from profrec.cache_build import ensure_cache
    info = ensure_cache(log=lambda *_: None)
    yield info
    mp.undo()


@pytest.fixture()
def engine(built):
    from profrec.engine import Engine
    return Engine(auto_build=False)
