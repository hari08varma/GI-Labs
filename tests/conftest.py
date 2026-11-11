import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gca.world.generator import StreamBuilder, load_config  # noqa: E402

CFG_PATH = os.path.join(ROOT, "configs", "stream_default.yaml")


@pytest.fixture(scope="session")
def cfg():
    return load_config(CFG_PATH)


def mini_cfg(cfg, ids):
    c = dict(cfg)
    c["systems"] = [s for s in cfg["systems"] if s["id"] in ids]
    return c


@pytest.fixture(scope="session")
def stream0(cfg):
    sb = StreamBuilder(cfg, 0, 0.0)
    return sb, sb.build()


@pytest.fixture(scope="session")
def mini(cfg):
    """A 3-system mini stream (S01-S03), noise-free."""
    sb = StreamBuilder(mini_cfg(cfg, {"S01", "S02", "S03"}), 0, 0.0)
    return sb, sb.build()
