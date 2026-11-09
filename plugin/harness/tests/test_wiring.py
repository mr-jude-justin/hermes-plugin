"""register() wiring and the manifests that must agree with it."""

import importlib
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2]
REPO = PLUGIN_DIR.parent


@pytest.fixture
def plugin_pkg(tmp_path, monkeypatch):
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_db = lambda pid: sqlite3.connect(":memory:", check_same_thread=False)
    storage.plugin_data_dir = lambda pid: tmp_path
    monkeypatch.setitem(sys.modules, "plugins", types.ModuleType("plugins"))
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)
    monkeypatch.syspath_prepend(str(REPO))
    return importlib.import_module("plugin")


class FakeCtx:
    def __init__(self):
        self.hooks, self.commands = {}, {}

    def get_config(self, key=None, default=None):
        return default

    def register_hook(self, name, fn):
        self.hooks[name] = fn

    def register_command(self, name, fn, description=""):
        self.commands[name] = fn

    def __getattr__(self, name):  # anything else the L3 client probes for
        raise AttributeError(name)


def _yaml_list(text, key):
    block = re.search(rf"^\s*{key}:\s*\n((?:\s*- .+\n)+)", text, re.M).group(1)
    return [line.split("-", 1)[1].split("#")[0].strip() for line in block.strip().splitlines()]


def test_register_wires_every_hook_and_the_command(plugin_pkg):
    ctx = FakeCtx()
    plugin_pkg.register(ctx)
    assert set(ctx.hooks) == set(plugin_pkg.HOOKS)
    assert all(callable(fn) for fn in ctx.hooks.values())
    assert "forecast" in ctx.commands


def test_manifests_agree_with_the_code(plugin_pkg):
    manifest = (PLUGIN_DIR / "plugin.yaml").read_text()
    catalog = (REPO / "docs" / "submission.md").read_text()
    assert set(_yaml_list(manifest, "provides_hooks")) == set(plugin_pkg.HOOKS)
    assert set(_yaml_list(catalog, "provides_hooks")) == set(plugin_pkg.HOOKS)
    version = re.search(r"^version:\s*(\S+)", manifest, re.M).group(1)
    assert re.search(rf'^version:\s*"{re.escape(version)}"', catalog, re.M)
