"""Loading a custom pack (`module:Class` from rlcd.yaml) and the note about feedback a pack did not use."""
from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from open_rlcd.core.config import load_project
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.head import HeadSpec
from open_rlcd.labeling.store import LabelItem
from open_rlcd.packs.base import BasePack, dataset_from_labels, load_pack
from open_rlcd.packs.tabular import TabularPack
from tests.toypack import ToyPack, ToyProvider

PACK = """
from open_rlcd.packs.base import BasePack
{extra}

class Pack(BasePack):
    name = "{name}"
"""


def project_with_pack(tmp_path, module: str, package: bool = False, extra: str = "") -> object:
    root = tmp_path / "project"
    root.mkdir()
    code = textwrap.dedent(PACK).format(name=module, extra=extra)
    if package:
        (root / module).mkdir()
        (root / module / "__init__.py").write_text("")
        (root / module / "pack.py").write_text(code)
        (root / module / "helper.py").write_text("VALUE = 7\n")
        target = f"{module}.pack:Pack"
    else:
        (root / f"{module}.py").write_text(code)
        target = f"{module}:Pack"
    (root / "rlcd.yaml").write_text(f"pack: {target}\n")
    return load_project(root)


@pytest.fixture
def clean_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("rlcdtest_"):
            del sys.modules[name]


def test_a_pack_next_to_rlcd_yaml_loads_with_the_project_root(tmp_path, clean_modules):
    project = project_with_pack(tmp_path, "rlcdtest_single")
    path = list(sys.path)
    pack = load_pack(project.pack, root=project.root)
    assert pack.name == "rlcdtest_single" and isinstance(pack, BasePack)
    assert sys.path == path                                  # the folder was importable only during the import
    assert sys.modules["rlcdtest_single"].__file__.startswith(str(project.root))


def test_a_pack_package_keeps_importing_its_own_modules_afterwards(tmp_path, clean_modules):
    extra = "def late():\n    from .helper import VALUE\n    return VALUE"
    project = project_with_pack(tmp_path, "rlcdtest_pkg", package=True, extra=extra)
    load_pack(project.pack, root=project.root)
    assert str(project.root) not in sys.path
    assert sys.modules["rlcdtest_pkg.pack"].late() == 7      # a lazy relative import, after sys.path was restored


def test_without_a_root_the_nearest_rlcd_yaml_is_the_project(tmp_path, monkeypatch, clean_modules):
    project = project_with_pack(tmp_path, "rlcdtest_cwd")
    with pytest.raises(ImportError, match="next to rlcd.yaml"):
        load_pack(project.pack)                              # the tests' own folder has no such module
    monkeypatch.chdir(project.root)
    assert load_pack(project.pack).name == "rlcdtest_cwd"    # as `cd project && rlcd train` does


def test_rlcd_project_names_the_project_too(tmp_path, monkeypatch, clean_modules):
    project = project_with_pack(tmp_path, "rlcdtest_env")
    monkeypatch.setenv("RLCD_PROJECT", str(project.root))
    assert load_pack(project.pack).name == "rlcdtest_env"


def test_the_project_folder_never_shadows_an_installed_module(tmp_path, clean_modules):
    project = project_with_pack(tmp_path, "rlcdtest_shadow", extra="import yaml\nYAML = yaml.__file__")
    (project.root / "yaml.py").write_text("raise RuntimeError('a project file shadowed an installed package')\n")
    load_pack(project.pack, root=project.root)
    assert not sys.modules["rlcdtest_shadow"].YAML.startswith(str(project.root))


def test_missing_module_and_missing_class_say_where_they_looked(tmp_path, clean_modules):
    project = project_with_pack(tmp_path, "rlcdtest_missing")
    with pytest.raises(ImportError, match=f"looked in {project.root}"):
        load_pack("rlcdtest_nowhere:Pack", root=project.root)
    with pytest.raises(ImportError, match="has no 'Other'"):
        load_pack("rlcdtest_missing:Other", root=project.root)


def test_builtin_packs_load_as_before(tmp_path):
    assert isinstance(load_pack("tabular"), TabularPack)
    assert isinstance(load_pack("tabular", root=tmp_path), TabularPack)
    with pytest.raises(ValueError, match="unknown pack"):
        load_pack("trading", root=tmp_path)


# ----------------------------------------------------------------------------- feedback a pack did not use

SPEC = HeadSpec("good", "noul", view="v")
FEEDBACK = [{"id": "f1", "decision_id": "d1", "head": "good", "label": True, "state": {"f": [1.0] * 6}},
            {"id": "f2", "decision_id": "d2", "head": "good", "label": False, "state": {"f": [0.0] * 6}},
            {"id": "f3", "decision_id": "d3", "head": "other", "label": True, "state": {"f": [0.5] * 6}}]


def labelled(n: int = 8) -> list[dict]:
    rng = np.random.default_rng(0)
    return [{"item": LabelItem(id=f"i{i}", view="v", state={"f": rng.normal(size=6).tolist()}),
             "label": bool(i % 2), "source": "human"} for i in range(n)]


class IgnoresFeedback(BasePack):
    name = "ignores"

    def heads(self, project):
        return {"good": SPEC}

    def provider(self, project, config=None):
        return ToyProvider()

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        rows = labelled()
        X = provider.transform("v", [r["item"].state for r in rows])
        return {h: Dataset(X=X, y=[int(r["label"]) for r in rows], feature_names=provider.names("v"),
                           feature_version=provider.feature_version) for h in heads}


class UsesFeedback(IgnoresFeedback):
    name = "uses"

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        return {h: dataset_from_labels(SPEC, provider, labelled(), feedback, feedback_weight) for h in heads}


class ExplainsItself(IgnoresFeedback):
    name = "explains"

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        out = super().datasets(project, provider, heads, feedback, feedback_weight)
        for ds in out.values():
            ds.notes = ["feedback rows are not used by this pack: outcomes arrive through the nightly export"]
        return out


def test_a_pack_that_drops_feedback_gets_a_note():
    pack = IgnoresFeedback()
    for call in (lambda: pack.datasets(None, ToyProvider(), ["good"], FEEDBACK, 5.0),
                 lambda: pack.datasets(None, ToyProvider(), ["good"], feedback=FEEDBACK)):
        notes = call()["good"].notes
        assert len(notes) == 1 and notes[0].startswith("2 feedback rows for head 'good'")
        assert "IgnoresFeedback.datasets()" in notes[0] and "dataset_from_labels" in notes[0]
    assert pack.datasets(None, ToyProvider(), ["good"])["good"].notes == []          # no feedback, no note
    assert pack.datasets(None, ToyProvider(), ["good"], FEEDBACK[2:])["good"].notes == []   # none for this head


def test_a_pack_that_uses_or_explains_feedback_gets_none():
    ds = UsesFeedback().datasets(None, ToyProvider(), ["good"], FEEDBACK, 5.0)["good"]
    assert ds.source.tolist().count("feedback") == 2
    assert ds.notes == ["2 feedback rows added to training (weight x 5)"]
    notes = ExplainsItself().datasets(None, ToyProvider(), ["good"], FEEDBACK, 5.0)["good"].notes
    assert notes == ["feedback rows are not used by this pack: outcomes arrive through the nightly export"]
    state = {"f": [0.0] * 6}
    toy = ToyPack(n=30).datasets(None, ToyProvider(), ["good"], [{**FEEDBACK[0], "state": state}], 5.0)["good"]
    assert sum("feedback" in n for n in toy.notes) == 1
