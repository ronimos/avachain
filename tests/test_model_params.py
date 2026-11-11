"""model_params is the one home of the release-model parameters (2026-11-11).

Three ways a moved parameter silently stops reaching the physics, each pinned:
  * a module redefines it (a second copy that calibration.apply never sets);
  * a function binds it as a default argument (frozen at import, the bug that
    made [calibration] c_fit a no-op until 2026-11-08);
  * `from model_params import X` (frozen at import).
And the values must match the standalone's central config (CLAUDE.md:
behavioural differences from ~/avalanche-release-areas are porting defects).
"""
import ast
import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "avachain"
sys.path.insert(0, str(SRC))
import model_params as mp   # noqa: E402

MP_NAMES = {n for n in vars(mp) if re.fullmatch(r"[A-Z][A-Z0-9_]*", n)}
STANDALONE_CONFIG = Path.home() / "avalanche-release-areas" / "src" / "release_areas" / "config.py"


def _modules():
    return [p for p in sorted(SRC.glob("*.py")) if p.name != "model_params.py"]


def test_model_params_imports_nothing_from_the_project():
    tree = ast.parse((SRC / "model_params.py").read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_redefines_a_model_parameter(path):
    tree = ast.parse(path.read_text())
    clash = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else \
            [node.target] if isinstance(node, ast.AnnAssign) else []
        clash |= {t.id for t in targets if isinstance(t, ast.Name)} & MP_NAMES
    assert not clash, f"{path.name} redefines {sorted(clash)}; use mp.<name>"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_model_parameter_is_frozen_at_import(path):
    tree = ast.parse(path.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "model_params":
            bad.append(f"line {node.lineno}: from model_params import ...")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
                for x in ast.walk(d):
                    if (isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
                            and x.value.id in ("mp", "_mp")):
                        bad.append(f"line {node.lineno}: default {ast.unparse(d)}")
    assert not bad, f"{path.name}: " + "; ".join(bad)


@pytest.mark.skipif(not STANDALONE_CONFIG.exists(), reason="standalone repo not checked out")
def test_values_match_the_standalone_config():
    ns = {}
    tree = ast.parse(STANDALONE_CONFIG.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in MP_NAMES:
            try:
                ns[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass   # derived (e.g. THETA_FIT_RADIUS_M = L_SS); covered by its source
    assert len(ns) > 40, "parsed too few standalone parameters"
    diff = {k: (getattr(mp, k), v) for k, v in ns.items() if getattr(mp, k) != v}
    assert not diff, f"avachain vs standalone: {diff}"
