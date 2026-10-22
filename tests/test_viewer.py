"""Run the viewer's own tests under the repository's unittest discovery.

The viewer keeps its fixtures next to the app (viewer/tests) so it can be
used on its own. This shim loads that module under a distinct name so
both files can be named test_viewer.py without colliding.
"""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / 'viewer' / 'tests' / 'test_viewer.py'
_spec = importlib.util.spec_from_file_location('viewer_tests', _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

IndexTests = _mod.IndexTests
ApiTests = _mod.ApiTests
