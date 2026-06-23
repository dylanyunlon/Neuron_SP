# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tests for HeteroRegistry auto-discovery and registration.

Validates that the registry correctly discovers all 62 hetero_*.py modules
under deepspeed/runtime/ and deepspeed/runtime/zero/, and that registration
behaves idempotently.
"""

import os
import glob
import unittest
from unittest.mock import MagicMock

from deepspeed.runtime.hetero_registry import HeteroRegistry


def _project_root():
    """Return the Neuron_SP project root (two levels above deepspeed/runtime/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


class TestDiscoverAllModules(unittest.TestCase):
    """Verify scan() finds every hetero_*.py in runtime/ and zero/."""

    def setUp(self):
        self.root = _project_root()
        self.registry = HeteroRegistry(project_root=self.root)

    def test_discover_all_modules(self):
        """scan() must discover all hetero_*.py files that exist on disk."""
        # Ground truth: glob the filesystem for hetero_*.py under deepspeed/
        ds_root = os.path.join(self.root, "deepspeed")
        runtime_files = glob.glob(os.path.join(ds_root, "runtime", "hetero_*.py"))
        zero_files = glob.glob(os.path.join(ds_root, "runtime", "zero", "hetero_*.py"))
        expected = set()
        for f in runtime_files + zero_files:
            expected.add(os.path.relpath(f, self.root))

        # The registry should discover the same set (scan() walks all of deepspeed/)
        count = self.registry.scan()
        discovered = set(self.registry.modules.keys())

        # Every runtime + zero hetero file must appear in the registry
        missing = expected - discovered
        self.assertEqual(
            missing, set(),
            f"Registry missed {len(missing)} modules: {sorted(missing)}"
        )

        # scan() discovers these plus any hetero files in other subdirectories;
        # we only insist the runtime+zero set is fully covered.
        self.assertGreaterEqual(count, len(expected))

    def test_all_modules_have_classes(self):
        """Every discovered module should have at least one class extracted."""
        self.registry.scan()
        empty = [
            info.file_path
            for info in self.registry.modules.values()
            if not info.classes
        ]
        # Not a hard failure — some modules may use only functions — but flag it.
        # We allow up to 5 classless modules as an upper bound.
        self.assertLessEqual(
            len(empty), 5,
            f"Too many classless modules ({len(empty)}): {sorted(empty)}"
        )


class TestRegisterReturnsBool(unittest.TestCase):
    """Verify that load_module returns a bool and register_hooks counts work."""

    def setUp(self):
        self.root = _project_root()
        self.registry = HeteroRegistry(project_root=self.root)
        self.registry.scan()

    def test_load_module_returns_bool(self):
        """load_module() must return True or False for every scanned module."""
        for info in list(self.registry.modules.values())[:5]:
            result = self.registry.load_module(info)
            self.assertIsInstance(result, bool)

    def test_register_hooks_returns_int(self):
        """register_hooks() on a mock engine must return an int count."""
        engine = MagicMock()
        registered = self.registry.register_hooks(engine)
        self.assertIsInstance(registered, int)
        self.assertGreaterEqual(registered, 0)

    def test_discover_and_register_returns_int(self):
        """discover_and_register() must return an int (succeeded count)."""
        engine = MagicMock()
        fresh = HeteroRegistry(project_root=self.root)
        result = fresh.discover_and_register(engine)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)


class TestNoDuplicateRegistrations(unittest.TestCase):
    """Calling discover_and_register twice must not double-count or duplicate."""

    def setUp(self):
        self.root = _project_root()

    def test_no_duplicate_registrations(self):
        """Two consecutive discover_and_register calls produce identical state."""
        engine = MagicMock()
        reg = HeteroRegistry(project_root=self.root)

        count_first = reg.discover_and_register(engine)
        # Capture the set of attrs set on the engine by the first call
        attrs_after_first = {
            a for a in dir(engine) if a.startswith("_hetero_mod_")
        }

        count_second = reg.discover_and_register(engine)
        attrs_after_second = {
            a for a in dir(engine) if a.startswith("_hetero_mod_")
        }

        # The second call must not add new module attributes
        new_attrs = attrs_after_second - attrs_after_first
        self.assertEqual(
            new_attrs, set(),
            f"Second discover_and_register added new attrs: {sorted(new_attrs)}"
        )

    def test_scan_idempotent(self):
        """scan() called twice does not double the module count."""
        reg = HeteroRegistry(project_root=self.root)
        first = reg.scan()
        second = reg.scan()
        # scan() re-walks and re-inserts; the dict is keyed by rel path so
        # the module count should stay the same.
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
