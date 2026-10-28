"""Import every package module so syntax or import regressions fail CI."""

import importlib
import pkgutil

import proteus


def test_all_package_modules_import():
    modules = [m.name for m in pkgutil.walk_packages(proteus.__path__, "proteus.")]
    assert modules, "no package modules found"

    failures = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - failure path
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    assert not failures, "failed imports:\n" + "\n".join(failures)
