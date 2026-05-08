#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Минимальный раннер тестов из tests/test_core.py — работает без установленного
pytest. Реализует только те API pytest, которые используются в тестах:
    pytest.raises, pytest.approx, fixture tmp_path.

Для штатной разработки рекомендуется устанавливать настоящий pytest.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import pathlib
import sys
import tempfile
import traceback


# -----------------------------------------------------------------------------
# Заглушка для pytest.
# -----------------------------------------------------------------------------

class _RaisesCM:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"Ожидалось исключение {self.exc.__name__}, но ничего не выброшено")
        return issubclass(exc_type, self.exc)


class _Approx:
    def __init__(self, value, abs_tol=None, rel_tol=None):
        self.value = value
        self.abs_tol = abs_tol if abs_tol is not None else 1e-9
        self.rel_tol = rel_tol if rel_tol is not None else 1e-7

    def __eq__(self, other):
        diff = abs(other - self.value)
        if diff <= self.abs_tol:
            return True
        return diff <= self.rel_tol * abs(self.value)

    def __repr__(self):
        return f"approx({self.value}, abs={self.abs_tol})"


class _PytestStub:
    @staticmethod
    def raises(exc):
        return _RaisesCM(exc)

    @staticmethod
    def approx(value, abs=None, rel=None):
        return _Approx(value, abs_tol=abs, rel_tol=rel)


def _install_pytest_stub():
    sys.modules["pytest"] = _PytestStub()


# -----------------------------------------------------------------------------
# Загрузка модуля тестов.
# -----------------------------------------------------------------------------

def _load_test_module():
    here = os.path.abspath(os.path.dirname(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    test_path = os.path.join(here, "tests", "test_core.py")
    spec = importlib.util.spec_from_file_location("test_core", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Запуск.
# -----------------------------------------------------------------------------

def main() -> int:
    _install_pytest_stub()
    module = _load_test_module()

    passed = 0
    failed = 0
    failures = []

    for cls_name, cls in inspect.getmembers(module, inspect.isclass):
        if not cls_name.startswith("Test"):
            continue
        print(f"\n=== {cls_name} ===")
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            sig = inspect.signature(fn)
            kwargs = {}
            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = pathlib.Path(tempfile.mkdtemp())
            try:
                instance = cls()
                fn(instance, **kwargs)
                print(f"  PASS  {name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {name}")
                failures.append((cls_name, name, traceback.format_exc()))
                failed += 1

    print(f"\nИтого: {passed} пройдено, {failed} упало.")
    if failures:
        print("\n--- Детали ошибок ---")
        for cls_name, name, tb in failures:
            print(f"\n{cls_name}.{name}:\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
