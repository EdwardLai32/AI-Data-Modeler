"""Test package.

Present so the test modules form a real package and can share the harness in
``conftest.py`` via ``from .conftest import ...`` — the fake LLM client and the
schema autofill are imported as library code, not only injected as fixtures.
"""
