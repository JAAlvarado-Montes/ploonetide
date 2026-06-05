"""Lightweight package metadata checks."""

from ploonetide.version import __version__


def test_version_is_declared():
    """The package should expose a non-empty version string."""
    assert isinstance(__version__, str)
    assert __version__
