"""Photon Spectrum (iMessage) platform plugin entry point."""


def register(ctx) -> None:
    # Keep package import light so deferred client-tool discovery can load
    # tools.py without importing the adapter and its gateway dependencies.
    from .adapter import register as register_platform

    register_platform(ctx)


__all__ = ["register"]
