"""Photon Spectrum (iMessage) platform plugin entry point."""


def register(ctx) -> None:
    # Keep package import light so deferred client-tool discovery can load
    # tools.py without importing the adapter and its gateway dependencies.
    from .adapter import register as register_platform
    from .conversation_actions import suppress_redundant_final

    register_platform(ctx)
    ctx.register_hook("transform_llm_output", suppress_redundant_final)


__all__ = ["register"]
