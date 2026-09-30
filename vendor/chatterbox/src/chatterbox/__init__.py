try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")


_PUBLIC_MODULES = {
    "ChatterboxTTS": ".tts",
    "ChatterboxVC": ".vc",
    "ChatterboxMultilingualTTS": ".mtl_tts",
    "SUPPORTED_LANGUAGES": ".mtl_tts",
}
__all__ = list(_PUBLIC_MODULES)


def __getattr__(name):
    """Load only the requested public API, not every voice pipeline."""
    if name not in _PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(_PUBLIC_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC_MODULES))
