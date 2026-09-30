from .const import S3GEN_SR

__all__ = ["S3GEN_SR", "S3Gen"]


def __getattr__(name):
    if name != "S3Gen":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .s3gen import S3Token2Wav
    globals()[name] = S3Token2Wav
    return S3Token2Wav


def __dir__():
    return sorted(set(globals()) | set(__all__))
