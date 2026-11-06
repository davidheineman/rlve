__all__ = ["parse_args"]


def __getattr__(name):
    # CPU-only dataset/reward clients can import utilities without importing
    # the training CLI and its SGLang/CUDA dependencies.
    if name == "parse_args":
        from slime.utils.arguments import parse_args

        return parse_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
