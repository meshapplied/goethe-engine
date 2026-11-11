__version__ = "0.1.0"
_LAZY = {"Backend": "backend", "KVGeometry": "backend", "Work": "backend", "Engine": "engine",
         "RequestOutput": "engine", "SamplingParams": "sampling", "Config": "scheduler"}


def __getattr__(name):   # lazy: the index and pack CLIs must not import torch
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
