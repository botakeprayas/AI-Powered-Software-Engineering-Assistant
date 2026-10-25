"""Dynamic plugin loading (cannot be resolved statically)."""
import importlib


def run_plugin(name, action):
    """Call a function whose name is only known at runtime."""
    module = importlib.import_module(f"plugins.{name}")
    handler = getattr(module, action)
    return handler()