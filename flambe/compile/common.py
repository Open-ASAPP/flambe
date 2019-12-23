import inspect
from typing import Dict, Callable, Any


_EMPTY = inspect.Parameter.empty


class Singleton(type):

    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


def function_defaults(function: Callable[..., Any]) -> Dict[str, Any]:
    """Use function signature to add missing kwargs to a dictionary"""
    signature = inspect.signature(function)
    defaults = {}
    for name, param in signature.parameters.items():
        if name == "self" or name == "cls":
            continue
        default = param.default
        if default != _EMPTY:
            defaults[name] = default
    return defaults
