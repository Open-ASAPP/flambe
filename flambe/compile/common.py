import inspect
from typing import Dict, Callable, Any, Sequence, List


_EMPTY = inspect.Parameter.empty


class Singleton(type):

    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
