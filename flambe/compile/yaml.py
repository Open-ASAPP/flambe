from typing import Callable, Optional, Any, Union, TextIO
import functools
from warnings import warn

import ruamel.yaml

from flambe.compile.registrable import get_registry
from flambe.compile.registered_types import Tagged


def from_yaml(constructor: Any, node: Any, factory_name: str) -> Any:
    """Use constructor to create an instance of cls"""
    pass


def to_yaml(representer: Any, node: Any, tag: str) -> Any:
    """Use representer to create yaml representation of node"""
    pass


def transform_to(to_yaml_fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(to_yaml_fn)
    def wrapped(representer: Any, node: Any) -> Any:
        if isinstance(node, Tagged):
            tag = node._created_with_tag
        else:
            warn("No tag recorded for {node}, using default instead. "
                 "This may be incorrect.")
            if isinstance(node, object):
                callable = type(node)
            elif isinstance(node, functools.partial):
                callable = node.func
            else:
                raise Exception()
            tag = get_registry().get_default_tag(callable)
        return to_yaml_fn(representer, node, tag=tag)
    return wrapped


def transform_from(from_yaml_fn: Callable[..., Any],
                   tag: str,
                   factory_name: Optional[str] = None) -> Callable[..., Any]:
    @functools.wraps(from_yaml_fn)
    def wrapped(constructor: Any, node: Any) -> Any:
        obj = from_yaml_fn(constructor, node, factory_name=factory_name)
        # Access dict directly because obj may be a Schema, and have
        # special dot notation access behavior
        obj.__dict__['_created_with_tag'] = tag
        return obj
    return wrapped


def combine(*args):
    return '!' + '.'.join(args)


def sync_registry_with_yaml(yaml, registry):
    for entry in registry:
        yaml.representer.add_representer(entry.class_, transform_to(entry.to_yaml))
        combos = [(tag, factory) for tag in entry.tags for factory in entry.factories]
        for tag, factory in combos:
            full_tag = combine(entry.namespace, tag, factory)
            yaml.constructor.add_constructor(full_tag,
                                             transform_from(entry.from_yaml, full_tag, factory))


def erase_registry_from_yaml(yaml, registry):
    pass  # TODO might need this because of the problems we've seen


class synced_yaml:

    def __init__(self, registry):
        self.registry = registry
        self.yaml = None

    def __enter__(self):
        self.yaml = ruamel.yaml.YAML()
        sync_registry_with_yaml(self.yaml, self.registry)
        return self.yaml

    def __exit__(self):
        erase_registry_from_yaml(self.yaml, self.registry)


# TODO should we tighten signatures (below) Any -> schema?


def load_config(yaml_config: Union[TextIO, str]) -> Any:
    with synced_yaml(get_registry()) as yaml:
        result = yaml.load(yaml_config)
    return result


def dump_config(obj: Any, stream):
    with synced_yaml(get_registry()) as yaml:
        yaml.dump(obj)
