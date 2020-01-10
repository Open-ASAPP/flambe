from typing import Type, TypeVar, Callable, Mapping, Dict, List, Any, Optional, Set, NamedTuple, Sequence, Iterable, Union
from abc import abstractmethod, ABC
from collections import defaultdict
import functools
import logging
import inspect
import pprint

from ruamel.yaml import YAML

from flambe.compile.common import Singleton


ROOT_NAMESPACE = ''


class RegistryEntry(NamedTuple):
    """Entry in the registry representing a callable, tags, factories"""

    callable: Callable
    default_tag: str
    tags: List[str]
    factories: List[str]
    from_yaml: Callable  # TODO tighten interface
    to_yaml: Callable


class RegistrationError(Exception):
    """Thrown when invalid input is being registered"""

    pass


# Maps from class to registry entry, which contains the tags, aliases,
# and factory methods
SubRegistry = Dict[Callable, RegistryEntry]


class Registry(metaclass=Singleton):

    def __init__(self):
        self.namespaces: Dict[str, SubRegistry] = defaultdict(dict)
        self.callable_to_namespace: Dict[Callable, str] = {}

    def create(self,
               callable: Callable,
               namespace: str = ROOT_NAMESPACE,
               tags: Optional[Union[str, List[str]]] = None,
               factories: Optional[Sequence[str]] = None,
               from_yaml: Optional[Callable] = None,
               to_yaml: Optional[Callable] = None):
        if callable in self.callable_to_namespace and \
                namespace == self.callable_to_namespace[callable]:
            raise RegistrationError(f"Can't create entry for existing callable {callable.__name__}"
                                    f"in namespace {namespace}. Try updating instead")
        if tags is not None and isinstance(tags, list) and len(tags) < 1:
            raise ValueError('At least one tag must be specified. If the default (class name) '
                             'desired, pass in nothing or None.')
        if factories is not None:
            if len(factories) < 1:
                raise ValueError('At least one factory must be specified if any are given.')
            if not isinstance(callable, type):
                raise ValueError('Callable must be a class if factories are specified')
        try:
            tags = tags or [callable.__name__]
        except AttributeError:
            raise ValueError(f'Tags argument not given and callable argument {callable} '
                             'has no __name__ property.')
        missing_methods_err = (f'Missing `from_yaml` and or `to_yaml` and given callable '
                               f'{callable} is not a Class with these methods')
        try:
            if not isinstance(callable, type):
                raise ValueError(missing_methods_err)
            from_yaml = from_yaml or callable.from_yaml
            to_yaml = to_yaml or callable.to_yaml
        except AttributeError:
            raise ValueError(missing_methods_err)
        tags = [tags] if isinstance(tags, str) else tags
        default_tag = tags[0]
        factories = factories or []
        new_entry = RegistryEntry(callable, default_tag, tags, factories, from_yaml, to_yaml)
        self.namespaces[namespace][callable] = new_entry
        self.callable_to_namespace[callable] = namespace

    def default_tag(self, callable: Callable) -> str:
        namespace = self.callable_to_namespace[callable]
        entry = self.namespaces[namespaces][callable]
        return entry.default_tag

    def add_tag(self,
                callable: Callable,
                tag: str,
                namespace: str = ROOT_NAMESPACE):
        try:
            self.namespaces[namespace][callable].tags.append(tag)
        except KeyError:
            pass

    def add_factory(self,
                    callable: Callable,
                    factory: str,
                    namespace: str = ROOT_NAMESPACE):
        try:
            self.namespaces[namespace][callable].factories.append(factory)
        except KeyError:
            pass

    def delete(self, callable: Callable, namespace: Optional[str] = None) -> bool:
        if namespace is not None:
            if callable not in self.callable_to_namespace:
                return False
            if namespace not in self.namespaces:
                return False
            if namespace not in self.class_to_namespaces[callable]:
                return False
            try:
                del self.namespaces[namespace][callable]
            except KeyError:
                raise RegistrationError('Invalid registry state')
            del self.class_to_namespace[callable]
            return True
        else:
            count = 0
            for sub_registry in self.namespaces.values():
                if callable in sub_registry:
                    del sub_registry[callable]
                    count += 1
            if count == 0:
                return False
            if count >= 1:
                if callable not in self.class_to_namespaces:
                    raise RegistrationError('Invalid registry state')
                del self.class_to_namespace[callable]
                return True

    def __iter__(self) -> Iterable[RegistryEntry]:
        for callable, namespace in self.callable_to_namespace.items():
            # print(namespace)
            # print(self.callable_to_namespace)
            # print(self.pretty_str())
            # print(self.namespaces[namespace][callable])
            yield (namespace, self.namespaces[namespace][callable])

    def pretty_str(self) -> str:
        s = ['Flambe Registry']
        for namespace in self.namespaces:
            if namespace == ROOT_NAMESPACE:
                s.append(f"  ROOT:")
            else:
                s.append(f"  '{namespace}':")
            for entry in self.namespaces[namespace].values():
                s.append(f"    '{entry.default_tag}'")
                s.append(f"      callable: {entry.callable}")
                s.append(f"      factories:")
                for factory in entry.factories:
                    s.append(f"        {factory}")
        s = '\n'.join(s)
        return s


def get_registry():
    """Future proof entry point for registry access

    Eventually we might want to refactor again or even sync registry
    across machines, support multi-threading etc. so that logic can
    appear here.

    """
    return Registry()


A = TypeVar('A')


def get_class_namespace(class_: Type):
    modules = class_.__module__.split('.')
    top_level_module_name = modules[0] if len(modules) > 0 else None
    if top_level_module_name is not None and \
            (top_level_module_name != 'flambe' and top_level_module_name != 'tests'):
        return top_level_module_name
    else:
        return ROOT_NAMESPACE


def register_class(cls: Type[A],
             tag: str,
             from_yaml: Optional[Callable] = None,
             to_yaml: Optional[Callable] = None) -> Type[A]:
    """Safely register a new tag for a class"""
    tag_namespace = get_class_namespace(cls)
    get_registry().create(cls, tag_namespace, tag, from_yaml=from_yaml, to_yaml=to_yaml)
    return cls


RT = TypeVar('RT', bound=Type['Registrable'])


def alias(tag: str) -> Callable[[RT], RT]:
    """Decorate a registered class with a new tag

    Can be added multiple times to give a class multiple aliases,
    however the top most alias tag will be the default tag which means
    it will be used when representing the class in YAML

    """

    def decorator(cls: RT) -> RT:
        namespace = get_class_namespace(cls)
        get_registry().add_tag(cls, tag, namespace)
        return cls

    return decorator


class registrable_factory:
    """Decorate Registrable factory method for use in the config

    This Descriptor class will set properties that allow the factory
    method to be specified directly in the config as a suffix to the
    tag; for example:

    .. code-block:: python

        class MyModel(Component):

            @registrable_factory
            def from_file(cls, path):
                # load instance from path
                ...
                return instance

    defines the factory, which can then be used in yaml:

    .. code-block:: yaml

        model: !MyModel.from_file
            path: some/path/to/file.pt

    """

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __set_name__(self, owner: type, name: str) -> None:
        namespace = get_class_namespace(owner)
        try:
            get_registry().add_factory(owner, name, namespace)
        except:
            pass
        setattr(owner, name, self.fn)
