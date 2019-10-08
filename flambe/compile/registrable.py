from typing import Type, TypeVar, Callable, Mapping, Dict, List, Any, Optional, Set, NamedTuple, Sequence, Iterable, Union
from abc import abstractmethod, ABC
from collections import defaultdict
from warnings import warn
import functools
import logging
import inspect

from ruamel.yaml import YAML, ScalarNode

from flambe.compile.common import Singleton


ROOT_NAMESPACE = ''


class RegistryEntry(NamedTuple):
    """Entry in the registry representing a class, tags, factories"""

    callable: Callable
    default_tag: str
    aliases: List[str]
    factories: List[str]
    from_yaml: Callable  # TODO tighten interface
    to_yaml: Callable


class RegistrationError(Exception):
    """Thrown when invalid input is being registered"""

    pass


# Maps from class to registry entry, which contains the tags, aliases,
# and factory methods
SubRegistry = Dict[Type, RegistryEntry]


class Registry(metaclass=Singleton):

    def __init__(self):
        self.namespaces: Dict[str, SubRegistry] = defaultdict(SubRegistry)
        self.callable_to_namespaces: Dict[type, Sequence[str]] = defaultdict(list)

    def create(self,
               callable: Callable,
               namespace: str = ROOT_NAMESPACE,
               tags: Optional[Union[str, List[str]]] = None,
               factories: Optional[Sequence[str]] = None,
               from_yaml: Optional[Callable] = None,
               to_yaml: Optional[Callable] = None):
        if callable in self.callable_to_namespaces and \
                namespace in self.callable_to_namespaces[callable]:
            raise RegistrationError(f"Can't create entry for existing callable {callable.__name__}"
                                    f"in namespace {namespace}. Try updating instead")
        if tags is not None and isinstance(tags, list) and len(tags) < 1:
            raise ValueError('At least one tag must be specified. If the default (class name) '
                             'desired, pass in nothing or None.')
        if factories is not None and len(factories) < 1:
            raise ValueError('At least one factory must be specified if any are given.')
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
        self.callable_to_namespaces[callable].append(namespace)

    def read(self):
        pass

    def add_tag(self,
                class_: Type,
                tag: str,
                namespace: str = ROOT_NAMESPACE):
        try:
            self.namespaces[namespace][class_].tags.append(tag)
        except KeyError:
            pass

    def add_factory(self,
                    class_: Type,
                    factory: str,
                    namespace: str = ROOT_NAMESPACE):
        try:
            self.namespaces[namespace][class_].factories.append(factory)
        except KeyError:
            pass

    def delete(self, callable: Callable, namespace: Optional[str] = None) -> bool:
        if namespace is not None:
            if callable not in self.class_to_namespaces:
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
        for class_, namespace in self.class_to_namespace.items():
            yield self.namespaces[namespace][class_]


def get_registry():
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


def register(cls: Type[A],
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


class Registrable:
    """Subclasses automatically registered as yaml tags

    Automatically registers subclasses with the yaml loader by
    adding a constructor and representer which can be overridden
    """

    def __init_subclass__(cls: Type['Registrable'],
                          should_register: Optional[bool] = True,
                          tag_override: Optional[str] = None,
                          from_yaml: Optional[Callable] = None,
                          to_yaml: Optional[Callable] = None,
                          **kwargs: Mapping[str, Any]) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore
        if should_register:
            register(cls, tag_override, from_yaml, to_yaml)


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


class MappedRegistrable(Registrable):

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        return representer.represent_mapping(tag, node._saved_kwargs)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> Any:
        """Use constructor to create an instance of cls"""
        if inspect.isabstract(cls):
            msg = f"You're trying to initialize an abstract class {cls.__name__}. " \
                  + "If you think it's concrete, double check you've spelled " \
                  + "all the originally abstract method names correctly."
            raise Exception(msg)
        if isinstance(node, ScalarNode):
            nothing = constructor.construct_yaml_null(node)
            if nothing is not None:
                warn(f"Non-null scalar argument to {cls.__name__} will be ignored. A map of kwargs"
                     " should be used instead.")
            return cls()
        # NOTE: construct_yaml_map is a generator that yields the
        # constructed data and then updates it
        kwargs, = list(constructor.construct_yaml_map(node))
        if factory_name is not None:
            factory_method = getattr(cls, factory_name)
        else:
            factory_method = cls
        instance = factory_method(**kwargs)
        instance._saved_kwargs = kwargs
        return instance


def from_yaml(constructor: Any, node: Any, factory_name: str) -> Any:
    """Use constructor to create an instance of cls"""
    pass


def to_yaml(representer: Any, node: Any, tag: str) -> Any:
    """Use representer to create yaml representation of node"""
    pass


def transform_to(to_yaml_fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(to_yaml_fn)
    def wrapped(representer: Any, node: Any) -> Any:
        if hasattr(node, '_created_with_tag'):
            tag = node._created_with_tag
        else:
            tag = Registrable.get_default_tag(type(node))
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


def load_config(yaml_config: Union[stream, str]) -> Any:
    yaml = YAML()
    sync_registry_with_yaml(yaml, get_registry())
    result = yaml.load(yaml_config)
    return result


def dump_to_config(obj: Any, stream):
    yaml = YAML()
    sync_registry_with_yaml(yaml, get_registry())
    yaml.dump(obj, stream)
