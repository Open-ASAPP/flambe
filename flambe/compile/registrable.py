from typing import Type, TypeVar, Callable, Mapping, Dict, List, Any, Optional, Set
from abc import abstractmethod, ABC
from collections import defaultdict
from warnings import warn
import functools
import logging
import inspect

from ruamel.yaml import YAML, ScalarNode


logger = logging.getLogger(__name__)

yaml = YAML()

_reg_prefix: Optional[str] = None

R = TypeVar('R', bound='Registrable')
A = TypeVar('A')
RT = TypeVar('RT', bound=Type['Registrable'])


class RegistrationError(Exception):
    """Error thrown when acessing yaml tag on a non-registered class

    Thrown when trying to access the default yaml tag for a class
    typically occurs when called on an abstract class
    """

    pass


def make_from_yaml_with_metadata(from_yaml_fn: Callable[..., Any],
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


def make_to_yaml_with_metadata(to_yaml_fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(to_yaml_fn)
    def wrapped(representer: Any, node: Any) -> Any:
        if hasattr(node, '_created_with_tag'):
            tag = node._created_with_tag
        else:
            tag = Registrable.get_default_tag(type(node))
        return to_yaml_fn(representer, node, tag=tag)
    return wrapped


class registration_context:

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace

    def __enter__(self) -> None:
        global _reg_prefix
        self._prev_reg_prefix = _reg_prefix
        _reg_prefix = self._namespace

    def __exit__(self, *args: Any) -> int:
        global _reg_prefix
        _reg_prefix = self._prev_reg_prefix
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def decorate_reg_context(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)
        return decorate_reg_context


class Registrable(ABC):
    """Subclasses automatically registered as yaml tags

    Automatically registers subclasses with the yaml loader by
    adding a constructor and representer which can be overridden
    """

    _yaml_tags: Dict[Any, List[str]] = defaultdict(list)
    _yaml_tag_namespace: Dict[Type, str] = defaultdict(str)
    _yaml_registered_factories: Set[str] = set()

    def __init_subclass__(cls: Type[R],
                          should_register: Optional[bool] = True,
                          tag_override: Optional[str] = None,
                          tag_namespace: Optional[str] = None,
                          **kwargs: Mapping[str, Any]) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore
        # Copy parent set so that factories are inherited
        # But not shared across cousin classes
        cls._yaml_registered_factories = set(cls._yaml_registered_factories)
        if should_register:
            default_tag = cls.__name__ if tag_override is None else tag_override
            # NOTE: abstract classes are registered too. This allows us
            # to raise an exception if you actually try to use one,
            # in case you think a class should be concrete but is
            # actually still abstract
            Registrable.register_tag(cls, default_tag, tag_namespace)

    @staticmethod
    def register_tag(class_: RT, tag: str, tag_namespace: Optional[str] = None) -> None:
        modules = class_.__module__.split('.')
        top_level_module_name = modules[0] if len(modules) > 0 else None
        global _reg_prefix
        if _reg_prefix is not None:
            tag_namespace = _reg_prefix
        elif tag_namespace is not None:
            tag_namespace = tag_namespace
        elif (tag_namespace is None and top_level_module_name is not None) and \
                (top_level_module_name != 'flambe' and top_level_module_name != 'tests'):
            tag_namespace = top_level_module_name
        else:
            tag_namespace = None
        # Create a tag that includes namespace e.g. `!torch.Adam`
        if tag_namespace is not None:
            full_tag = f"!{tag_namespace}.{tag}"
        else:
            full_tag = f"!{tag}"
        # full_tag = f"!{tag_namespace}.{tag}" if tag_namespace is
        # not None else f"!{tag}"
        if class_ in class_._yaml_tag_namespace:
            if tag_namespace != class_._yaml_tag_namespace[class_]:
                # Don't register anything not matching the already set
                # namespace
                # Helps limit chance of tag collisions
                msg = (f"You are trying to register class {class_} with namespace "
                       f"{tag_namespace} != {class_._yaml_tag_namespace[class_]} "
                       "so ignoring")
                warn(msg)
                return
        elif tag_namespace is not None:
            # Set namespace so that the above branch can catch
            # accidentally forgetting namespace
            class_._yaml_tag_namespace[class_] = tag_namespace
        # Ensure all tags are only associated with that specific class,
        # NOT any subclasses
        class_._yaml_tags[class_].append(full_tag)
        # Code based on the ruamel.yaml yaml_object decorator
        # Look for to_yaml and from_yaml methods -- if not present
        # default to built in default flow style

        def registration_helper(factory_name: Optional[str] = None) -> None:
            from_yaml_tag = full_tag if factory_name is None else full_tag + "." + factory_name
            logger.debug(f"Registering tag: {from_yaml_tag}")
            try:
                to_yaml = class_.to_yaml
            except AttributeError:
                def t_y(representer: Any, node: Any, tag: str) -> Any:
                    return representer.represent_yaml_object(
                        tag, node, class_, flow_style=representer.default_flow_style
                    )
                to_yaml = t_y
            finally:
                yaml.representer.add_representer(class_, make_to_yaml_with_metadata(to_yaml))
            try:
                from_yaml = class_.from_yaml
            except AttributeError:
                def f_y(constructor: Any, node: Any, factory_name: str) -> Any:
                    return constructor.construct_yaml_object(node, class_)
                from_yaml = f_y
            finally:
                yaml.constructor.add_constructor(
                    from_yaml_tag,
                    make_from_yaml_with_metadata(from_yaml, from_yaml_tag, factory_name)
                )

        registration_helper()
        for factory_name in class_._yaml_registered_factories:
            # Add factory tag to registry
            factory_full_tag = f'{full_tag}.{factory_name}'
            class_._yaml_tags[(class_, factory_name)] = [factory_full_tag]
            # Every time we register a new tag, make sure that you can
            # use each factory with that new tag
            registration_helper(factory_name)

    @staticmethod
    def get_default_tag(class_: RT, factory_name: Optional[str] = None) -> str:
        """Retrieve default yaml tag for class `cls`

        Retrieve the default tag (aka the last one, which will
        be the only one, or the alias if it exists) for use in
        yaml representation
        """
        if class_ in class_._yaml_tags:
            tag = class_._yaml_tags[class_][-1]
            if (factory_name is not None) and \
                    (factory_name not in class_._yaml_registered_factories):
                raise RegistrationError(f"This class has no factory {factory_name}")
            elif factory_name is not None:
                tag = tag + '.' + factory_name
            return tag
        raise RegistrationError("This class has no registered tags")

    @classmethod
    @abstractmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node

        See Component class, and experiment/options for examples

        """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> Any:
        """Use constructor to create an instance of cls

        See Component class, and experiment/options for examples

        """
        pass


def alias(tag: str,
          tag_namespace: Optional[str] = None) -> Callable[[RT], RT]:
    """Decorate a Registrable subclass with a new tag

    Can be added multiple times to give a class multiple aliases,
    however the top most alias tag will be the default tag which means
    it will be used when representing the class in YAML

    """

    def decorator(cls: RT) -> RT:
        Registrable.register_tag(cls, tag, tag_namespace)
        return cls

    return decorator


def register(cls: Type[A], tag: str) -> Type[A]:
    """Safely register a new tag for a class

    Similar to alias, but it's intended to be used on classes that are
    not already subclasses of Registrable, and it is NOT a decorator

    """
    if not hasattr(cls, '_yaml_tags'):
        cls._yaml_tags = defaultdict(list)  # type: ignore
    if not hasattr(cls, '_yaml_tag_namespace'):
        cls._yaml_tag_namespace = defaultdict(str)  # type: ignore
    if not hasattr(cls, '_yaml_registered_factories'):
        cls._yaml_registered_factories = set()  # type: ignore
    return alias(tag)(cls)  # type: ignore


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
        if not hasattr(owner, '_yaml_registered_factories'):
            raise RegistrationError(f"class {owner} doesn't have property "
                                    f"_yaml_registered_factories; {owner} should subclass "
                                    "Registrable or Component")
        owner._yaml_registered_factories.add(name)  # type: ignore
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
