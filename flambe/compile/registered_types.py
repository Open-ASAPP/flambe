import inspect
from typing import Optional, Callable, Mapping, Any, Type
from warnings import warn

from ruamel.yaml import ScalarNode

from flambe.compile.registry import register_class, transform_from_yaml, transform_to_yaml
from flambe.compile.schema import Schema


class Tagged:

    def __new__(self, *args, **kwargs):
        self = super().__new__(*args, **kwargs)
        self._created_with_tag = None


class Registrable(Tagged):
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
            register_class(cls, tag_override, from_yaml, to_yaml)


class RegisteredStatelessMap(Registrable, should_register=False):

    def __init__(self):
        self._saved_kwargs = None

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


class RegisteredMap(RegisteredStatelessMap, should_register=False):

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Represent all current values in __dict__"""
        return representer.represent_mapping(tag, node.__dict__)


class Schematic(Registrable):

    def __init_subclass__(cls: Type['Registrable'],
                          **kwargs: Mapping[str, Any]) -> None:
        from_yaml_fn = transform_from_yaml(Schema.from_yaml, callable=cls)
        to_yaml_fn = transform_to_yaml(Schema.to_yaml)
        super().__init_subclass__(from_yaml=from_yaml_fn, to_yaml=to_yaml_fn, **kwargs)
