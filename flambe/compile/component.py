# from __future__ import annotations
import inspect
import dill
import logging
from reprlib import recursive_repr
from warnings import warn
from typing import Type, TypeVar, Any, Mapping, Dict, Optional, List, Union
from typing import Generator, MutableMapping, Callable, Set, Tuple, Sequence
from functools import WRAPPER_ASSIGNMENTS
from collections import OrderedDict
import copy

import ray
import torch
from io import StringIO
from ruamel.yaml.representer import RepresenterError
from ruamel.yaml import ScalarNode
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
                                  CommentedKeySeq, CommentedSeq, TaggedScalar,
                                  CommentedKeyMap)

from flambe.compile.serialization import load_state_from_file, State, load as flambe_load, \
    save as flambe_save
from flambe.compile.registrable import Registrable, alias, yaml, registrable_factory
from flambe.compile.const import STATE_DICT_DELIMETER, FLAMBE_SOURCE_KEY, FLAMBE_CLASS_KEY, \
    FLAMBE_CONFIG_KEY, FLAMBE_DIRECTORIES_KEY, KEEP_VARS_KEY, VERSION_KEY, FLAMBE_STASH_KEY


_EMPTY = inspect.Parameter.empty
A = TypeVar('A')
C = TypeVar('C', bound="Component")

YAML_TYPES = (CommentedMap, CommentedOrderedMap, CommentedSet, CommentedKeySeq, CommentedSeq,
              TaggedScalar, CommentedKeyMap)

logger = logging.getLogger(__name__)


class CompilationError(Exception):
    pass


class LoadError(Exception):
    pass


class Schema(MutableMapping[str, Any]):
    """Holds and recursively initializes Component's with kwargs

    Holds a Component subclass and keyword arguments to that class's
    compile method. When an instance is called it will perform the
    recursive compilation process, turning the nested structure of
    Schema's into initialized Component objects

    Implements MutableMapping methods to facilitate inspection and
    updates to the keyword args. Implements dot-notation access to
    the keyword args as well.

    Parameters
    ----------
    component_subclass : Type[Component]
        Subclass of Component that will be compiled
    **keywords : Any
        kwargs passed into the Schema's `compile` method

    Examples
    -------
    Create a Schema from a Component subclass

    >>> class Test(Component):
    ...     def __init__(self, x=2):
    ...         self.x = x
    ...
    >>> tp = Schema(Test)
    >>> t1 = tp()
    >>> t2 = tp()
    >>> assert t1 is t2  # the same Schema always gives you same obj
    >>> tp = Schema(Test) # create a new Schema
    >>> tp['x'] = 3
    >>> t3 = tp()
    >>> assert t1.x == 3  # dot notation works as well

    Attributes
    ----------
    component_subclass : Type[Component]
        Subclass of Schema that will be compiled
    keywords : Dict[str, Any]
        kwargs passed into the Schema's `compile` method

    """

    def __init__(self,
                 component_subclass: Type[C],
                 _flambe_custom_factory_name: Optional[str] = None,
                 **keywords: Any) -> None:
        # Check if `Component` instead of just checking if callable
        if not issubclass(component_subclass, Component):
            raise TypeError("The first argument must be Component")

        self.component_subclass: Type[C] = component_subclass
        self.factory_method: Optional[str] = _flambe_custom_factory_name
        self.keywords: Dict[str, Any] = keywords
        self.post_init_hooks: Sequence[Callable] = []
        self._compiled: Optional[C] = None
        self._extensions: Dict[str, str] = {}
        # Flag changes getattr functionality (for dot notation access)
        self._created: bool = True

    def __call__(self, stash: Optional[Dict[str, Any]] = None, **keywords: Any) -> C:
        # The same yaml node => the same Schema => the same object
        # So cache the compiled object
        if self._compiled is not None:
            return self._compiled
        newkeywords = self.keywords.copy()
        newkeywords.update(keywords)
        compiled = self.component_subclass.compile(
            _flambe_custom_factory_name=self.factory_method,
            _flambe_extensions=self._extensions,
            _flambe_stash=stash,
            **newkeywords)
        self._compiled = compiled
        compiled._schema = self  # type: ignore
        compiled._created_with_tag = self._created_with_tag  # type: ignore
        for hook in self.post_init_hooks:
            hook(compiled)
        return compiled

    def add_extensions_metadata(self, extensions: Dict[str, str]) -> None:
        """Add extensions used when loading this schema and children

        Uses ``component_subclass.__module__`` to filter for only the
        single relevant extension for this object; extensions relevant
        for children are saved only on those children schemas directly.
        Use ``aggregate_extensions_metadata`` to generate a dictionary
        of all extensions used in the object hierarchy.

        """
        # Get top level module
        modules = self.component_subclass.__module__.split('.')
        # None sentinel won't be in extensions
        top_level_module = modules[0] if len(modules) > 0 else None
        if top_level_module is not None and top_level_module in extensions:
            self._extensions = {top_level_module: extensions[top_level_module]}
        else:
            self._extensions = {}

        def helper(data):
            if isinstance(data, Schema):
                data.add_extensions_metadata(extensions)
            elif isinstance(data, list) or isinstance(data, tuple):
                for val in data:
                    helper(val)
            elif isinstance(data, Mapping):
                for val in data.values():
                    helper(val)

        for child in self.keywords.values():
            helper(child)

    def aggregate_extensions_metadata(self) -> Dict[str, str]:
        """Aggregate extensions used in object hierarchy"""
        exts = dict(self._extensions or {})  # non-nested so shallow copy ok

        def helper(data):
            if isinstance(data, Schema):
                exts.update(data.aggregate_extensions_metadata())
            elif isinstance(data, list) or isinstance(data, tuple):
                for val in data:
                    helper(val)
            elif isinstance(data, Mapping):
                for val in data.values():
                    helper(val)

        for child in self.keywords.values():
            helper(child)

        return exts

    def contains(self, schema: 'Schema', original_link: 'Link') -> Tuple[bool, List[str]]:
        if self is schema:
            return True, []

        def helper(current, schematic_path):
            if current is schema:
                schematic_path = [''] if len(schematic_path) == 0 else schematic_path
                return True, schematic_path
            for key, child in current.keywords.items():
                new_path = schematic_path[:] + [key]
                if isinstance(child, Schema):
                    present, temp = helper(current=child, schematic_path=new_path)
                    if present:
                        return present, temp
                if isinstance(child, Link) and child is not original_link:
                    resolved = child.resolved
                    if hasattr(resolved, '_schema'):
                        # Pass through the link to the original object
                        # TODO enforce links to earlier objects
                        present, temp = helper(current=child.resolved._schema,
                                               schematic_path=new_path)
                        if present:
                            return present, temp
                elif child is original_link:
                    return False, []
            return False, []

        return helper(self, [])

    def __setitem__(self, key: str, value: Any) -> None:
        self.keywords[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.keywords[key]

    def __delitem__(self, key: str) -> None:
        del self.keywords[key]

    def __iter__(self) -> Generator[str, None, None]:
        yield from self.keywords

    def __len__(self) -> int:
        return len(self.keywords)

    def __getattr__(self, item: str) -> Any:
        if '_created' not in self.__dict__ or item in self.__dict__:
            # TODO typing - not sure why __getattr__ isn't defined for
            # super, it works fine
            return super().__getattr__(self, item)  # type: ignore
        elif item in self.keywords:
            return self.__getitem__(item)
        else:
            raise AttributeError(f"Object {type(self).__name__}"
                                 f"[{self.component_subclass.__name__}] has no attribute {item}.")

    def __setattr__(self, key: str, value: Any) -> None:
        if '_created' not in self.__dict__ or key in self.__dict__:
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    @recursive_repr()
    def __repr__(self) -> str:
        """Identical to super (schema), but sorts keywords"""
        keywords = ", ".join("{}={!r}".format(k, v) for k, v in sorted(self.keywords.items()))
        format_string = "{module}.{cls}({component_subclass}, {keywords})"
        return format_string.format(module=self.__class__.__module__,
                                    cls=self.__class__.__qualname__,
                                    component_subclass=self.component_subclass,
                                    keywords=keywords)

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str = '') -> Any:
        if tag == '':
            tag = Registrable.get_default_tag(node.component_subclass, node.factory_method)
        return representer.represent_mapping(tag, node.keywords)

    @staticmethod
    def serialize(obj: Any) -> Dict[str, Any]:
        """Return dictionary representation of schema

        Includes yaml as a string, and extensions

        Parameters
        ----------
        obj: Any
            Should be schema or dict of schemas

        Returns
        -------
        Dict[str, Any]
            dictionary containing yaml and extensions dictionary

        """
        with StringIO() as stream:
            yaml.dump(obj, stream)
            serialized = stream.getvalue()
        exts: Dict[str, str] = {}
        # TODO: temporary until Pipeline object exists
        if isinstance(obj, dict):
            for value in obj.values():
                exts.update(value.aggregate_extensions_metadata())
        else:
            exts.update(obj.aggregate_extensions_metadata())
        rep = {'yaml': serialized, 'extensions': exts}
        return rep

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Any:
        """Construct Schema from dict returned by Schema.serialize

        Parameters
        ----------
        data: Dict[str, Any]
            dictionary returned by ``Schema.serialize``

        Returns
        -------
        Any
            Schema or dict of schemas (depending on yaml in ``data``)

        """
        yaml_str = data['yaml']
        extensions = data['extensions']
        obj = yaml.load(yaml_str)
        # TODO: temporary until Pipeline object exists
        if isinstance(obj, dict):
            for value in obj.values():
                value.add_extensions_metadata(extensions)
        else:
            obj.add_extensions_metadata(extensions)
        return obj


# Add representer for dumping Schema back to original yaml
# Behaves just like Component `to_yaml` but compilation not needed
yaml.representer.add_representer(Schema, Schema.to_yaml)


# Used to contextualize the representation of links during YAML
# representation
_link_root_obj: Optional['Component'] = None
_link_context_active = False
_link_obj_stash: Dict[str, Any] = {}


class contextualized_linking:
    """Context manager used to change the representation of links

    Links are always defined in relation to some root object and an
    attribute path, so when representing some piece of a larger object
    all the links need to be redefined in relation to the target object

    """

    def __init__(self, root_obj: Any, prefix: str) -> None:
        self.root_obj = root_obj
        self.prefix = prefix
        self.old_root: Optional['Component'] = None
        self.old_active = False
        self.old_stash: Dict[str, Any] = {}

    def __enter__(self) -> 'contextualized_linking':
        global _link_root_obj
        global _link_context_active
        global _link_obj_stash
        self.old_root = _link_root_obj
        self.old_active = _link_context_active
        self.old_stash = _link_obj_stash
        _link_root_obj = self.root_obj
        _link_context_active = True
        _link_obj_stash = {}
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _link_root_obj
        global _link_context_active
        global _link_obj_stash
        _link_root_obj = self.old_root
        _link_context_active = self.old_active
        _link_obj_stash = self.old_stash


@alias('$')
class PickledDataLink(Registrable):

    def __init__(self, obj_id: str, value: Any = None):
        self.obj_id = obj_id
        self.obj_value = value

    def __call__(self, stash: Dict[str, Any]) -> Any:
        if self.obj_value is not None:
            if self.obj_value != stash[self.obj_id]:
                warn("PickledDataLink called second time with different stash")
            return self.obj_value
        self.obj_value = stash[self.obj_id]
        return self.obj_value

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        global _link_obj_stash
        obj_id = str(len(_link_obj_stash.keys()))
        _link_obj_stash[obj_id] = node.obj_value
        return representer.represent_scalar(tag, obj_id)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> 'PickledDataLink':
        obj_id = constructor.construct_scalar(node)
        return cls(obj_id=obj_id)


class LinkError(Exception):
    pass


class MalformedLinkError(LinkError):
    pass


class UnpreparedLinkError(LinkError):
    pass


def parse_link_str(link_str: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parse link to extract schematic and attribute paths

    Links should be of the format ``obj[key1][key2].attr1.attr2`` where
    obj is the entry point; in a pipeline, obj would be the stage name,
    in a single-object config obj would be the target keyword at the
    top level. The following keys surrounded in brackets traverse
    the nested dictionary structure that appears in the config; this
    is intentonally analagous to how you would access properties in the
    dictionary when loaded into python. Then, you can use the dot
    notation to access the runtime instance attributes of the object
    at that location.

    Parameters
    ----------
    link_str : str
        Link to earlier object in the config of the format
        ``obj[key1][key2].attr1.attr2``

    Returns
    -------
    Tuple[Sequence[str], Sequence[str]]
        Tuple of the schematic and attribute paths respectively

    Raises
    -------
    MalformedLinkError
        If the link is written incorrectly

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> parse_link_str('obj[key1][key2].attr1.attr2')
    (['obj', 'key1', 'key2'], ['attr1', 'attr2'])

    """
    schematic_path: List[str] = []
    attr_path: List[str] = []
    temp: List[str] = []
    x = link_str
    # Parse schematic path
    bracket_open = False
    root_extracted = False
    while '[' in x or ']' in x:
        if bracket_open:
            temp = x.split(']', 1)
            if '[' in temp[0]:
                raise MalformedLinkError(f"Previous bracket unclosed in {link_str}")
            if len(temp) != 2:
                # Error case: [ not closed
                raise MalformedLinkError(f"Open bracket '[' not closed in {link_str}")
            schematic_path.append(temp[0])
            bracket_open = False
        else:
            # No bracket open yet
            temp = x.split('[', 1)
            if ']' in temp[0]:
                raise MalformedLinkError(f"Close ']' before open in {link_str}")
            if len(temp) != 2:
                # Error case: ] encountered without [
                raise MalformedLinkError(f"']' encountered before '[' in {link_str}")
            if len(temp[0]) != 0:
                if len(schematic_path) != 0:
                    # Error case: ]text[
                    raise MalformedLinkError(f"Text between brackets in {link_str}")
                # Beginning object name
                schematic_path.append(temp[0])
                root_extracted = True
            else:
                if len(schematic_path) == 0:
                    raise MalformedLinkError(f"No top level object in {link_str}")
            bracket_open = True
        # First part already added to schematic path, keep remainder
        x = temp[1]
    # Parse attribute path
    attr_path = x.split('.')
    if not root_extracted:
        if len(attr_path[0]) == 0:
            raise MalformedLinkError(f"No top level object in {link_str}")
        schematic_path.append(attr_path[0])
    elif len(attr_path) > 1:
        # Schematic processing did happen, so leading dot
        if attr_path[0] != '':
            # Error case: attr without dot beforehand
            raise MalformedLinkError(f"Attribute without preceeding dot notation in {link_str}")
        if attr_path[-1] == '':
            # Error case: trailing dot
            raise MalformedLinkError(f"Trailing dot in {link_str}")
    attr_path = attr_path[1:]
    return schematic_path, attr_path


def create_link_str(schematic_path: Sequence[str],
                    attr_path: Optional[Sequence[str]] = None) -> str:
    """Create a string representation of the specified link

    Performs the reverse operation of
    :func:`~flambe.compile.component.parse_link_str`

    Parameters
    ----------
    schematic_path : Sequence[str]
        List of entries corresponding to dictionary keys in a nested
        :class:`~flambe.compile.Schema`
    attr_path : Optional[Sequence[str]]
        List of attributes to access on the target object
        (the default is None).

    Returns
    -------
    str
        The string representation of the schematic + attribute paths

    Raises
    -------
    MalformedLinkError
        If the schematic_path is empty

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> create_link_str(['obj', 'key1', 'key2'], ['attr1', 'attr2'])
    'obj[key1][key2].attr1.attr2'

    """
    if len(schematic_path) == 0:
        raise MalformedLinkError("Can't create link without schematic path")
    root, schematic_path = schematic_path[0], schematic_path[1:]
    schematic_str = ''
    attr_str = ''
    if len(schematic_path) > 0:
        schematic_str = '[' + "][".join(schematic_path) + ']'
    if attr_path is not None and len(attr_path) > 0:
        attr_str = '.' + '.'.join(attr_path)
    return root + schematic_str + attr_str


@alias('@')
@alias('link')
class Link(Registrable):
    """Delayed access to another object in an object hiearchy

    Currently only supported in the context of Experiment but this
    may be updated in a future release

    A Link delays the access of some property, or the calling of some
    method, until the Link is called. Links can be passed directly
    into a Component subclass `compile`, Component's method called
    compile will automatically record the links and call them to
    access their values before running `__new__` and `__init__`. The
    recorded links will show up in the config if `yaml.dump()` is
    called on your object hierarchy. This typically happens when
    logging individual configs during a grid search, and when
    serializing between multiple processes.

    For example, if the schematic path is ['model', 'encoder'] and the
    attribute path is ['rnn', 'hidden_size'] then before the link can
    be compiled, the target attribute should be set to point to the
    model schema (this is handled automatically by Experiment) then,
    during compilation the child schema 'encoder' will be accessed,
    and finally the attribute encoder.rnn.hidden_size will be returned

    Parameters
    ----------
    schematic_path : Sequence[str]
        Path to the relevant schema denoted by dictionary-like bracket
        access e.g. ['model', 'encoder']
    attr_path : Sequence[str]
        Path to the relevant attribute on the given schema (after it's
        been compiled) using standard attribute dot notation e.g.
        ['rnn', 'hidden_size']
    target : Optional[Schema]
        The root object corresponding to the first element in the
        schematic path; needs to be passed in here or set later before
        link can be resolved
    local : bool
        if true, changes tune convert behavior to insert a dummy link;
        used for links to global variables ("resources" in config)
        (defaults to True)

    """

    def __init__(self,
                 schematic_path: Sequence[str],
                 attr_path: Optional[Sequence[str]] = None,
                 target: Optional[Schema] = None,
                 local: bool = True) -> None:
        self.schematic_path = schematic_path
        self.attr_path = attr_path
        self.target = target
        self.target_leaf: Optional[Schema] = None
        self.local = local
        self.resolved: Optional[Any] = None
        self.post_init_hooks: Sequence[Callable] = []

    @property
    def root_schema(self) -> str:
        return self.schematic_path[0]

    def __repr__(self) -> str:
        return f'link({create_link_str(self.schematic_path, self.attr_path)})'

    def __call__(self) -> Any:
        # Link already resolved once, so return cached result
        if self.resolved is not None:
            return self.resolved  # type: ignore
        # The relevant root object must be set to resolve the link
        if self.target is None:
            raise UnpreparedLinkError('Link object was not properly updated')
        current_obj: Any = self.target

        def auto_resolve_link_and_move_to_schema(obj):
            if isinstance(obj, Link):
                obj()
                obj = obj.resolved
            if isinstance(obj, Component):
                obj = obj._schema
            return obj

        # Traverse the schematic path, automatically resolving
        # any chained links; only traverse the schema structure
        current_obj = auto_resolve_link_and_move_to_schema(current_obj)
        for schema in self.schematic_path[1:]:
            try:
                current_obj = current_obj.keywords[schema]
            except KeyError:
                raise KeyError(f'Could not resolve link {schema}. (Check all !@ entries.)')
            current_obj = auto_resolve_link_and_move_to_schema(current_obj)
        self.target_leaf = current_obj
        # At the end of the schematic path, access the compiled object
        # If it's not compiled, it's either later in the config
        # or is a parent of the link, both of which are invalid links
        if isinstance(current_obj, Schema):
            if current_obj._compiled is not None:
                current_obj = current_obj._compiled
            else:
                raise UnpreparedLinkError("Cannot resolve link to non-compiled object "
                                          f"{current_obj}. Remember only non-parent objects above "
                                          "the link in the config  will be compiled when the link "
                                          "is resolved")
        # Then access the attributes of the compiled object
        if self.attr_path is not None:
            for attr in self.attr_path:
                current_obj = getattr(current_obj, attr)
        self.resolved = current_obj
        for hook in self.post_init_hooks:
            hook(current_obj)
        return current_obj

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Build contextualized link based on the root node

        If the link refers to something inside of the current object
        hierarchy (as determined by the schema of ``_link_root_obj``)
        then it will be represented as a link; if the link refers to
        something out-of-scope, i.e. not inside the current object
        hiearchy, then replace the link with the resolved value. If
        the value cannot be represented, pickle it and include a
        reference to its id in the object stash that will be saved
        alongside the config

        """
        global _link_root_obj
        global _link_context_active
        global _link_obj_stash
        if _link_context_active:
            if _link_root_obj is None:
                raise Exception("Contextual linking requires root object to be set")
            present = False
            if _link_root_obj._schema is not None:
                present, schematic_path = _link_root_obj._schema.contains(node.target_leaf, node)
            if present:
                link_str = create_link_str(schematic_path, node.attr_path)
                return representer.represent_scalar(tag, link_str)
            else:
                if isinstance(node.resolved, Registrable):
                    return node.resolved.to_yaml(representer, node.resolved,
                                                 node.resolved._created_with_tag)  # type: ignore  # noqa: E501
                else:
                    try:
                        return representer.represent_data(node.resolved)
                    except RepresenterError:
                        data_link = PickledDataLink(obj_id='0', value=node.resolved)
                        return PickledDataLink.to_yaml(representer, data_link, '!$')
        # No contextualization necessary
        link_str = create_link_str(node.schematic_path, node.attr_path)
        return representer.represent_scalar(tag, link_str)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> 'Link':
        link_str = constructor.construct_scalar(node)
        schematic_path, attr_path = parse_link_str(link_str)
        kwargs = {'schematic_path': schematic_path, 'attr_path': attr_path}
        return cls(**kwargs)  # type: ignore

    def convert(self) -> Callable[..., Any]:
        if self.local:
            return ray.tune.function(lambda spec: eval(f'spec'))  # TODO what do here
        return ray.tune.function(lambda spec: eval(f'spec.config.params.{self.root_schema}'))


@alias('call')
class FunctionCallLink(Link):
    """Calls the link attribute instead of just accessing it"""

    def __call__(self) -> Any:
        if self.resolved is not None:
            return self.resolved
        fn = super().__call__()
        self.resolved = fn()
        return self.resolved


K = TypeVar('K')


def fill_defaults(kwargs: Dict[str, Any], function: Callable[..., Any]) -> Dict[str, Any]:
    """Use function signature to add missing kwargs to a dictionary"""
    signature = inspect.signature(function)
    kwargs_with_defaults = kwargs.copy()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        default = param.default
        if name not in kwargs and default != _EMPTY:
            kwargs_with_defaults[name] = default
    return kwargs_with_defaults


def merge_kwargs(kwargs: Dict[str, Any], compiled_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Replace non links in kwargs with corresponding compiled values

    For every key in `kwargs` if the value is NOT a link and IS a
    Schema, replace with the corresponding value in `compiled_kwargs`

    Parameters
    ----------
    kwargs : Dict[str, Any]
        Original kwargs containing Links and Schemas
    compiled_kwargs : Dict[str, Any]
        Processes kwargs containing no links and no Schemas

    Returns
    -------
    Dict[str, Any]
        kwargs with links, but with Schemas replaced by compiled
        objects

    """
    merged_kwargs = {}
    for kw in kwargs:
        if not isinstance(kwargs[kw], Link) and isinstance(kwargs[kw], Schema):
            if kw not in compiled_kwargs:
                raise CompilationError('Non matching kwargs and compiled_kwargs')
            merged_kwargs[kw] = compiled_kwargs[kw]
        else:
            merged_kwargs[kw] = kwargs[kw]
    return merged_kwargs


class Component(Registrable):
    """Class which can be serialized to yaml and implements `compile`

    IMPORTANT: ALWAYS inherit from Component BEFORE `torch.nn.Module`

    Automatically registers subclasses via Registrable and
    facilitates immediate usage in YAML with tags. When loaded,
    subclasses' initialization is delayed; kwargs are wrapped in a
    custom schema called Schema that can be easily initialized
    later.

    """

    _flambe_version = '0.0.0'  # >0.0.0 opts into semantic versioning

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._schema: Optional[Schema] = None
        if isinstance(self, torch.nn.Module):
            self._register_state_dict_hook(self._state_dict_hook)
            self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    def run(self) -> bool:
        """Run a single computational step.

        When used in an experiment, this computational step should
        be on the order of tens of seconds to about 10 minutes of work
        on your intended hardware; checkpoints will be performed in
        between calls to run, and resources or search algorithms will
        be updated. If you want to run everything all at once, make
        sure a single call to run does all the work and return False.

        Returns
        -------
        bool
            True if should continue running later i.e. more work to do

        """
        # By default it doesn't do anything and doesn't continue
        continue_ = False
        return continue_

    def metric(self) -> Optional[float]:
        """Override this method to enable scheduling and searching.

        Returns
        -------
        float
            The metric to compare different variants of your Component

        """
        return None

    @property
    def _config_str(self):
        """Represent object's architecture as a YAML string

        Includes the extensions relevant to the object as well; NOTE:
        currently this section may include a superset of the extensions
        actually needed, but this will be changed in a future release.

        """
        stream = None
        if not hasattr(self, '_saved_kwargs'):
            raise AttributeError(f"{type(self).__name__} object was not compiled from YAML (or "
                                 "created via the factory method 'compile') and does not have an"
                                 " associated config")
        try:
            config = ""
            stream = StringIO()
            try:
                exts = self.aggregate_extensions_metadata()
                if exts is not None and len(exts) > 0:
                    yaml.dump_all([exts, self], stream)
                else:
                    yaml.dump(self, stream)
                config = stream.getvalue()
            except RepresenterError as re:
                print(re)
                logger.warn("Exception representing attribute in yaml... ", re)
            finally:
                if not stream.closed:
                    stream.close()
                return config
        except AttributeError as a:
            if stream is not None and not stream.closed:
                stream.close()
            print(a)
            raise AttributeError(f"{type(self).__name__} object was not compiled from YAML (or "
                                 "created via the factory method 'compile') and does not have an"
                                 "associated config")
        except Exception as e:
            if stream is not None and not stream.closed:
                stream.close()
            raise e

    def register_attrs(self, *names: str) -> None:
        """Set attributes that should be included in state_dict

        Equivalent to overriding `obj._state` and `obj._load_state` to
        save and load these attributes. Recommended usage: call inside
        `__init__` at the end: `self.register_attrs(attr1, attr2, ...)`
        Should ONLY be called on existing attributes.

        Parameters
        ----------
        *names : str
            The names of the attributes to register

        Raises
        -------
        AttributeError
            If `self` does not have existing attribute with that name

        """
        if not hasattr(self, '_registered_attributes'):
            self._registered_attributes: Set[str] = set()
        for name in names:
            if not hasattr(self, name):
                raise AttributeError(f"{type(self).__name__} object has no attribute {name}, so "
                                     "it cannot be registered")
        self._registered_attributes.update(names)

    @staticmethod
    def _state_dict_hook(self,
                         state_dict: State,
                         prefix: str,
                         local_metadata: Dict[str, Any]) -> State:
        """Add metadata and recurse on Component children

        This hook is used to integrate with the PyTorch `state_dict`
        mechanism; as either `nn.Module.state_dict` or
        `Component.get_state` recurse, this hook is responsible for
        adding Flambe specific metadata and recursing further on any
        Component children of `self` that are not also nn.Modules,
        as PyTorch will handle recursing to the latter.

        Flambe specific metadata includes the class version specified
        in the `Component._flambe_version` class property, the name
        of the class, the source code, and the fact that this class is
        a `Component` and should correspond to a directory in our
        hiearchical save format

        Finally, this hook calls a helper `_state` that users can
        implement to add custom state to a given class

        Parameters
        ----------
        state_dict : State
            The state_dict as defined by PyTorch; a flat dictionary
            with compound keys separated by '.'
        prefix : str
            The current prefix for new compound keys that reflects the
            location of this instance in the object hierarchy being
            represented
        local_metadata : Dict[str, Any]
            A subset of the metadata relevant just to this object and
            its children

        Returns
        -------
        type
            The modified state_dict

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """
        warn_use_state = False
        if FLAMBE_DIRECTORIES_KEY not in state_dict._metadata:
            state_dict._metadata[FLAMBE_DIRECTORIES_KEY] = set()
            warn_use_state = True
        if KEEP_VARS_KEY not in state_dict._metadata:
            state_dict._metadata[KEEP_VARS_KEY] = False
            warn_use_state = True
        if warn_use_state:
            warn("Use '.get_state()' on flambe objects, not state_dict "
                 f"(from {type(self).__name__})")
        # 1 need to add in any extras like config
        local_metadata[VERSION_KEY] = self._flambe_version
        local_metadata[FLAMBE_CLASS_KEY] = type(self).__name__
        local_metadata[FLAMBE_SOURCE_KEY] = dill.source.getsource(type(self))
        # All links should be relative to the current object `self`
        with contextualized_linking(root_obj=self, prefix=prefix[:-1]):
            try:
                local_metadata[FLAMBE_CONFIG_KEY] = self._config_str
                global _link_obj_stash
                if len(_link_obj_stash) > 0:
                    local_metadata[FLAMBE_STASH_KEY] = copy.deepcopy(_link_obj_stash)
            except AttributeError:
                pass
        # 2 need to recurse on Components
        # Iterating over __dict__ does NOT include pytorch children
        # modules, parameters or buffers
        # torch.optim.Optimizer does exist so ignore mypy
        for name, attr in self.__dict__.items():
            current_path = prefix + name
            if isinstance(attr, Component) and not isinstance(attr, (
                    torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler)):  # type: ignore
                # If self is not nn.Module, need to recurse because
                # that will not happen elsewhere
                # If self *is* an nn.Module, don't need to recurse on
                # child nn.Module's because pytorch will already do
                # that; just recurse on non-nn.Module's
                # The latter case shouldn't happen, this is just an
                # extra check for safety;
                # child modules are not stored in __dict__
                if not isinstance(self, torch.nn.Module) or not isinstance(attr, torch.nn.Module):
                    state_dict = attr.get_state(destination=state_dict,
                                                prefix=current_path + STATE_DICT_DELIMETER,
                                                keep_vars=state_dict._metadata[KEEP_VARS_KEY])
                state_dict._metadata[FLAMBE_DIRECTORIES_KEY].add(current_path)
            # Iterate over modules to make sure NON-Component
            # nn.Modules' state is added. Only needed if self is not
            # nn.Module, because otherwise this hook is being called
            # via nn.Module.state_dict, and will already recurse to
            # all children modules
            if not isinstance(self, torch.nn.Module) and isinstance(attr, torch.nn.Module) \
                    and not isinstance(attr, Component):
                state_dict = attr.state_dict(destination=state_dict,
                                             prefix=current_path + STATE_DICT_DELIMETER,
                                             keep_vars=state_dict._metadata[KEEP_VARS_KEY])
        state_dict._metadata[FLAMBE_DIRECTORIES_KEY].add(prefix[:-1])
        state_dict = self._add_registered_attrs(state_dict, prefix)
        state_dict = self._state(state_dict, prefix, local_metadata)
        return state_dict

    def _add_registered_attrs(self, state_dict: State, prefix: str) -> State:
        if hasattr(self, '_registered_attributes'):
            for attr_name in self._registered_attributes:
                state_dict[prefix + attr_name] = getattr(self, attr_name)
        return state_dict

    def _state(self, state_dict: State, prefix: str, local_metadata: Dict[str, Any]) -> State:
        """Add custom state to state_dict

        Parameters
        ----------
        state_dict : State
            The state_dict as defined by PyTorch; a flat dictionary
            with compound keys separated by '.'
        prefix : str
            The current prefix for new compound keys that reflects the
            location of this instance in the object hierarchy being
            represented
        local_metadata : Dict[str, Any]
            A subset of the metadata relevant just to this object and
            its children

        Returns
        -------
        State
            The modified state_dict

        """
        return state_dict

    def get_state(self,
                  destination: Optional[State] = None,
                  prefix: str = '',
                  keep_vars: bool = False) -> State:
        """Extract PyTorch compatible state_dict

        Adds Flambe specific properties to the state_dict, including
        special metadata (the class version, source code, and class
        name). By default, only includes state that PyTorch `nn.Module`
        includes (Parameters, Buffers, child Modules). Custom state can
        be added via the `_state` helper method which subclasses should
        override.

        The metadata `_flambe_directories` indicates which objects are
        Components and should be a subdirectory in our hierarchical
        save format. This object will recurse on `Component` and
        `nn.Module` children, but NOT `torch.optim.Optimizer`
        subclasses, `torch.optim.lr_scheduler._LRScheduler` subclasses,
        or any other arbitrary python objects.

        Parameters
        ----------
        destination : Optional[State]
            The state_dict as defined by PyTorch; a flat dictionary
            with compound keys separated by '.'
        prefix : str
            The current prefix for new compound keys that reflects the
            location of this instance in the object hierarchy being
            represented
        keep_vars : bool
            Whether or not to keep Variables (only used by PyTorch)
            (the default is False).

        Returns
        -------
        State
            The state_dict object

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """
        if destination is None:
            destination = State()
            destination._metadata = OrderedDict({FLAMBE_DIRECTORIES_KEY: set(),
                                                 KEEP_VARS_KEY: keep_vars})
            destination._metadata[FLAMBE_DIRECTORIES_KEY].add(prefix)
        if isinstance(self, torch.nn.Module):
            destination = self.state_dict(destination, prefix, keep_vars)
        # torch.optim.Optimizer does exist so ignore mypy
        elif isinstance(self, (torch.optim.Optimizer,  # type: ignore
                               torch.optim.lr_scheduler._LRScheduler)):
            pass
        else:
            local_metadata: Dict[str, Any] = {}
            destination._metadata[prefix[:-1]] = local_metadata
            destination = self._state_dict_hook(self, destination, prefix, local_metadata)

        return destination  # type: ignore

    def _load_state_dict_hook(self,
                              state_dict: State,
                              prefix: str,
                              local_metadata: Dict[str, Any],
                              strict: bool,
                              missing_keys: List[Any],
                              unexpected_keys: List[Any],
                              error_msgs: List[Any]) -> None:
        """Load flambe-specific state

        Parameters
        ----------
        state_dict : State
            The state_dict as defined by PyTorch; a flat dictionary
            with compound keys separated by '.'
        prefix : str
            The current prefix for new compound keys that reflects the
            location of this instance in the object hierarchy being
            represented
        local_metadata : Dict[str, Any]
            A subset of the metadata relevant just to this object and
            its children
        strict : bool
            Whether missing or unexpected keys should be allowed;
            should always be False in Flambe
        missing_keys : List[Any]
            Missing keys so far
        unexpected_keys : List[Any]
            Unexpected keys so far
        error_msgs : List[Any]
            Any error messages so far

        Raises
        -------
        LoadError
            If the state for some object does not have a matching major
            version number

        """
        # Custom subclass behavior
        self._load_state(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                         error_msgs)
        self._load_registered_attrs(state_dict, prefix)
        # Check state compatibility
        version = local_metadata[VERSION_KEY].split('.')
        if min(map(int, version)) > 0:
            # Opt-in to semantic versioning
            versions = local_metadata[VERSION_KEY], type(self)._flambe_version
            load_version, current_version = map(lambda x: x.split('.'), versions)
            if load_version[0] != current_version[0]:
                raise LoadError(f'Incompatible Versions: {load_version} and {current_version}')
            if load_version[1] != current_version[1]:
                logger.warn(f'Differing Versions (Minor): {load_version} and {current_version}')
            if load_version[2] != current_version[2]:
                logger.debug(f'Differing Versions (Patch): {load_version} and {current_version}')
        else:
            original_source = local_metadata[FLAMBE_SOURCE_KEY]
            current_source = dill.source.getsource(type(self))
            if original_source != current_source:
                # Warn / Error
                logger.warn(f"Source code for object {self} does not match the source code saved "
                            f"with the state dict\nSource code: {current_source}\n"
                            f"Original source code:{original_source}\n")

    def _load_registered_attrs(self, state_dict: State, prefix: str):
        if hasattr(self, '_registered_attributes'):
            for attr_name in self._registered_attributes:
                setattr(self, attr_name, state_dict[prefix + attr_name])

    def _load_state(self,
                    state_dict: State,
                    prefix: str,
                    local_metadata: Dict[str, Any],
                    strict: bool,
                    missing_keys: List[Any],
                    unexpected_keys: List[Any],
                    error_msgs: List[Any]) -> None:
        """Load custom state (that was included via `_state`)

        Subclasses should override this function to add custom state
        that isn't normally included by PyTorch nn.Module

        Parameters
        ----------
        state_dict : State
            The state_dict as defined by PyTorch; a flat dictionary
            with compound keys separated by '.'
        prefix : str
            The current prefix for new compound keys that reflects the
            location of this instance in the object hierarchy being
            represented
        local_metadata : Dict[str, Any]
            A subset of the metadata relevant just to this object and
            its children
        strict : bool
            Whether missing or unexpected keys should be allowed;
            should always be False in Flambe
        missing_keys : List[Any]
            Missing keys so far
        unexpected_keys : List[Any]
            Unexpected keys so far
        error_msgs : List[Any]
            Any error messages so far

        """
        pass

    def load_state(self, state_dict: State, strict: bool = False) -> None:
        """Load `state_dict` into `self`

        Loads state produced by `get_state` into the current object,
        recursing on child `Component` and `nn.Module` objects

        Parameters
        ----------
        state_dict : State
            The state_dict as defined by PyTorch; a flat dictionary
            with compound keys separated by '.'
        strict : bool
            Whether missing or unexpected keys should be allowed;
            should ALWAYS be False in Flambe (the default is False).

        Raises
        -------
        LoadError
            If the state for some object does not have a matching major
            version number

        """
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # For loading, the _load_from_state_dict and
        # _load_state_dict_hook are NOT recursive.
        # We emulate PyTorch's structure by having a recursive
        # helper here, for compatibility reasons.

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if isinstance(module, torch.nn.Module):
                module_load_fn = module._load_from_state_dict
            else:
                module_load_fn = module._load_state_dict_hook
            module_load_fn(state_dict, prefix, local_metadata, True, missing_keys,
                           unexpected_keys, error_msgs)
            for name, child in module.__dict__.items():
                if child is not None and isinstance(child, Component):
                    if not isinstance(child, (torch.optim.Optimizer,
                                              torch.optim.lr_scheduler._LRScheduler)):
                        load(child, prefix + name + STATE_DICT_DELIMETER)
            if isinstance(module, torch.nn.Module):
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + STATE_DICT_DELIMETER)
        load(self)
        # PyTorch 1.1 error handling
        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(0, 'Unexpected key(s) in state_dict: '
                                     f'{", ".join(f"{k}" for k in unexpected_keys)}. ')
            if len(missing_keys) > 0:
                error_msgs.insert(0, 'Missing key(s) in state_dict: '
                                     f'{", ".join(f"{k}" for k in missing_keys)}. ')
        if len(error_msgs) > 0:
            newline_tab = '\n\t'
            raise RuntimeError('Error(s) in loading state_dict for '
                               f'{self.__class__.__name__}:{newline_tab}'
                               f'{newline_tab.join(error_msgs)}')

    @registrable_factory
    @classmethod
    def load_from_path(cls,
                       path: str,
                       map_location: Union[torch.device, str] = None,
                       use_saved_config_defaults: bool = True,
                       **kwargs: Any):
        if use_saved_config_defaults:
            instance = flambe_load(path, map_location=map_location)
        else:
            loaded_state = load_state_from_file(path, map_location=map_location)
            instance = cls(**kwargs)
            instance.load_state(loaded_state)
        return instance

    def save(self, path: str, **kwargs: Any):
        flambe_save(self, path, **kwargs)

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        return representer.represent_mapping(tag, node._saved_kwargs)

    @classmethod
    def from_yaml(cls: Type[C], constructor: Any, node: Any, factory_name: str) -> Schema:
        # Normally you would create an instance of this class with
        # cls(...) but in this case we don't want to init the object
        # yet so we create a modified schema that will recursively
        # initialize kwargs via compile when the top level compilation
        # begins
        if inspect.isabstract(cls):
            msg = f"You're trying to initialize an abstract class {cls}. " \
                  + "If you think it's concrete, double check you've spelled " \
                  + "all the method names correctly."
            raise Exception(msg)
        if isinstance(node, ScalarNode):
            nothing = constructor.construct_yaml_null(node)
            if nothing is not None:
                warn(f"Non-null scalar argument to {cls.__name__} will be ignored")
            return Schema(cls, _flambe_custom_factory_name=factory_name)
        # NOTE: construct_yaml_map is a generator that yields the
        # constructed data and then updates it
        kwargs, = list(constructor.construct_yaml_map(node))
        return Schema(cls, _flambe_custom_factory_name=factory_name, **kwargs)

    @classmethod
    def precompile(cls: Type[C], **kwargs: Any) -> None:
        """Change kwargs before compilation occurs.

        This hook is called after links have been activated, but before
        calling the recursive initialization process on all other
        objects in kwargs. This is useful in a number of cases, for
        example, in Trainer, we compile several objects ahead of time
        and move them to the GPU before compiling the optimizer,
        because it needs to be initialized with the model parameters
        *after* they have been moved to GPU.

        Parameters
        ----------
        cls : Type[C]
            Class on which method is called
        **kwargs : Any
            Current kwargs that will be compiled and used to initialize
            an instance of cls after this hook is called

        """
        return

    def aggregate_extensions_metadata(self) -> Dict[str, str]:
        """Aggregate extensions used in object hierarchy

        TODO: remove or combine with schema implementation in refactor

        """
        # non-nested so shallow copy ok
        exts = dict(self._extensions or {})  # type: ignore

        def helper(data):
            if isinstance(data, Component):
                exts.update(data.aggregate_extensions_metadata())
            elif isinstance(data, list) or isinstance(data, tuple):
                for val in data:
                    helper(val)
            elif isinstance(data, Mapping):
                for val in data.values():
                    helper(val)

        for child in self._saved_kwargs.values():  # type: ignore
            helper(child)

        return exts

    @classmethod
    def compile(cls: Type[C],
                _flambe_custom_factory_name: Optional[str] = None,
                _flambe_extensions: Optional[Dict[str, str]] = None,
                _flambe_stash: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> C:
        """Create instance of cls after recursively compiling kwargs

        Similar to normal initialization, but recursively initializes
        any arguments that should be compiled and allows overriding
        arbitrarily deep kwargs before initializing if needed. Also
        activates any Link instances passed in as kwargs, and saves
        the original kwargs for dumping to yaml later.

        Parameters
        ----------
        **kwargs : Any
            Keyword args that should be forwarded to the initialization
            function (a specified factory, or the normal `__new__`
            and `__init__` methods)

        Returns
        -------
        C
            An instance of the class `cls`

        """
        extensions: Dict[str, str] = _flambe_extensions or {}
        stash: Dict[str, Any] = _flambe_stash or {}
        # Allow objects to do custom operations such as adding hooks
        cls.precompile(**kwargs)
        # Recursively compile any remaining un-compiled kwargs

        def helper(obj: Any) -> Any:
            if isinstance(obj, Schema):
                obj.add_extensions_metadata(extensions)
                out = obj(stash)  # type: ignore
            elif isinstance(obj, Link):
                out = obj()
            elif isinstance(obj, PickledDataLink):
                out = obj(stash)
            # string passes as sequence
            elif isinstance(obj, list) or isinstance(obj, tuple):
                out = []
                for value in obj:
                    out.append(helper(value))
            elif isinstance(obj, Mapping):
                out = {}
                for key, value in obj.items():
                    out[key] = helper(value)
            else:
                out = obj
            return out

        newkeywords = helper(kwargs)
        # Check for remaining yaml types
        for kw in newkeywords:
            if isinstance(newkeywords[kw], YAML_TYPES):
                msg = f"'{cls}' property '{kw}' is still yaml type {type(newkeywords[kw])}\n"
                msg += f"This could be because of a typo or the class is not registered properly"
                warn(msg)
        # Find intended constructor in case using some factory
        factory_method: Callable[..., Any] = cls
        if _flambe_custom_factory_name is not None:
            factory_method = getattr(cls, _flambe_custom_factory_name)
        # Replace non link Schemas with compiled objects in kwargs
        # for dumping
        kwargs_non_links_compiled = merge_kwargs(kwargs, newkeywords)
        # Fill the *original* kwargs with defaults specified by factory
        kwargs_with_defaults = fill_defaults(kwargs_non_links_compiled, factory_method)
        # Creat the compiled instance of `cls`
        try:
            instance = factory_method(**newkeywords)
        except TypeError as te:
            print(f"class {cls} method {_flambe_custom_factory_name} failed with "
                  f"keyword args:\n{newkeywords}")
            raise te
        # Record kwargs used for compilation for YAML dumping later
        # Includes defaults for better safety / reproducibility
        instance._saved_kwargs = kwargs_with_defaults
        instance._extensions = extensions
        return instance


def dynamic_component(class_: Type[A],
                      tag: str,
                      tag_namespace: Optional[str] = None,
                      parent_component_class: Type[Component] = Component) -> Type[Component]:
    """Decorate given class, creating a dynamic `Component`

    Creates a dynamic subclass of `class_` that inherits from
    `Component` so it will be registered with the yaml loader and
    receive the appropriate functionality (`from_yaml`, `to_yaml` and
    `compile`). `class_` should not implement any of the aforementioned
    functions.

    Parameters
    ----------
    class_ : Type[A]
        Class to register with yaml and the compilation system
    tag : str
        Tag that will be used with yaml
    tag_namespace : str
        Namespace aka the prefix, used. e.g. for `!torch.Adam` torch is
        the namespace

    Returns
    -------
    Type[Component]
        New subclass of `_class` and `Component`

    """
    if not issubclass(parent_component_class, Component):
        raise Exception("Only a subclass of Component should be used for 'parent_component_class'")
    if issubclass(class_, parent_component_class):
        return class_

    # Copy over class attributes so it still looks like the original
    # Useful for inspection and debugging purposes
    _MISSING = object()
    copied_attrs = {}
    for k in WRAPPER_ASSIGNMENTS:
        v = getattr(class_, k, _MISSING)
        if v is not _MISSING:
            copied_attrs[k] = v

    # Create new subclass of `class_` and `Component`
    # Ignore mypy, extra kwargs are okay in python 3.6+ usage of type
    # and Registrable uses them
    new_component = type(class_.__name__,  # type: ignore
                         (parent_component_class, class_),
                         copied_attrs,
                         tag_override=tag,
                         tag_namespace=tag_namespace)  # type: ignore

    return new_component
