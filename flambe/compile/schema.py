import inspect
from reprlib import recursive_repr
from typing import MutableMapping, Any, Callable, Optional, Dict, Sequence, Tuple, List, \
                   Iterable, Type, Mapping
from warnings import warn
import copy
import functools

from ruamel.yaml import ScalarNode
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
                                  CommentedKeySeq, CommentedSeq, TaggedScalar,
                                  CommentedKeyMap)

from flambe.compile.common import function_defaults
from flambe.compile.registry import get_registry
from flambe.compile.registered_types import Registrable


YAML_TYPES = (CommentedMap, CommentedOrderedMap, CommentedSet, CommentedKeySeq, CommentedSeq,
              TaggedScalar, CommentedKeyMap)


class LinkError(Exception):
    pass


class MalformedLinkError(LinkError):
    pass


class UnpreparedLinkError(LinkError):
    pass


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


class Variants(Registrable, tag_override="g"):

    def __init__(self, options: Iterable[Any]):
        self.options = options

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        raise NotImplementedError()

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> 'Link':
         # construct_yaml_seq returns wrapper tuple, need to unpack;
         #  will also recurse so items in options can also be links
         options, = list(constructor.construct_yaml_seq(node))
         return Variants(options)


class Link(Registrable, tag_override="@"):

    def __init__(self,
                 link_str: Optional[str] = None,
                 schematic_path: Optional[Sequence[str]] = None,
                 attr_path: Optional[Sequence[str]] = None) -> None:
        if link_str is not None:
            if schematic_path is not None or attr_path is not None:
                raise ValueError()
            schematic_path, attr_path = parse_link_str(link_str)
        if schematic_path is None:
            raise ValueError()
        self.schematic_path = schematic_path
        self.attr_path = attr_path

    def resolve(self, cache: Dict[str, Any]) -> Any:
        try:
            obj = cache[self.schematic_path]
        except KeyError:
            raise MalformedLinkError('Link does not point to an object that has been initialized. '
                                     'Make sure the link points to a non-parent above the link '
                                     'in the config.')
        if self.attr_path is not None:
            for attr in self.attr_path:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    raise MalformedLinkError(f'Link {self} failed when resolving '
                                             f'on object {obj}. Failed at attribute {attr}')
        return obj

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> 'Link':
        if isinstance(node, ScalarNode):
            link_str = constructor.construct_scalar(node)
            return cls(link_str)
        # elif isinstance(node, SequenceNode):
            # # construct_yaml_seq returns wrapper tuple, need to unpack;
            # #  will also recurse so items in options can also be links
            # options, = list(constructor.construct_yaml_seq(node))
            # return Variants(options)

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        raise NotImplementedError()

    def __call__(self, cache: Dict[str, Any]) -> Any:
        return self.resolve(cache)

    def __repr__(self) -> str:
        return f'link({create_link_str(self.schematic_path, self.attr_path)})'


class Schema(MutableMapping[str, Any]):

    def __init__(self,
                 callable: Callable,
                 kwargs: Dict[str, Any],
                 factory_name: Optional[str] = None,
                 tag: Optional[str] = None):
        self.callable = callable
        # TODO auto-register logic
        if factory_name is None:
            if not isinstance(self.callable, type):
                raise NotImplementedError('Using non-class callables with Schema is not yet supported')
            self.factory_method = callable
        else:
            if not isinstance(self.callable, type):
                raise Exception(f'Cannot specify factory name on non-class callable {callable}')
            self.factory_method = getattr(self.callable, factory_name)
        self.kwargs = function_defaults(self.factory_method)
        self.kwargs.update(kwargs)
        if tag is None:
            if isinstance(node, functools.partial):
                tag = node.func.__name__
            elif isinstance(node, object):
                tag = type(node).__name__
        self.created_with_tag = tag

    def __setitem__(self, key: str, value: Any) -> None:
        self.kwargs[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.kwargs[key]

    def __delitem__(self, key: str) -> None:
        del self.kwargs[key]

    def __iter__(self) -> Iterable[str]:
        yield from self.kwargs

    def __len__(self) -> int:
        return len(self.kwargs)

    def __call__(self,
                 path: Optional[List[str]] = None,
                 cache: Optional[Dict[str, Any]] = None):
        return self.initialize(path, cache)

    @classmethod
    def from_yaml(cls,
                  constructor: Any,
                  node: Any,
                  factory_name: str,
                  tag: str,
                  callable: Callable) -> Any:
        """Use constructor to create an instance of cls"""
        if inspect.isabstract(callable):
            msg = f"You're trying to initialize an abstract class {cls}. " \
                  + "If you think it's concrete, double check you've spelled " \
                  + "all the method names correctly."
            raise Exception(msg)
        if isinstance(node, ScalarNode):
            nothing = constructor.construct_yaml_null(node)
            if nothing is not None:
                raise Exception(f"Non-null scalar argument to {cls.__name__} will be ignored")
            return cls(callable, {}, factory_name, tag)
        kwargs, = list(constructor.construct_yaml_map(node))
        return cls(callable, kwargs, factory_name, tag)

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        pass

    @staticmethod
    def traverse(obj: Any,
                 current_path: Optional[List[str]] = None,
                 fn: Optional[Callable] = None,
                 yield_schema: Optional[str] = None) -> Iterable[Tuple[str, Any]]:
        current_path = current_path or tuple()
        fn = fn or (lambda x: x)
        if isinstance(obj, Link):
            yield (current_path, obj)
        elif isinstance(obj, Schema):
            if yield_schema is None or yield_schema == 'before':
                yield (current_path, fn(obj))
                yield from Schema.traverse(obj.kwargs, current_path, fn, yield_schema)
            elif yield_schema == 'only':
                yield (current_path, fn(obj))
            elif yield_schema == 'after':
                yield from Schema.traverse(obj.kwargs, current_path, fn, yield_schema)
                yield (current_path, fn(obj))
            elif yield_schema == 'never':
                yield from Schema.traverse(obj.kwargs, current_path, fn, yield_schema)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                next_path = current_path + (k,)
                yield from Schema.traverse(v, next_path, fn, yield_schema)
        elif isinstance(obj, (list, tuple)):
            for i, e in enumerate(obj):
                next_path = current_path[:] + (i,)
                yield from Schema.traverse(e, next_path, fn, yield_schema)
        else:
            yield (current_path, obj)

    def set_param(self, path: Optional[Tuple[str]], value: Any) -> None:
        current_obj = self
        for item in path[:-1]:
            current_obj = current_obj[item]
        current_obj[path[-1]] = value

    def initialize(self,
                   path: Optional[Tuple[str]] = None,
                   cache: Optional[Dict[str, Any]] = None) -> Any:
        cache = cache or {}
        path = path or tuple()
        if path in cache:
            return cache[path]

        initialized = copy.deepcopy(self)
        for path, obj in self.traverse(self.kwargs, yield_schema='only'):
            if isinstance(obj, Link):
                initialized.set_param(path, obj(cache))
            elif isinstance(obj, Schema):
                initialized.set_param(path, obj(path, cache))
        initialized_kwargs = initialized.kwargs

        for k, v in initialized_kwargs.items():
            if isinstance(v, YAML_TYPES):
                msg = f"keyword '{k}' is still yaml type {type(v)}\n"
                msg += f"This could be because of a typo or the class is not registered properly"
                warn(msg)
        try:
            cache[path] = self.factory_method(**initialized_kwargs)
        except TypeError as te:
            print(f"Constructor {self.factory_method} failed with "
                  f"keyword args:\n{initialized_kwargs}")
            raise te
        return cache[path]

    def extract_search_space(self) -> Dict[str, Tuple[str]]:
        search_space = {}

        for path, item in traverse(self, yield_schema='never'):
            if isinstance(item, Link):
                pass
            elif isinstance(item, Distribution):
                search_space[path] = item

        return search_space

    def set_from_search_space(self, search_space: Dict[str, Tuple[str]]) -> None:
        for path, value in search_space.items():
            self.set_param(path, value)

    def iter_variants(self) -> 'Schema':
        """Yield variants selecting the parallel options from each"""
        for selection_index in range(self.num_options):
            variant_schema = copy.deepcopy(self)
            for path, item in traverse(self, yield_schema='never'):
                if isinstance(item, Variants):
                    value = item[selection_index]
                    variant_schema.set_param(path, value)
            yield variant_schema

    def merge_union(self, *others) -> None:
        """Merge into self keeping all options in parallel"""
        raise NotImplementedError()

    def merge_intersect(self, *others) -> None:
        """Merge into self keeping options that appear in all others"""
        raise NotImplementedError()

    def remove(self, lookup_fn: Callable[['Schema'], bool]) -> None:
        """Remove options according to lookup function"""
        raise NotImplementedError()

    def sort_options(self, objective_fn: Callable[['Schema'], bool]) -> None:
        """Sort options according to objectiv function"""
        raise NotImplementedError()

    def reduce(self) -> None:
        """Remove options based on top-k reduce values in Links"""
        raise NotImplementedError()

    @recursive_repr()
    def __repr__(self) -> str:
        kwargs = ", ".join("{}={!r}".format(k, v) for k, v in sorted(self.kwargs.items()))
        format_string = "{module}.{cls}({callable}, {kwargs})"
        return format_string.format(module=self.__class__.__module__,
                                    cls=self.__class__.__qualname__,
                                    tag=self.created_with_tag,
                                    callable=self.callable,
                                    factory_method=self.factory_method,
                                    kwargs=kwargs)


def add_callable_from_yaml(from_yaml_fn: Callable, callable: Callable) -> Callable:
    """Add callable to call on from_yaml"""
    @functools.wraps(from_yaml_fn)
    def wrapped(constructor: Any, node: Any, factory_name: str, tag: str) -> Any:
        obj = from_yaml_fn(constructor, node, factory_name, tag, callable)
        return obj
    return wrapped


class Schematic(Registrable, should_register=False):

    def __init_subclass__(cls: Type['Registrable'],
                          **kwargs: Mapping[str, Any]) -> None:
        # Schema from_yaml function is generic, so need to add
        # class information
        from_yaml_fn = add_callable_from_yaml(Schema.from_yaml, callable=cls)
        to_yaml_fn = Schema.to_yaml
        super().__init_subclass__(from_yaml=from_yaml_fn, to_yaml=to_yaml_fn, **kwargs)
