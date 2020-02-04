import inspect
from reprlib import recursive_repr
from typing import MutableMapping, Any, Callable, Optional, Dict, Sequence
from typing import Tuple, List, Iterable, Type, Mapping
from warnings import warn
import copy
import functools

from ruamel.yaml import ScalarNode
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
                                  CommentedKeySeq, CommentedSeq, TaggedScalar,
                                  CommentedKeyMap)

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


class Options(Registrable, should_register=False):
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


class Variants(Registrable, tag_override="v"):

    def __init__(self, options: Iterable[Any]):
        self.options = options

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        return representer.represent_sequence(tag, node.options)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str, tag: str) -> 'Link':
        # construct_yaml_seq returns wrapper tuple, need to unpack;
        # will also recurse so items in options can also be links
        options, = list(constructor.construct_yaml_seq(node))
        return Variants(options)


class GridVariants(Registrable, tag_override="g"):

    def __init__(self, options: Iterable[Any]):
        self.options = options

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        return representer.represent_sequence(tag, node.options)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str, tag: str) -> 'Link':
        # construct_yaml_seq returns wrapper tuple, need to unpack;
        # will also recurse so items in options can also be links
        options, = list(constructor.construct_yaml_seq(node))
        return GridVariants(options)


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
        self.schematic_path = tuple(schematic_path)
        self.attr_path = tuple(attr_path)

    def resolve(self, cache: Dict[str, Any]) -> Any:
        try:
            obj = cache[self.schematic_path]
        except KeyError:
            raise MalformedLinkError(f'Link for schema at {self.schematic_path} does not point to '
                                     'an object that has been initialized. '
                                     'Make sure the link points to a non-parent above the link '
                                     'in the config.')
        if self.attr_path is not None:
            for attr in self.attr_path:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    raise MalformedLinkError(f'Link {self} failed. {obj} has no attribute {attr}')
        return obj

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str, tag: str) -> 'Link':
        link_str = constructor.construct_scalar(node)
        return cls(link_str)

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        link_str = create_link_str(node.schematic_path, node.attr_path)
        return representer.represent_scalar(tag, link_str)

    def __call__(self, cache: Dict[str, Any]) -> Any:
        return self.resolve(cache)

    def __repr__(self) -> str:
        return f'link({create_link_str(self.schematic_path, self.attr_path)})'


class CopyLink(Link, tag_override='#'):

    def resolve(self, cache: Dict[str, Any]) -> Any:
        obj = super().resolve(cache)
        return copy.deepcopy(obj)


class Schema(MutableMapping[str, Any]):

    def __init__(self,
                 callable: Callable,
                 args: Optional[Sequence[Any]] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 factory_name: Optional[str] = None,
                 tag: Optional[str] = None):
        if not isinstance(callable, type):
            raise NotImplementedError('Using non-class callables with Schema is not yet supported')
        self.callable = callable
        registry = get_registry()
        if callable not in registry:
            pass
            # TODO auto-register logic? for functions?
            # registry.create(
            #   callable,
            #   namespace=get_class_namespace(callable),
            #   factories=[factory_name],
            #   from_yaml=add_callable_from_yaml(
            #       Schema.from_yaml,
            #       callable=callable
            #   ),
            #   to_yaml=Schema.to_yaml
            # )
        if factory_name is None:
            self.factory_method = callable
        else:
            if not isinstance(self.callable, type):
                raise ValueError(f'Cannot specify factory name on non-class callable {callable}')
            self.factory_method = getattr(self.callable, factory_name)
        args = args if args is not None else []
        kwargs = kwargs if kwargs is not None else {}
        s = inspect.signature(self.factory_method)
        self.bound_arguments = s.bind(*args, **kwargs)
        self.bound_arguments.apply_defaults()
        for k, v in self.bound_arguments.arguments.items():
            if s.parameters[k].kind not in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                            inspect.Parameter.KEYWORD_ONLY]:
                raise TypeError(f'Argument {k} of type {s.parameters[k].kind} not supported')
        if tag is None:
            if isinstance(self.callable, type):
                tag = type(self.callable).__name__
            else:
                tag = self.callable.__name__
        self.created_with_tag = tag

    @property
    def arguments(self):
        return self.bound_arguments.arguments

    def __setitem__(self, key: str, value: Any) -> None:
        print(f"     update bargs: {self.bound_arguments}")
        print(f"       id: {id(self.bound_arguments)}")
        self.bound_arguments.arguments[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.bound_arguments.arguments[key]

    def __delitem__(self, key: str) -> None:
        del self.bound_arguments.arguments[key]

    def __iter__(self) -> Iterable[str]:
        yield from self.bound_arguments.arguments

    def __len__(self) -> int:
        return len(self.bound_arguments.arguments)

    def __call__(self,
                 path: Optional[List[str]] = None,
                 cache: Optional[Dict[str, Any]] = None,
                 root: Optional['Schema'] = None):
        return self.initialize(path, cache, root)

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
            return cls(callable, factory_name=factory_name, tag=tag)
        kwargs, = list(constructor.construct_yaml_map(node))
        # TODO support constructing sequence for positional args
        return cls(callable, kwargs=kwargs, factory_name=factory_name, tag=tag)

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        pass

    @staticmethod
    def traverse(obj: Any,
                 current_path: Optional[List[str]] = None,
                 fn: Optional[Callable] = None,
                 yield_schema: Optional[str] = None) -> Iterable[Tuple[str, Any]]:
        current_path = current_path if current_path is not None else tuple()
        fn = fn if fn is not None else (lambda x: x)
        if isinstance(obj, Link):
            yield (current_path, obj)
        elif isinstance(obj, Schema):
            if yield_schema is None or yield_schema == 'before':
                yield (current_path, fn(obj))
                yield from Schema.traverse(obj.bound_arguments, current_path, fn, yield_schema)
            elif yield_schema == 'only':
                yield (current_path, fn(obj))
            elif yield_schema == 'after':
                yield from Schema.traverse(obj.bound_arguments, current_path, fn, yield_schema)
                yield (current_path, fn(obj))
            elif yield_schema == 'never':
                yield from Schema.traverse(obj.bound_arguments, current_path, fn, yield_schema)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                next_path = current_path + (k,)
                yield from Schema.traverse(v, next_path, fn, yield_schema)
        elif isinstance(obj, (list, tuple)):
            for i, e in enumerate(obj):
                next_path = current_path[:] + (str(i),)
                yield from Schema.traverse(e, next_path, fn, yield_schema)
        elif isinstance(obj, inspect.BoundArguments):
            params = obj.signature.parameters
            for k, v in obj.arguments.items():
                if params[k].kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      inspect.Parameter.KEYWORD_ONLY]:
                    next_path = current_path + (k,)
                    yield from Schema.traverse(v, next_path, fn, yield_schema)
                elif params[k].kind == inspect.Parameter.VAR_KEYWORD:
                    raise NotImplementedError('Variable keyword arguments not supported')
                else:
                    raise Exception(f'Invalid state for Schema. Invalid argument types at {obj}')
        else:
            yield (current_path, obj)

    def set_param(self, path: Optional[Tuple[str]], value: Any) -> None:
        """Set path in schema to value

        Convenience method for setting a value deep in a schema. For
        example `root.set_param(('a', 'b', 'c'), val)` is the
        equivalent of `root['a']['b']['c'] = val`. NOTE: you can only
        use set_param on existing paths in the schema. If `c` does not
        already exist in the above example, a `KeyError` is raised.

        Parameters
        ----------
        path : Optional[Tuple[str]]
            Description of parameter `path`.
        value : Any
            Description of parameter `value`.

        Raises
        -------
        KeyError
            If any value in the path does not exist as the name of a
            child schema

        """
        current_obj = self
        last_item = None
        try:
            for item in path[:-1]:
                last_item = item
                current_obj = current_obj[item]
            last_item = path[-1]
            if last_item not in current_obj:
                raise KeyError()
            current_obj[last_item] = value
        except KeyError:
            raise KeyError(f'{self} has no path {path}. Failed at {last_item}')

    def get_param(self, path: Optional[Tuple[str]]) -> Any:
        current_obj = self
        last_item = None
        try:
            for item in path[:-1]:
                last_item = item
                current_obj = current_obj[item]
            last_item = path[-1]
            return current_obj[last_item]
        except KeyError:
            raise KeyError(f'{self} has no path {path}. Failed at {last_item}')

    def initialize(self,
                   path: Optional[Tuple[str]] = None,
                   cache: Optional[Dict[str, Any]] = None,
                   root: Optional['Schema'] = None) -> Any:
        # Set defaults for values that will be used in recursion
        cache = cache if cache is not None else {}
        path = path if path is not None else tuple()
        new_root = root is None
        # Need to keep a reference to the first caller
        # for set_param updates later
        root = root if root is not None else copy.deepcopy(self)
        initialized = root if new_root else self
        # If object has already been initialized, return that value
        if path in cache:
            return cache[path]
        # Else, recursively initialize all children (bound arguments)
        for current_path, obj in self.traverse(
            initialized.bound_arguments,
            current_path=path,
            yield_schema='only'
        ):
            if isinstance(obj, Link):
                new_value = obj(cache=cache)
                cache[current_path] = new_value
                root.set_param(current_path, new_value)
            elif isinstance(obj, Schema):
                new_value = obj(path=current_path, cache=cache, root=root)
                cache[current_path] = new_value
                root.set_param(current_path, new_value)
            else:
                cache[current_path] = obj
        initialized_arguments = initialized.bound_arguments

        for k, v in initialized_arguments.arguments.items():
            if isinstance(v, YAML_TYPES):
                msg = f"keyword '{k}' is still yaml type {type(v)}\n"
                msg += f"This could be because of a typo or the class is not registered properly"
                warn(msg)
        try:
            cache[path] = initialized.factory_method(*initialized_arguments.args,
                                                     **initialized_arguments.kwargs)
        except TypeError as te:
            print(f"Constructor {self.factory_method} failed with "
                  f"arguments:\n{initialized_arguments.arguments.items()}")
            raise te
        return cache[path]

    def extract_search_space(self) -> Dict[Tuple[str, ...], Options]:
        search_space = {}

        for path, item in Schema.traverse(self, yield_schema='never'):
            if isinstance(item, Link):
                pass
            elif isinstance(item, Options):
                search_space[path] = item

        return search_space

    def extract_links(self) -> List[Link]:
        links = []

        for path, item in Schema.traverse(self, yield_schema='never'):
            if isinstance(item, Link):
                links.append(item)

        return links

    def set_from_search_space(self, search_space: Dict[Tuple[str, ...], Any]) -> None:
        for path, value in search_space.items():
            self.set_param(path, value)

    def iter_variants(self) -> 'Schema':
        """Yield variants selecting the parallel options from each"""
        for selection_index in range(self.num_options):
            variant_schema = copy.deepcopy(self)
            for path, item in Schema.traverse(self, yield_schema='never'):
                if isinstance(item, Variants):
                    value = item[selection_index]
                    variant_schema.set_param(path, value)
            yield variant_schema

    @recursive_repr()
    def __repr__(self) -> str:
        args = ", ".join("{}={!r}".format(k, v) for k, v in sorted(self.arguments.items()))
        format_string = "{module}.{cls}({callable}, {args})"
        return format_string.format(module=self.__class__.__module__,
                                    cls=self.__class__.__qualname__,
                                    tag=self.created_with_tag,
                                    callable=self.callable,
                                    factory_method=self.factory_method,
                                    args=args)


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
