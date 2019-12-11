import inspect
from typing import MutableMapping, Any, Callable, Optional, Dict, Sequence, Tuple, List, Iterable
from warnings import warn

from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
                                  CommentedKeySeq, CommentedSeq, TaggedScalar,
                                  CommentedKeyMap)

from flambe.compile.common import function_defaults
from flambe.compile.registered_types import Registrable


YAML_TYPES = (CommentedMap, CommentedOrderedMap, CommentedSet, CommentedKeySeq, CommentedSeq,
              TaggedScalar, CommentedKeyMap)


class Schema(MutableMapping[str, Any]):

    def __init__(self,
                 callable: Callable,
                 kwargs: Dict[str, Any],
                 factory_name: Optional[str] = None,
                 tag: Optional[str] = None):
        self.callable = callable
        if factory_name is None:
            if not isinstance(self.callable, type):
                raise NotImplementedError('Using non-class callables with Schema is not yet supported')
            self.factory_method = callable
        else:
            if not isinstance(self.callable, type):
                raise Exception(f'Cannot specify factory name on non-class callable {callable}')
            self.factory_method = getattr(self.callable, factory_name)
        self.kwargs = function_defaults(self.factory_method).update(kwargs)
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
        self.initialize(path, cache)

    @classmethod
    def from_yaml(cls, callable: Callable, constructor: Any, node: Any, factory_name: str) -> Any:
        """Use constructor to create an instance of cls"""
        pass

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        pass

    def initialize(self,
                   path: Optional[List[str]] = None,
                   cache: Optional[Dict[str, Any]] = None) -> Any:
        cache = cache or {}
        path = path or []
        if path in cache:
            return cache[path]

        def helper(obj, current_path):
            if isinstance(obj, Link):
                return obj(cache)
            elif isinstance(obj, Schema):
                return obj(current_path, cache)
            elif isinstance(obj, dict):
                return {k: helper(v, current_path[:] + [k]) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [helper(x, current_path[:] + [str(i)]) for i, x in enumerate(obj)]
            else:
                return obj

        initialized_kwargs = helper(self.kwargs)
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

    def extract_copy(self, new_root: 'Schema') -> 'Schema':
        if self is new_root:
            return self
        # Recurse structure
        # Set ipath when find root, deep copy new root
        #  recurse on new root
        #  on each link if schematic path not child of ipath, do copy-by-value strategy
        #  else update schematic path
        # return new root

    def extract_search_space(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def iter_variants(self) -> 'Schema':
        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self)  # TODO


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


class Link(Registrable, tag_override="@"):

    def __init__(self,
                 schematic_path: Sequence[str],
                 attr_path: Optional[Sequence[str]] = None) -> None:
        self.schematic_path = schematic_path
        self.attr_path = attr_path
        ###
        self.target = None
        self.target_leaf: Optional[Schema] = None
        self.local = None
        self.resolved: Optional[Any] = None

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
        raise NotImplementedError()

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        raise NotImplementedError()

    def __call__(self, cache: Dict[str, Any]) -> Any:
        return self.resolve(cache)

    def __repr__(self) -> str:
        return f'link({create_link_str(self.schematic_path, self.attr_path)})'
