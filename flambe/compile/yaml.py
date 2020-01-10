from typing import Callable, Optional, Any, Union, TextIO, Dict
import functools
from warnings import warn

import ruamel.yaml

from flambe.compile.registry import get_registry, Registry
from flambe.compile.registered_types import Tagged
from flambe.compile.extensions import import_modules


TAG_DELIMETER = '.'
TAG_BEGIN = '!'


class MalformedConfig(Exception):
    pass


def from_yaml(constructor: Any, node: Any, factory_name: str, tag: str, callable: Callable) -> Any:
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
            warn(f"No tag recorded for {node}, using default instead. "
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
        obj = from_yaml_fn(constructor, node, factory_name=factory_name, tag=tag)
        obj._created_with_tag = tag
        return obj
    return wrapped


def _combine_tag(*args):
    if len(args) == 0:
        raise ValueError('Must have >= 1 tag elements to combine')
    if args[0] == '':
        args = args[1:]
        if len(args) == 0:
            raise ValueError('Must have >= 1 tag elements to combine; only element was empty')
    if '' in args:
        raise ValueError('No tag elements can be empty strings except the first (root namespace)')
    raw_text = ''.join(args)
    if TAG_BEGIN in raw_text or TAG_DELIMETER in raw_text:
        raise ValueError('')
    return TAG_BEGIN + TAG_DELIMETER.join(args)


def _split_tag(tag) -> str:
    if tag[0] != TAG_BEGIN:
        raise ValueError(f'Invalid tag {tag}')
    tag_components = tag.split(TAG_BEGIN).split(TAG_DELIMETER)
    if len(tag_components) == 0:
        raise ValueError(f'Invalid tag {tag}')
    return tag_components


def sync_registry_with_yaml(yaml, registry):
    for namespace, entry in registry:
        yaml.representer.add_representer(entry.callable, transform_to(entry.to_yaml))
        for tag in entry.tags:
            full_tag = _combine_tag(namespace, tag)
            yaml.constructor.add_constructor(full_tag,
                                             transform_from(entry.from_yaml, full_tag, None))
            for factory in entry.factories:
                full_tag = _combine_tag(namespace, tag, factory)
                print(f"****   ****   adding tag {full_tag}")
                yaml.constructor.add_constructor(full_tag,
                                                 transform_from(entry.from_yaml, full_tag, factory))


def erase_registry_from_yaml(yaml, registry):
    pass  # TODO might need this because of the problems we've seen


class synced_yaml:

    def __init__(self, registry: Registry):
        self.registry = registry
        self.yaml = None

    def __enter__(self):
        self.yaml = ruamel.yaml.YAML()
        sync_registry_with_yaml(self.yaml, self.registry)
        return self.yaml

    def __exit__(self, *exc):
        erase_registry_from_yaml(self.yaml, self.registry)


def _contains_tag(tag: str, registry: Registry) -> bool:
    # TODO currently inefficient because amibguity of tag syntax
    #  i.e. which part is namespace / tag / factory is ambiguous
    for namespace, entry in get_registry():
        if tag == _combine_tag(namespace, entry.default_tag):
            return True
        for factory in entry.factories:
            if tag == _combine_tag(namespace, entry.default_tag, factory):
                return True
    return False


def _check_tags(yaml, stream: Any, registry: Registry, strict: bool = True) -> bool:
    """Checks if all tags in YAML are present in the registry.

    Parameters
    ----------
    yaml_str : str
        String containing YAML to check
    registry : Registry
        Flambe registry to check tags against
    strict : bool
        If true will raise an exception instead of a warning.
        (the default is True).

    Returns
    -------
    bool
        Whether all tags were in the registry or not

    Raises
    -------
    MalformedConfig
        If an unknown tag is detected and strict == True

    """
    # Check against tags in this config
    parsing_events = yaml.parse(stream)
    all_tags_in_registry = True
    for event in parsing_events:
        if hasattr(event, 'tag') and event.tag is not None:
            if not _contains_tag(event.tag, registry):
                msg = (f"Unknown tag: {event.tag}. Make sure the class or factory was correctly "
                       "registered.")
                if strict:
                    raise MalformedConfig(msg)
                else:
                    logger.warn(msg)
                    all_tags_in_registry = False
    return all_tags_in_registry


def _check_extensions(extensions: Dict[str, str], registry: Registry, strict: bool = True) -> bool:
    """Check if extensions are all present in the registry being used.

    Parameters
    ----------
    extensions : Dict[str, str]
        Mapping from module names to package names (and versions)
    registry : Registry
        Flambe registry to check against
    strict : bool
        If true, will raise exception instead of a warning.
        (the default is True).

    Returns
    -------
    bool
        True if all extensions were in the registry.

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    all_modules_registered = True
    for module_name in extensions.keys():
        if module_name not in registry.namespaces.keys():
            msg = (f"Module {module_name} not in registry. Make sure the module was imported and "
                   "contains at least one class or other callable registered.")
            if strict:
                raise Exception()
            else:
                logger.warn()
                all_modules_registered = False
    return all_modules_registered


# TODO should we tighten signatures (below) Any -> schema?


def load_config(yaml_config: Union[TextIO, str]) -> Any:
    """Load yaml config using the flambe registry

    This function will read extensions from the YAML if any, update
    the registry, sync the registry with YAML, and then load the
    object from the YAML representation.

    Parameters
    ----------
    yaml_config: Union[TextIO, str]
        Description of parameter `obj`.

    Returns
    -------
    Any
        Initialized object defined by the YAML config

    Raises
    -------
    FileNotFoundError
        If the specified file is not found
    MalformedConfig
        TODO
    ValueError
        If the file does not contain either 1 or 2 YAML documents
        separated by '---'

    """
    vanilla_yaml = ruamel.yaml.YAML()
    try:
        yamls = list(vanilla_yaml.load_all(yaml_config))
    except TypeError as e:
        # TODO
        raise MalformedConfig('TODO more info on this error type')
    if len(yamls) > 2:
        raise ValueError(f"{len(yamls)} yaml streams found in file {yaml_config}. "
                         "A file should contain an (optional) extensions section " +
                         "and the main runnable object (<= 2 streams separated by '---').")
    extensions: Dict[str, str] = {}
    if len(yamls) == 2:
        extensions = dict(yamls[0])
    for module in extensions.keys():
        if not is_installed_module(module):
            raise ImportError(
                f"Module {x} is required and not installed. Please 'pip install'"
                "the package containing the module or set auto_install flag"
                " to True."
            )
    # TODO torch setup ?
    # setup_default_modules()
    import_modules(extensions.keys())
    registry = get_registry()
    with synced_yaml(registry) as yaml:
        _check_tags(yaml, yaml_config, registry, strict=True)
        result = list(yaml.load_all(yaml_config))[-1]
    return result


def dump_config(obj: Any, stream: Any, extensions: Optional[Dict[str, str]] = None):
    """Dump the given object into the stream including any extensions

    Only dump objects that inherit from a Flambé class, or that have
    been manually registered in the registry.

    Parameters
    ----------
    obj : Any
        Object to be serialized into YAML. State will not be included,
        only kwargs used to initialize the object in the same way;
        Object must be serializable, meaning an entry exists in the
        registry.
    stream : Any
        File stream to dump to; will be forwarded to yaml.dump()
    extensions : Optional[Dict[str, str]]
        A mapping from module names to package names (and versions).
        Default is None.

    """
    registry = get_registry()
    _check_extensions(extensions, registry, strict=True)
    with synced_yaml(registry) as yaml:
        if len(extensions) > 0:
            yaml.dump_all([extensions, obj], stream)
        else:
            yaml.dump(obj, stream)
