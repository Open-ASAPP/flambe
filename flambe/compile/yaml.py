from typing import Callable, Optional, Any, Union, TextIO, Dict
import functools

import ruamel.yaml

from flambe.compile.registry import get_registry, Registry, register_class
from flambe.compile.registered_types import Tagged
from flambe.compile.extensions import import_modules, is_installed_module


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
        if hasattr(node, '_created_with_tag'):
            tag = node._created_with_tag
        else:
            raise Exception('')
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
                yaml.constructor.add_constructor(full_tag,
                                                 transform_from(entry.from_yaml, full_tag, factory))


def erase_registry_from_yaml(yaml, registry):
    for namespace, entry in registry:
        del yaml.representer.yaml_representers[entry.callable]
        for tag in entry.tags:
            full_tag = _combine_tag(namespace, tag)
            del yaml.constructor.yaml_constructors[full_tag]
            for factory in entry.factories:
                full_tag = _combine_tag(namespace, tag, factory)
                del yaml.constructor.yaml_constructors[full_tag]


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


def _load_environment(yaml_config: Any) -> Dict[str, Any]:
    vanilla_yaml = ruamel.yaml.YAML()
    try:
        yamls = list(vanilla_yaml.load_all(yaml_config))
    except TypeError as e:
        # TODO
        raise MalformedConfig('TODO more info on this error type')
    if len(yamls) > 2:
        raise ValueError(f"{len(yamls)} yaml streams found in file {yaml_config}. "
                         "A file should contain an (optional) environment section " +
                         "and the main runnable object (<= 2 streams separated by '---').")
    environment: Dict[str, Any] = {}
    if len(yamls) == 2:
        environment = dict(yamls[0])
    return environment

# TODO should we tighten signatures (below) Any -> schema?


def load_config(yaml_config: Union[TextIO, str]) -> Any:
    """Load yaml config using the flambe registry

    This function will read extensions from the YAML if any, update
    the registry, sync the registry with YAML, and then load the
    object from the YAML representation.

    Parameters
    ----------
    yaml_config: Union[TextIO, str]
        flambe config as a string, or file pointer (result from
        file.read())

    Returns
    -------
    Any
        Initialized object defined by the YAML config

    Raises
    -------
    MalformedConfig
        TODO
    ValueError
        If the file does not contain either 1 or 2 YAML documents
        separated by '---'

    """
    extensions = _load_environment(yaml_config)
    for module in extensions.keys():
        if not is_installed_module(module):
            raise ImportError(
                f"Module {x} is required and not installed. Please 'pip install'"
                "the package containing the module or set auto_install flag"
                " to True."
            )
    # TODO torch setup ?
    # setup_default_modules()
    # registry.add_extensions(extensions)
    import_modules(extensions.keys())
    registry = get_registry()
    from flambe.compile.schema import Schema, add_callable_from_yaml
    import torch
    from_yaml_fn = add_callable_from_yaml(Schema.from_yaml, callable=torch.nn.NLLLoss)
    to_yaml_fn = Schema.to_yaml
    register_class(torch.nn.NLLLoss, 'NLLLoss', from_yaml=from_yaml_fn, to_yaml=to_yaml_fn)
    with synced_yaml(registry) as yaml:
        _check_tags(yaml, yaml_config, registry, strict=True)
        result = list(yaml.load_all(yaml_config))[-1]
    return result


def load_config_from_file(file_path: str) -> Any:
    """Load config after reading it from the file path

    Parameters
    ----------
    file_path : str
        Location of YAML config file.

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
    with open(file_path) as f:
        result = load_config(f.read())
    return result


def dump_config(obj: Any, stream: Any, environment: Optional[Dict[str, Any]] = None):
    """Dump the given object into the stream including the environment.

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
    environment : Optional[Dict[str, Any]]
        Envrionment parameters.
        Default is None.

    """
    registry = get_registry()
    environment = environment or {}
    _check_extensions(environment.get('extensions', {}), registry, strict=True)
    # TODO check that all top level modules in object hierarchy are in
    #  the registry
    with synced_yaml(registry) as yaml:
        if len(environment) > 0:
            yaml.dump_all([environment, obj], stream)
        else:
            yaml.dump(obj, stream)


def load_environment(yaml_config: Union[TextIO, str]) -> Dict[str, Any]:
    """Load environment from a flambe YAML config file

    The environment should be the first document in a two document YAML
    file where the sections are separated by '---'. If no environment
    section is present, an empty dictionary will be returned.

    Parameters
    ----------
    yaml_config : Union[TextIO, str]
        flambe config as a string, or file path pointing to config

    Returns
    -------
    Dict[str, Any]
        A mapping from module names to package names (and versions).
        Default is None.

    Raises
    -------
    FileNotFoundError
        If the specified file is not found
    MalformedConfig
        TODO
    ValueError
        If the file does not contain 1 or 2 YAML documents
        separated by '---'

    """
    extensions = _load_environment(yaml_config)
    return extensions


def load_environment_from_file(file_path: str) -> Dict[str, Any]:
    """Load environment after reading it from the file path.

    Parameters
    ----------
    file_path : str
        Location of YAML config file.

    Returns
    -------
    Dict[str, Any]
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
    with open(file_path) as f:
        result = load_environment(f.read())
    return result
