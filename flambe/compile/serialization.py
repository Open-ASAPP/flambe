import logging
import os
from collections import OrderedDict
from typing import Dict, Any, Iterable, Tuple, Optional, Sequence, NamedTuple, List, Mapping
import tarfile
import tempfile

import dill
import torch

from flambe.compile.registrable import yaml
from flambe.compile.downloader import download_manager
from flambe.compile.extensions import import_modules, is_installed_module, install_extensions, \
    setup_default_modules

# Constants used for state representation & serialization
from flambe.compile.const import STATE_DICT_DELIMETER, FLAMBE_SOURCE_KEY, FLAMBE_CLASS_KEY, \
    FLAMBE_CONFIG_KEY, FLAMBE_DIRECTORIES_KEY, VERSION_KEY, \
    HIGHEST_SERIALIZATION_PROTOCOL_VERSION, DEFAULT_SERIALIZATION_PROTOCOL_VERSION, \
    DEFAULT_PROTOCOL, STATE_FILE_NAME, VERSION_FILE_NAME, SOURCE_FILE_NAME, CONFIG_FILE_NAME, \
    PROTOCOL_VERSION_FILE_NAME, FLAMBE_STASH_KEY, STASH_FILE_NAME


logger = logging.getLogger(__name__)


class LoadError(Exception):
    """Error thrown because of fatal error when loading"""


class SaveTreeNode(NamedTuple):
    """Tree representation corresponding to the directory save format"""

    state: Dict[str, Any]
    version: str
    class_name: str
    source_code: str
    config: str
    object_stash: Dict[str, Any]
    children: Dict[str, Any]  # Nested typing not supported yet


class State(OrderedDict):
    """A state object for Flambe."""

    _metadata: Dict[str, Any]  # TODO should be instance property


# Private Helpers

def _convert_to_tree(metadata: Dict[str, Any]) -> SaveTreeNode:
    root_state: Dict[str, Any] = OrderedDict()
    # PyTorch / flambe states need this property
    root_state._metadata = {}  # type: ignore
    tree = SaveTreeNode(state=root_state, version=metadata[''][VERSION_KEY],
                        class_name=metadata[''][FLAMBE_CLASS_KEY],
                        source_code=metadata[''][FLAMBE_SOURCE_KEY],
                        config=metadata[''].get(FLAMBE_CONFIG_KEY, ''),
                        object_stash=metadata[''].get(FLAMBE_STASH_KEY, {}),
                        children={})
    for compound_key in metadata[FLAMBE_DIRECTORIES_KEY]:
        if compound_key == '':
            continue
        current_dict = tree
        component_keys = compound_key.split(STATE_DICT_DELIMETER)
        last_node_i = 0
        for i in range(len(component_keys)):
            key = STATE_DICT_DELIMETER.join(component_keys[last_node_i:i + 1])
            if STATE_DICT_DELIMETER.join(component_keys[:i + 1]) \
                    not in metadata[FLAMBE_DIRECTORIES_KEY]:
                continue
            last_node_i += 1
            prefix = STATE_DICT_DELIMETER.join(component_keys[:i + 1])
            current_value = current_dict.children.get(key)
            if key not in current_dict.children:
                # nested key not yet created
                m = metadata[prefix]
                new_state: Dict[str, Any] = OrderedDict()
                new_state._metadata = {}  # type: ignore
                current_dict.children[key] = SaveTreeNode(state=new_state,
                                                          version=m[VERSION_KEY],
                                                          class_name=m[FLAMBE_CLASS_KEY],
                                                          source_code=m[FLAMBE_SOURCE_KEY],
                                                          config=m.get(FLAMBE_CONFIG_KEY, ''),
                                                          object_stash=m.get(FLAMBE_STASH_KEY, {}),
                                                          children={})
                current_dict = current_dict.children[key]
            elif isinstance(current_value, SaveTreeNode):
                # key was already created, descend a layer further
                current_dict = current_value
            else:
                # key was already created but shouldn't have been
                raise Exception()
        if not isinstance(current_dict, SaveTreeNode):
            raise Exception('current dict not save tree node')

    return tree


def _fetch_tree_item(save_tree: SaveTreeNode, key: Sequence[str]) -> Tuple[str, SaveTreeNode]:
    current = save_tree
    last_i = 0
    for i, _ in enumerate(key):
        current_key = STATE_DICT_DELIMETER.join(key[last_i:i + 1])
        if current_key in current.children:
            current = current.children[current_key]  # type: ignore
            last_i = i + 1
            current_key = STATE_DICT_DELIMETER.join(key[last_i:i + 1])
    return current_key, current


def _update_save_tree(save_tree: SaveTreeNode, key: Sequence[str], value: Any) -> None:
    current_key, current = _fetch_tree_item(save_tree, key)
    current.state[current_key] = value


def _update_save_tree_metadata(save_tree: SaveTreeNode, key: Sequence[str], value: Any) -> None:
    current_key, current = _fetch_tree_item(save_tree, key)
    current.state._metadata[current_key] = value  # type: ignore


def _traverse_all_nodes(save_tree: SaveTreeNode,
                        path: Optional[List[str]] = None
                        ) -> Iterable[Tuple[List[str], SaveTreeNode]]:
    if path is None:
        path = []
    yield path, save_tree
    for key, value in save_tree.children.items():
        yield from _traverse_all_nodes(value, path + [key])


def _extract_prefix(root, directory):
    if directory.startswith(root):
        return directory[len(root):].lstrip(os.sep).replace(os.sep, STATE_DICT_DELIMETER)
    else:
        raise Exception()  # TODO


def _prefix_keys(state, prefix):
    for key in set(state.keys()):
        val = state[key]
        del state[key]
        state[prefix + key] = val
    return state


def traverse(nested: Mapping[str, Any], path: Optional[List[str]] = None) -> Iterable[Any]:
    """Iterate over a nested mapping returning the path and key, value.

    Parameters
    ----------
    nested : Mapping[str, Any]
        Mapping where some values are also mappings that should be
        traversed
    path : List[str]
        List of keys that were used to reach the current mapping

    Returns
    -------
    Iterable[Any]
        Iterable of path, key, value triples

    """
    if path is None:
        path = []
    for key, value in nested.items():
        if isinstance(value, Mapping):
            yield from traverse(value, path + [key])
        else:
            yield path, key, value


def _update_link_refs(schema: Mapping) -> None:
    """Resolve links in schemas at `block_id`.

    Parameters
    ----------
    schema : Dict[str, Schema[Any]]
        Map from `block_id` to `Schema` object

    """
    from flambe.compile.component import Link
    for _, _, value in traverse(schema):
        if isinstance(value, Link):
            value.target = schema[value.root_schema]


# Public Serialization Functions


def save_state_to_file(state: State,
                       path: str,
                       compress: bool = False,
                       pickle_only: bool = False,
                       overwrite: bool = False,
                       pickle_module=dill,
                       pickle_protocol=DEFAULT_PROTOCOL) -> None:
    """Save state to given path

    By default the state will be saved in directory structure that
    mirrors the object hierarchy, so that you can later inspect the
    save file and load individual components more easily. If you would
    like to compress this directory structure using tar + gz, set
    `compress` to True. You can also use pickle to write
    a single output file, more similar to how PyTorch's save function
    operates.

    Parameters
    ----------
    state : State
        The state_dict as defined by PyTorch; a flat dictionary
        with compound keys separated by '.'
    path : str
        Location to save the file / save directory to; This should be a
        new non-existent path; if the path is an existing directory
        and it contains files an exception will be raised. This is
        because the path includes the final name of the save file (if
        using pickle or compress) or the final name of the save
        directory.
    compress : bool
        Whether to compress the save file / directory via tar + gz
    pickle_only : bool
        Use given pickle_module instead of the hiearchical save format
        (the default is False).
    overwrite : bool
        If true, overwrites the contents of the given path to create a
        directory at that location
    pickle_module : type
        Pickle module that has load and dump methods; dump should
        accpet a pickle_protocol parameter (the default is dill).
    pickle_protocol : type
        Pickle protocol to use; see pickle for more details (the
        default is 2).

    Raises
    ------
    ValueError
        If the given path exists, is a directory, and already contains
        some files.

    """
    if os.path.exists(path):
        if os.path.isdir(path):
            dir_contents = os.listdir(path)
            if len(dir_contents) != 0 and not overwrite:
                raise ValueError(f'The given path ({path}) points to an existing directory '
                                 f'containing files:\n{dir_contents}\n'
                                 'Please use a new path, or an existing directory without files, '
                                 'or specify overwrite=True.')
        else:
            if not overwrite:
                raise ValueError(f'The given path ({path}) points to an existing file. Specify '
                                 'overwrite=True to overwrite the file with a save file / dir.')
    if compress:
        original_path = path
        temp = tempfile.TemporaryDirectory()
        path = os.path.join(temp.name, os.path.basename(original_path))
    if pickle_only:
        head, tail = os.path.split(path)
        if tail == '':
            path = head + '.pkl'
        else:
            path = path + '.pkl'
        if compress:
            orig_head, orig_tail = os.path.split(original_path)
            if orig_tail == '':
                original_path = orig_head + '.pkl'
            else:
                original_path = original_path + '.pkl'
        with open(path, 'wb') as f_pkl:
            pickle_module.dump(state, f_pkl, protocol=pickle_protocol)
    else:
        save_tree = _convert_to_tree(state._metadata)
        for key in state.keys():
            _update_save_tree(save_tree, key.split(STATE_DICT_DELIMETER), state[key])
        for key in state._metadata.keys():
            _update_save_tree_metadata(save_tree, key.split(STATE_DICT_DELIMETER),
                                       state._metadata[key])
        for node_path, node in _traverse_all_nodes(save_tree):
            current_path = os.path.join(path, *node_path)
            if not os.path.isdir(current_path):
                os.makedirs(current_path, exist_ok=True)
            torch.save(node.state, os.path.join(current_path, STATE_FILE_NAME), pickle_module,
                       pickle_protocol)
            with open(os.path.join(current_path, VERSION_FILE_NAME), 'w') as f_version:
                version_info = f"{node.class_name}:{node.version}"
                f_version.write(version_info)
            with open(os.path.join(current_path, SOURCE_FILE_NAME), 'w') as f_source:
                f_source.write(node.source_code)
            with open(os.path.join(current_path, CONFIG_FILE_NAME), 'w') as f_config:
                f_config.write(node.config)
            with open(os.path.join(current_path, PROTOCOL_VERSION_FILE_NAME), 'w') as f_proto:
                f_proto.write(str(DEFAULT_SERIALIZATION_PROTOCOL_VERSION))
            with open(os.path.join(current_path, STASH_FILE_NAME), 'wb') as f_stash:
                torch.save(node.object_stash, f_stash, pickle_module, pickle_protocol)
    if compress:
        compressed_file_name = original_path + '.tar.gz'
        with tarfile.open(name=compressed_file_name, mode='w:gz') as tar_gz:
            tar_gz.add(path, arcname=os.path.basename(path))
        temp.cleanup()


# TODO fix type of object to be Component without circular dependency

def save(obj: Any,
         path: str,
         compress: bool = False,
         pickle_only: bool = False,
         overwrite: bool = False,
         pickle_module=dill,
         pickle_protocol=DEFAULT_PROTOCOL) -> None:
    """Save `Component` object to given path

    See `save_state_to_file` for a more detailed explanation

    Parameters
    ----------
    obj : Component
        The component to save.
    path : str
        Location to save the file / save directory to
    compress : bool
        Whether to compress the save file / directory via tar + gz
    pickle_only : bool
        Use given pickle_module instead of the hiearchical save format
        (the default is False).
    overwrite : bool
        If true, overwrites the contents of the given path to create a
        directory at that location
    pickle_module : type
        Pickle module that has load and dump methods; dump should
        accept a pickle_protocol parameter (the default is dill).
    pickle_protocol : type
        Pickle protocol to use; see pickle for more details (the
        default is 2).

    """
    state = obj.get_state()
    save_state_to_file(state, path, compress, pickle_only, overwrite,
                       pickle_module, pickle_protocol)


def load_state_from_file(path: str,
                         map_location=None,
                         pickle_module=dill,
                         **pickle_load_args) -> State:
    """Load state from the given path

    Loads a flambe save directory, pickled save object, or a compressed
    version of one of these two formats (using tar + gz). Will
    automatically infer the type of save format and if the directory
    structure is used, the serialization protocol version as well.

    Parameters
    ----------
    path : str
        Path to the save file or directory
    map_location : type
        Location (device) where items will be moved. ONLY used when the
        directory save format is used. See torch.load documentation for
        more details (the default is None).
    pickle_module : type
        Pickle module that has load and dump methods; dump should
        accept a pickle_protocol parameter (the default is dill).
    **pickle_load_args : type
        Additional args that `pickle_module` should use to load; see
        torch.load documentation for more details

    Returns
    -------
    State
        state_dict that can be loaded into a compatible Component

    """
    with download_manager(path) as path:
        state = State()
        state._metadata = OrderedDict({FLAMBE_DIRECTORIES_KEY: set()})
        temp = None
        try:
            if not os.path.isdir(path) and tarfile.is_tarfile(path):
                temp = tempfile.TemporaryDirectory()
                with tarfile.open(path, 'r:gz') as tar_gz:
                    tar_gz.extractall(path=temp.name)
                    expected_name = tar_gz.getnames()[0]
                path = os.path.join(temp.name, expected_name)
            if os.path.isdir(path):
                for current_dir, subdirs, files in os.walk(path):
                    prefix = _extract_prefix(path, current_dir)
                    protocol_version_file = os.path.join(current_dir, PROTOCOL_VERSION_FILE_NAME)
                    with open(protocol_version_file) as f_proto:
                        saved_protocol_version = int(f_proto.read())
                        if saved_protocol_version > HIGHEST_SERIALIZATION_PROTOCOL_VERSION:
                            raise Exception('This version of Flambe only supports serialization'
                                            f'protocol versions <= '
                                            f'{HIGHEST_SERIALIZATION_PROTOCOL_VERSION}. '
                                            'Found version '
                                            f'{saved_protocol_version} at {protocol_version_file}')
                    component_state = torch.load(os.path.join(current_dir, STATE_FILE_NAME),
                                                 map_location, pickle_module, **pickle_load_args)
                    with open(os.path.join(current_dir, VERSION_FILE_NAME)) as f_version:
                        version_info = f_version.read()
                        class_name, version = version_info.split(':')
                    with open(os.path.join(current_dir, SOURCE_FILE_NAME)) as f_source:
                        source = f_source.read()
                    with open(os.path.join(current_dir, CONFIG_FILE_NAME)) as f_config:
                        config = f_config.read()
                    with open(os.path.join(current_dir, STASH_FILE_NAME), 'rb') as f_stash:
                        stash = torch.load(f_stash, map_location, pickle_module, **pickle_load_args)
                    local_metadata = {VERSION_KEY: version, FLAMBE_CLASS_KEY: class_name,
                                      FLAMBE_SOURCE_KEY: source, FLAMBE_CONFIG_KEY: config}
                    if len(stash) > 0:
                        local_metadata[FLAMBE_STASH_KEY] = stash
                    full_prefix = prefix + STATE_DICT_DELIMETER if prefix != '' else prefix
                    _prefix_keys(component_state, full_prefix)
                    state.update(component_state)
                    if hasattr(component_state, '_metadata'):
                        _prefix_keys(component_state._metadata, full_prefix)
                        # Load torch.nn.Module metadata
                        state._metadata.update(component_state._metadata)
                    # Load flambe.nn.Module metadata
                    state._metadata[prefix] = local_metadata
                    state._metadata[FLAMBE_DIRECTORIES_KEY].add(prefix)
            else:
                with open(path, 'rb') as f_pkl:
                    state = pickle_module.load(f_pkl)
        except Exception as e:
            raise e
        finally:
            if temp is not None:
                temp.cleanup()
        return state


def load(path: str,
         map_location=None,
         auto_install=False,
         pickle_module=dill,
         **pickle_load_args):
    """Load object with state from the given path

    Loads a flambe object by using the saved config files, and then
    loads the saved state into said object. See `load_state_from_file`
    for details regarding how the state is loaded from the save file or
    directory.

    Parameters
    ----------
    path : str
        Path to the save file or directory
    map_location : type
        Location (device) where items will be moved. ONLY used when the
        directory save format is used. See torch.load documentation for
        more details (the default is None).
    auto_install : bool
        If True, automatically installs extensions as needed.
    pickle_module : type
        Pickle module that has load and dump methods; dump should
        accept a pickle_protocol parameter (the default is dill).
    **pickle_load_args : type
        Additional args that `pickle_module` should use to load; see
        torch.load documentation for more details

    Returns
    -------
    Component
        object with both the architecture (config) and state that was
        saved to path

    Raises
    ------
    LoadError
        If a Component object is not loadable from the given path
        because extensions are not installed, or the config is empty,
        nonexistent, or otherwise invalid.

    """
    state = load_state_from_file(path, map_location, pickle_module, **pickle_load_args)
    yaml_config = state._metadata[''][FLAMBE_CONFIG_KEY]
    stash = state._metadata[''][FLAMBE_STASH_KEY] \
        if FLAMBE_STASH_KEY in state._metadata[''] else None
    setup_default_modules()
    yamls = list(yaml.load_all(yaml_config))

    if yamls is None:
        raise LoadError("Cannot load schema from empty config. This object may not have been saved"
                        " for any of the following reasons:\n - The object was not created from a"
                        "config or with compile method\n - The object originally linked to other"
                        "objects that cannot be represented in YAML")
    if len(yamls) > 2:
        raise LoadError(f"{os.path.join(path, CONFIG_FILE_NAME)} should contain an (optional) "
                        "extensions section and the main object.")
    if len(yamls) == 2:
        if yamls[0] is not None:
            extensions = dict(yamls[0])
            custom_modules = extensions.keys()
            for x in custom_modules:
                if not is_installed_module(x):
                    if auto_install:
                        logger.warn(f"auto_install==True, installing missing Module "
                                    f"{x}: {extensions[x]}")
                        install_extensions({x: extensions[x]})
                        logger.debug(f"Installed module {x} from {extensions[x]}")
                    else:
                        raise ImportError(
                            f"Module {x} is required and not installed. Please 'pip install'"
                            "the package containing the module or set auto_install flag"
                            " to True."
                        )
                import_modules([x])
                logger.debug(f"Automatically imported {x}")

            # Reload with extensions' module imported (and registered)
            schema = list(yaml.load_all(yaml_config))[1]

            # Set the extensions to the schema so that they are
            # passed when compiling the component.
            schema.add_extensions_metadata(extensions)
        else:
            schema = yamls[1]
    elif len(yamls) == 1:
        schema = yamls[0]
    else:
        raise LoadError("No config found at location; cannot load. Try just loading state with "
                        "the function 'load_state_from_file'")

    if schema is None:
        raise LoadError("Cannot load schema from empty config. This object may not have been saved"
                        " for any of the following reasons:\n - The object was not created from a"
                        "config or with compile method\n - The object originally linked to other"
                        "objects that cannot be represented in YAML")

    _update_link_refs(schema)
    # TODO: maybe replace with instance check if solution to circular
    # dependency with component is found

    try:
        instance = schema(stash)
    except TypeError:
        raise LoadError(f"Loaded object is not callable - likely because an extension is not "
                        f"installed. Check if {os.path.join(path, CONFIG_FILE_NAME)} has an "
                        f"extensions section at the top and install as necessary. Alternatively "
                        f"set auto_install=True")
    instance.load_state(state)
    return instance
