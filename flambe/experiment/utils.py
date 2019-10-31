import copy
import os
from collections import abc
from typing import Dict, List, Mapping, Any, Optional, Iterable, Set, Sequence, MutableMapping

import torch
from ruamel.yaml.compat import StringIO
from ruamel import yaml as original_yaml
from ray.tune.trial import Trial
from ray.tune.suggest import SearchAlgorithm
from ray.tune.schedulers import TrialScheduler

from flambe.compile import Link, Component
from flambe.compile import Schema as Schema
from flambe.runnable.error import LinkError, SearchComponentError
from flambe.experiment.options import Options, GridSearchOptions, SampledUniformSearchOptions


def check_links(blocks: Dict[str, Schema],
                global_vars: Optional[Dict[str, Any]] = None) -> None:
    """Check validity of links between blocks.

    Ensures dependency order, and that only Comparable
    blocks are being reduced through a LinkBest object.

    Parameters
    ----------
    blocks : OrderedDict[str, Schema[Component]]
        The blocks to check, in order

    Raises
    ------
    LinkError
        On undeclared blocks (i.e not the right config order)
    ProtocolError
        Attempt to reduce a non-comparable block

    """
    visited: Set[str] = set()
    if global_vars is not None:
        visited |= global_vars.keys()

    def helper(block):
        """Explore block"""
        for _, value in block.items():
            # Check link order
            if isinstance(value, Link):
                target_block_id = value.root_schema
                if target_block_id not in visited:
                    raise LinkError(block_id, target_block_id)

            # Check recurse
            if isinstance(value, Mapping):
                helper(value)

    for block_id, block in blocks.items():
        visited.add(block_id)
        helper(block)


def check_search(blocks: Dict[str, Schema],
                 search: Mapping[str, SearchAlgorithm],
                 schedulers: Mapping[str, TrialScheduler]):
    """Check validity of links between blocks.

    Ensures dependency order, and that only Comparable
    blocks are being reduced through a LinkBest object.

    Parameters
    ----------
    blocks : OrderedDict[str, Schema[Component]]
        Ordered mapping from block id to a schema of the block
    search : Mapping[str, SearchAlgorithm], optional
        Map from block id to hyperparameter search space generator
    schedulers : Mapping[str, TrialScheduler], optional
        Map from block id to search scheduler

    Raises
    ------
    ProtocolError
        Non computable block assigned a search or scheduler.
    ProtocolError
        Non comparable block assigned a non default search or scheduler

    """
    hyper_blocks = list(search.keys()) + list(schedulers.keys())

    for block_id in hyper_blocks:
        # Check all hyper blocks are computable
        block_type = blocks[block_id].component_subclass
        if not issubclass(block_type, Component):  # type: ignore
            raise SearchComponentError(block_id)


def convert_tune(data: Any):
    """Convert the options and links in the block.

    Convert Option objects to tune.grid_search or
    tune.sample_from functions, depending on the type.

    Parameters
    ----------
    data : Any
        Input object that may contain Options objects that should be
        converted to a Tune-compatible representation

    """
    if isinstance(data, Options) or isinstance(data, Link):
        return data.convert()
    elif isinstance(data, dict):
        return {k: convert_tune(v) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        return [convert_tune(el) for el in data]
    elif isinstance(data, Options):
        if hasattr(data, 'elements'):  # TODO: Bit hacky, make this better
            out = copy.deepcopy(data)
            out.elements = [convert_tune(elm) for elm in data.elements]  # type: ignore
            return out
    return data


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
        if isinstance(value, abc.Mapping):
            yield from traverse(value, path + [key])
        else:
            yield path, key, value


def traverse_all(obj: Any) -> Iterable[Any]:
    """Iterate over all nested mappings and sequences

    Parameters
    ----------
    obj: Any

    Returns
    -------
    Iterable[Any]
        Iterable of child values to obj

    """
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from traverse_all(value)
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        for value in obj:
            yield from traverse_all(value)
    else:
        yield obj


def traverse_spec(nested: Mapping[str, Any], path: Optional[List[str]] = None) -> Iterable[Any]:
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
            yield from traverse_spec(value, path + [key])
        else:
            yield "spec.config." + ".".join(path), key, value


def update_nested(nested: MutableMapping[str, Any],
                  prefix: Iterable[str],
                  key: str,
                  new_value: Any) -> None:
    """Multi-level set operation for nested mapping.

    Parameters
    ----------
    nested : Mapping[str, Any]
        Nested dictionary where keys are all strings
    prefix : Iterable[str]
        List of keys specifying path to value to be updated
    key : str
        Final key corresponding to value to be updated
    new_value : Any
        New value to set for `[p1]...[key]` in `nested`

    """
    current = nested
    for step in prefix:
        current = current[step]  # type: ignore
    current[key] = new_value  # type: ignore


def get_nested(nested: Mapping[str, Any], prefix: Iterable[str], key: str) -> Any:
    """Get nested value in standard Mapping.

    Parameters
    ----------
    nested : Mapping[str, Any]
        The mapping to index in
    prefix : Iterable[str]
        The path to the final key in the nested input
    key : str
        The key to query

    Returns
    -------
    Any
        The value at the given path and key

    """
    current = nested
    for step in prefix:
        current = current[step]
    return current[key]


def update_schema_with_params(schema: Schema, params: Dict[str, Any]) -> Schema:
    """Replace options in the schema recursivly.

    Parameters
    ----------
    schema : Schema[Any]
        The schema object to update
    params : Dict[str, Any]
        The corresponding nested diciontary with values

    Returns
    -------
    Schema[Any]
        The update schema (same object as the input, not a copy)

    """
    for path, key, value in traverse(schema):
        if isinstance(value, Options):
            selected_value = get_nested(params, path, key)
            update_nested(schema, path, key, selected_value)
    # Return schema for chaining purposes
    return schema


def has_schemas_or_options(x: Any) -> bool:
    """Check if object contains Schemas or Options.

    Recurses for Mappings and Sequences

    Parameters
    ----------
    x : Any
        Input object to check for Schemas and Options

    Returns
    -------
    bool
        True iff contains any Options or Schemas.

    """
    if isinstance(x, Schema):
        return True
    elif (isinstance(x, GridSearchOptions) or isinstance(x, SampledUniformSearchOptions)):
        return True
    elif isinstance(x, Mapping):
        return any(has_schemas_or_options(v) for k, v in x.items())
    elif isinstance(x, Sequence) and not isinstance(x, str):
        return any(has_schemas_or_options(e) for e in x)
    else:
        return False


def divide_nested_grid_search_options(
        config: MutableMapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Divide config into a config Iterable to remove nested Options.

    For every GridSearchOptions or SampledUniformSearchOptions, if any
    values contain more Options or Schemas, create copies with a
    single value selected in place of the option. Resulting configs
    will have no nested options.

    Parameters
    ----------
    config : MutableMapping[str, Any]
        MutableMapping (or Schema) containing Options and Schemas

    Returns
    -------
    Iterable[Mapping[str, Any]]
        Each Mapping contains variants from original config without
        nested options

    """
    no_options_yielded = True
    for prefix, key, value in traverse(config):
        if (isinstance(value, GridSearchOptions) or
                isinstance(value, SampledUniformSearchOptions)) and \
                any(has_schemas_or_options(x) for x in value):
            no_options_yielded = False
            for option in value:
                config_variant = copy.deepcopy(config)
                # Create copy that has one selected value
                update_nested(config_variant, prefix, key, option)
                # Continue yielding to select other values
                yield from divide_nested_grid_search_options(config_variant)
            return
    if no_options_yielded:
        yield config


def extract_dict(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn the schema into a dictionary, ignoring types.

    NOTE: We recurse if any value is itself a `Schema`, a `Sequence`
    of `Schema`s, or a `Mapping` of `Schema`s. Other unconvential
    collections will not be inspected.

    Parameters
    ----------
    schema: Schema
        The object to be converted into a dictionary

    Returns
    -------
    Dict
        The output dictionary representation.

    """
    def helper(obj):
        if isinstance(obj, Schema):
            out = helper(obj.keywords)
        elif isinstance(obj, Link):
            out = obj
        elif isinstance(obj, Options):
            if hasattr(obj, 'elements'):  # TODO: Bit hacky, make this better
                out = copy.deepcopy(obj)
                out.elements = [helper(elm) for elm in obj]
            else:
                out = obj
        elif isinstance(obj, list) or isinstance(obj, tuple):
            out = [helper(elm) for elm in obj]
        elif isinstance(obj, abc.Mapping):
            out = {k: helper(v) for k, v in obj.items()}
        else:
            out = obj

        return out

    return helper(config)


def extract_needed_blocks(schemas: Dict[str, Schema],
                          block_id: str,
                          global_vars: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Returns the set of all blocks that the input block links to.

    Parameters
    ----------
    schemas : Dict[str, Schema[Any]]
        Map from `block_id` to `Schema` object
    block_id : str
        The block containing links

    Returns
    -------
    List[str]
        The list of ancestor block ids

    """
    needed = set()
    this_block = schemas[block_id]

    # Get this block's links
    for _, _, value in traverse(this_block):
        if isinstance(value, Link) and value.root_schema != block_id:
            # Ensure intra-block links are not added to prevent inf loop
            needed.add(value.root_schema)
        elif isinstance(value, Iterable):
            for element in value:
                if isinstance(element, Link) and element.root_schema != block_id:
                    needed.add(element.root_schema)

    # Reccurse through the new needed blocks
    for linked_block_id in needed.copy():
        if linked_block_id not in schemas.keys():
            if global_vars is None or linked_block_id not in global_vars.keys():
                raise LinkError(block_id, linked_block_id)
        else:
            needed |= extract_needed_blocks(schemas, linked_block_id, global_vars)
    needed |= {block_id}
    return needed


def update_link_refs(schemas: Dict[str, Schema],
                     block_id: str,
                     global_vars: Dict[str, Any]) -> None:
    """Resolve links in schemas at `block_id`.

    Parameters
    ----------
    schemas : Dict[str, Schema[Any]]
        Map from `block_id` to `Schema` object
    block_id : str
        The block where links should be activated
    global_vars: Dict[str, Any]
        The environment links (ex: resources)

    """
    this_block = schemas[block_id]
    for value in traverse_all(this_block):
        if isinstance(value, Link):
            if value.root_schema in schemas:
                value.target = schemas[value.root_schema]
                if isinstance(value.target, Component):
                    value.target = value.target._schema
            elif value.root_schema in global_vars:
                value.target = global_vars[value.root_schema]
                value.local = False


def get_best_trials(trials: List[Trial], topk: int, metric='episode_reward_mean') -> List[Trial]:
    """Get the trials with the best result.

    Parameters
    ----------
    trials : List[ray.tune.Trial]
        The list of trials to examine
    topk : int
        The number of trials to reduce to
    metric : str, optional
        The metric used in comparaison (higher is better)

    Returns
    -------
    List[ray.tune.Trial]
        The list of best trials

    """
    if topk <= 0:
        return []

    sorted_trials = sorted(trials, key=lambda t: t.last_result.get(metric, 0), reverse=True)
    return sorted_trials[:topk]


def get_non_remote_config(experiment):
    """Returns a copy of the original config file without
    the remote configuration

    Parameters
    ----------
    experiment : Experiment
        The experiment object

    """
    new_experiment = copy.deepcopy(experiment)
    # Remove manager
    experiment.manager = None

    with StringIO() as s:
        native_yaml = original_yaml.YAML()
        native_yaml.dump(new_experiment, s)
        return s.getvalue()


def local_has_gpu() -> bool:
    """Returns is local process has GPU

    Returns
    -------
    bool

    """
    return torch.cuda.is_available()


def rel_to_abs_paths(d: Dict[str, str]) -> Dict[str, str]:
    """Convert relative paths to absolute paths.

    Parameters
    ----------
    d: Dict[str, str]
        A dict from name -> path.

    Returns
    -------
    Dict[str, str]
        The same dict received as parameter with relative paths
        replaced with absolute.

    """
    ret = d.copy()
    for k, v in ret.items():
        if os.path.exists(v) and not os.path.isabs(v):
            ret[k] = os.path.abspath(v)
    return ret


def shutdown_ray_node() -> int:
    """Call 'ray stop' locally to terminate
    the ray node.

    """
    return os.system("bash -lc 'ray stop'")


def shutdown_remote_ray_node(host: str,
                             user: str,
                             key: str) -> int:
    """Execute 'ray stop' on a remote machine through ssh to
    terminate the ray node.

    IMPORTANT: this method is intended to be run in the cluster.

    Parameters
    ----------
    host: str
        The Orchestrator's IP that is visible by the factories
        (usually the private IP)
    user: str
        The username for that machine.
    key: str
        The key that communicate with the machine.

    """
    cmd = f"ssh -i {key} -o StrictHostKeyChecking=no {user}@{host} \"bash -lc 'ray stop'\""
    return os.system(cmd)
