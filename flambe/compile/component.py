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
from flambe.compile.registry import registrable_factory
from flambe.compile.schema import Schematic
from flambe.compile.const import STATE_DICT_DELIMETER, FLAMBE_SOURCE_KEY, FLAMBE_CLASS_KEY, \
    FLAMBE_CONFIG_KEY, FLAMBE_DIRECTORIES_KEY, KEEP_VARS_KEY, VERSION_KEY, FLAMBE_STASH_KEY


_EMPTY = inspect.Parameter.empty
A = TypeVar('A')
C = TypeVar('C', bound="Component")

logger = logging.getLogger(__name__)


class Component(Schematic, should_register=False):
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
        if isinstance(self, torch.nn.Module):
            self._register_state_dict_hook(self._state_dict_hook)
            self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

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
            if isinstance(attr, Component) and not isinstance(attr, (
                    torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler)):  # type: ignore
                current_path = prefix + name
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
        # Iterate over modules to make sure Component
        # nn.Modules are added to flambe directories
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
        flambe_save(self, path)
