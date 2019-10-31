# from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Any, Union, Dict

import numpy as np
from ray.tune import grid_search

from flambe.compile import Registrable, alias


Number = Union[float, int]


class Options(Registrable, ABC):

    @classmethod
    @abstractmethod
    def from_sequence(cls, options: Sequence[Any]) -> 'Options':
        """Construct an options class from a sequence of values

        Parameters
        ----------
        options : Sequence[Any]
            Discrete sequence that defines what values to search over

        Returns
        -------
        T
            Returns a subclass of DiscreteOptions

        """
        pass

    @abstractmethod
    def convert(self) -> Dict:
        """Convert the options to Ray Tune representation.

        Returns
        -------
        Dict
            The Ray Tune conversion

        """
        pass

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        return representer.represent_sequence(tag, node.elements)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> 'Options':
        args, = list(constructor.construct_yaml_seq(node))
        if factory_name is None or factory_name == 'from_sequence':
            return cls.from_sequence(args)  # type: ignore
        else:
            factory = getattr(cls, factory_name)
            return factory(args)


@alias('g')
class GridSearchOptions(Sequence[Any], Options):
    """Discrete set of values used for grid search

    Defines a finite, discrete set of values to be substituted
    at the location where the set currently resides in the config

    """

    def __init__(self, elements: Sequence[Any]) -> None:
        self.elements = elements

    @classmethod
    def from_sequence(cls, options: Sequence[Any]) -> 'GridSearchOptions':
        return cls(options)

    def convert(self) -> Dict:
        return grid_search(list(self.elements))

    def __getitem__(self, key: Any) -> Any:
        return self.elements[key]

    def __len__(self) -> int:
        return len(self.elements)

    def __repr__(self) -> str:
        return 'gridoptions(' + repr(self.elements) + ')'


@alias('s')
class SampledUniformSearchOptions(Sequence[Number], Options):
    """Yields k values from the range (low, high)

    Randomly yields k values from the range (low, high) to be
    substituted at the location where the class currently resides in
    the config

    """

    def __init__(self, low: Number, high: Number, k: int, decimals: int = 10) -> None:
        self.elements: Sequence[Number]
        k = int(k)
        if k < 1:
            raise ValueError('k (number of samples) must be >= 1')
        if isinstance(low, int) and isinstance(high, int):
            self.elements = list(map(int, np.random.randint(low, high, k)))
        else:
            self.elements = list(map(float, np.round(np.random.uniform(low, high, k), decimals)))
        self._low = low
        self._high = high
        self._k = k
        self._decimals = decimals

    @classmethod
    def from_sequence(cls, options: Sequence[Any]) -> 'SampledUniformSearchOptions':
        if len(options) < 2 or len(options) > 4:
            raise ValueError(f'Incorrect number of args for {cls.__name__} - received: {options}')
        return cls(*options)

    def convert(self) -> Dict:
        return grid_search(list(self.elements))

    def __getitem__(self, key: Any) -> Any:
        return self.elements[key]

    def __len__(self) -> int:
        return len(self.elements)

    def __repr__(self) -> str:
        return 'randomoptions(' + repr(self.elements) + ')'

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        return representer.represent_sequence('!g', node.elements)


@alias('cluster')
class ClusterResource(Registrable):

    def __init__(self, location: str) -> None:
        self.location = location

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        # NOTE: here we are using scalar even considering this
        # is a string. Other representers are not able to representer
        # correctly this information
        return representer.represent_scalar(tag, node.location)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> 'ClusterResource':
        value = constructor.construct_yaml_str(node)
        return cls(location=value)
