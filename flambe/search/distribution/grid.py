from typing import Any, Sequence

from flambe.compile import alias
from flambe.search.distribution.distribution import Distribution


@alias('g')
class Grid(Distribution):
    """Discrete set of values used for grid search

    Defines a finite, discrete set of values to be substituted
    at the location where the set currently resides in the config

    """

    var_type = 'choice'
    is_numerical = False

    def __init__(self, options: Sequence) -> None:
        self.options = options

    def sample(self):
        raise ValueError("Grid options are not meant to be sampled.")

    def __getitem__(self, key: Any) -> Any:
        return self.options[key]

    def __iter__(self):
        yield from self.options

    def __len__(self) -> int:
        return len(self.options)

    def __repr__(self) -> str:
        return 'gridoptions(' + repr(self.options) + ')'
