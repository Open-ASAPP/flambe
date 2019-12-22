from flambe.search.distribution.distribution import Distribution
from flambe.search.distribution.grid import Grid
from flambe.search.distribution.choice import Choice
from flambe.search.distribution.continuous import Uniform, Normal, Beta
from flambe.search.distribution.discrete import QUniform


def grid(*args, **kwargs):
    return Grid(*args, **kwargs)


def choice(*args, **kwargs):
    return Choice(*args, **kwargs)


def normal(*args, **kwargs):
    return Normal(*args, **kwargs)


def beta(*args, **kwargs):
    return Beta(*args, **kwargs)


def uniform(*args, **kwargs):
    return Uniform(*args, **kwargs)


def quniform(*args, **kwargs):
    return QUniform(*args, **kwargs)


__all__ = ['Distribution', 'Uniform', 'Normal', 'Beta', 'uniform', 'normal', 'beta',
           'QUniform', 'quniform', 'Choice', 'choice', 'Grid', 'grid']
