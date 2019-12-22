import numpy as np

from flamb.compile import alias
from flambe.search.distribution.numerical import Numerical


class Discrete(Numerical):
    '''
    Base discrete variable class that inherbits from numerical class.
    '''

    var_type = 'discrete'

    def __init__(self, options, probs=None, dist_params={}, transform=None):
        super().__init__(var_min=np.min(options), var_max=np.max(options),
                         transform=transform, dist_params=dist_params)
        self.n_options = len(options)
        self.options = np.array(options)
        if probs is None:
            self.probs = np.array([1 / self.n_options] * self.n_options)
        else:
            self.probs = np.array(probs)

    def round_to_options(self, val):
        '''
        Find the closest numerical choice for a given value.

        val: The given value.
        '''
        argmin = np.argmin([abs(val - c) for c in self.options])
        return self.options[argmin]

    def sample_raw_dist(self):
        '''
        Sample from the distribution.
        '''
        return np.random.choice(self.options, p=self.probs)


@alias("~qu")
class QUniform(Discrete):
    '''
    Quantized uniform variable class.
    '''

    def __init__(self, low, high, n, transform=None):
        '''
        low: Minimum value.
        high: Maximum value.
        n: Number of evenly-spaced choices between [low, high].
        '''
        if low > high:
            raise ValueError('Parameter `low`={} cannot be greater '
                             'than parameter `high`={}.'.format(low, high))
        super().__init__(options=np.linspace(low, high, n),
                         probs=np.array([1 / n] * n),
                         dist_params={'low': low, 'high': high, 'n': n},
                         transform=transform)
