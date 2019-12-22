import numpy as np

from flambe.compile import alias
from flambe.search.distribution.numerical import Numerical


class Continuous(Numerical):
    '''
    Base continuous variable.
    '''
    var_type = 'continuous'


@alias("~u")
class Uniform(Continuous):

    def __init__(self, low, high, transform=None):
        '''
        low: Minimum value of variable.
        high: Maximum value of variable.
        '''
        if low > high:
            raise ValueError('Parameter `low`={} cannot be greater '
                             'than parameter `high`={}.'.format(low, high))
        super().__init__(var_min=low, var_max=high, transform=transform,
                         dist_params={'low': low, 'high': high})

    def sample_raw_dist(self):
        '''
        Sample from the distribution.
        '''
        return np.random.uniform(**self.dist_params)


@alias("~n")
class Normal(Continuous):

    def __init__(self, loc, scale, transform=None):
        '''
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution.
        '''
        if scale < 0:
            raise ValueError('Parameter `scale`={} cannot be less than 0.'.format(scale))
        super().__init__(var_min=-np.inf, var_max=np.inf, transform=transform,
                         dist_params={'loc': loc, 'scale': scale})

    def sample_raw_dist(self):
        '''
        Sample from the distribution.
        '''
        return np.random.normal(**self.dist_params)


class Beta(Continuous):

    def __init__(self, a, b, transform=None):
        '''
        a: Shape parameter of the distribution.
        b: Scale parameter of the distribution.
        '''
        if a <= 0 or b <= 0:
            raise ValueError('Parameter `a`={} and `b`={} cannot be non-positive.'.format(a, b))
        super().__init__(var_min=0, var_max=1, transform=transform,
                         dist_params={'a': a, 'b': b})

    def sample_raw_dist(self):
        '''
        Sample from the distribution.
        '''
        return np.random.beta(**self.dist_params)
