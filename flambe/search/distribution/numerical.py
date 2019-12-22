from abc import abstractmethod

import numpy as np

from flambe.search.distribution.distribution import Distribution


class Numerical(Distribution):
    '''
    Base numerical variable class.
    '''

    is_numerical = True

    def __init__(self, var_min, var_max, transform=None,
                 dist_params={}):
        '''
        var_min: Minimum bound of variable.
        var_max: Maximum bound of variable.
        transform: String or custom function for transforming
        the variable after sampling.  Possible string options
        include: ['pow2', 'pow10', 'exp'].
        dist_params: Dictionary denoting the parameters of
        the distribution.
        '''
        self.var_min = var_min
        self.var_max = var_max

        if transform is None:
            self.transform_fn = lambda x: x
        elif transform == 'pow2':
            self.transform_fn = lambda x: 2**x
        elif transform == 'pow10':
            self.transform_fn = lambda x: 10**x
        elif transform == 'exp':
            self.transform_fn = lambda x: np.exp(x)
        else:
            self.transform_fn = transform

        self.dist_params = dist_params

    def sample(self):
        '''
        Sample from the transformed distribution.
        '''
        samp = self.sample_raw_dist()
        return self.transform_fn(samp)

    @abstractmethod
    def sample_raw_dist(self):
        '''
        Sample from the raw distribution.
        '''
        pass

    def normalize_to_range(self, val):
        '''
        Standardize the variable by its min and max bounds.

        val: Float value of the variable.
        '''
        if val < self.var_min or val > self.var_max:
            raise ValueError('Given value outside of range!')
        else:
            return (val - self.var_min) / (self.var_max - self.var_min)

    def unnormalize(self, val):
        '''
        Reverse the normalize_to_range function.

        val: Float value of the normalized variable.
        '''
        if val < 0 or val > 1:
            raise ValueError('Normalized value must be between [0, 1].')
        else:
            return (self.var_max - self.var_min) * val + self.var_min
