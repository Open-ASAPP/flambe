import numpy as np

from flambe.compile import alias
from flambe.search.distribution.distribution import Distribution


@alias("~c")
class Choice(Distribution):

    var_type = 'choice'
    is_numerical = False

    def __init__(self, options, probs=None):
        '''
        choices: List of choices for discrete variable.
        probs: List of probabilities for the corresponding choices.
        '''
        self.n_options = len(options)
        self.options = np.array(options)
        if probs is None:
            self.probs = np.array([1 / self.n_options] * self.n_options)
        else:
            self.probs = np.array(probs)
            assert len(self.options) == len(self.probs)

    def sample(self):
        '''
        Sample from the categorical distribution.
        '''
        return np.random.choice(self.options, p=self.probs)

    def option_to_int(self, option):
        '''
        Convert the choice to a categorical integer.

        choice: One of the possible choices for the variable.
        '''
        idx = np.where(self.options == option)[0]
        if len(idx) == 0:
            raise ValueError('Value not found in choices!')
        else:
            return idx[0]

    def int_to_option(self, idx):
        '''
        Convert a categorical integer to a choice.

        int: A value in {0, ..., n_choices-1}.
        '''
        return self.options[idx]

    def __iter__(self):
        yield from list(self.options)

    def __len__(self):
        return len(self.options)
