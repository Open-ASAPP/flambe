import numpy as np

from flambe.search.searcher.searcher import Searcher


class RandomSearcher(Searcher):
    '''
    Random search samples parameterized distributions for
    hyperparameters.  At the moment, this searcher
    only supports an independent distribution for each hyperparameter.
    '''

    def __init__(self, seed=None):
        '''
        space: A hypertune.space.Space object.
        seed: Seed for the random generator.
        '''

        if seed:
            np.random.seed(seed)

        super().__init__()

    def _propose_new_params(self, **kwargs):
        '''
        Samples a single hyperparameter configuration
        from parameterized distributions.
        '''
        return self.space.sample()

    def register_results(self, results, **kwargs):
        pass
