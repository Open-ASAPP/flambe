import itertools
import numpy as np

from flambe.search.searcher.searcher import Searcher


class GridSearcher(Searcher):
    '''
    Grid search does a brute force search over all
    discretized hyperparameter configurations.
    '''

    def assign_space(self, space):
        '''
        space: A hypertune.space.Space object.
        '''
        super().assign_space(space)

        params_options = {name: d.options for name, d in self.space.distributions.items()}
        self.params_sets = self._dict_to_cartesian_list(params_options)

    @classmethod
    def _check_space(self, space):
        '''
        Check if the space is valid for this algorithm.

        space: A hypertune.space.Space object.
        '''
        if np.any(np.array(space.var_types) == 'continuous'):
            raise ValueError('For grid search, all dimensions must be `discrete` or `choice`!')

    def _dict_to_cartesian_list(self, d):
        '''
        Generates a list of the Cartesian product between
        hyperparameter options contained in dictionary format.

        d: Dictionary of choices.
        '''
        lst = []
        for key in d:
            lst.append([(key, val) for val in d[key]])
        cartesian_lst = list(itertools.product(*lst))[::-1]
        return cartesian_lst

    def _propose_new_params(self):
        '''
        Returns a hyperparameter configuration from the
        list of Cartesian products.
        '''
        if bool(self.params_sets):
            new_params = dict(self.params_sets.pop())
            return new_params
        else:
            return None

    def register_results(self, results, **kwargs):
        pass
