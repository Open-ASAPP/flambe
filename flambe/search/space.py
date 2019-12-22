from typing import Dict

import numpy as np

from flambe.search.distribution import Distribution


class Space(object):
    """Search space object.

    Manages the various dimensions that the hyperparameters are in.
    This object is used internally by the search aglrothms and should
    not be used directly.


    """

    def __init__(self, distributions: Dict[str, Distribution]):
        '''
        Initialize the search space with a list of search distributions.

        search_dists: List of search distributions, e.g.

            from hypertune.space import Choice, Uniform
            search_dists = [Choice('hp1', [5, 6, 7]),
                     Uniform('hp2', low=3, high=7, transform='pow10')]
        '''
        self.distributions = distributions
        self.var_names = list(distributions.keys())
        self.dists = [distributions[name] for name in self.var_names]
        self.n_vars = len(self.dists)

        self.var_types = [dist.var_type for dist in self.dists]
        self.var_bounds = [(dist.var_min, dist.var_max) if dist.is_numerical
                           else (np.nan, np.nan) for dist in self.dists]

        self.name2idx = {name: i for i, name in enumerate(self.var_names)}

        # Check that all var names are unique
        assert len(self.name2idx.keys()) == self.n_vars

    def sample(self):
        '''
        Randomly sample from the space.
        '''
        samp = {}
        for name, dist in self.distributions.items():
            samp[name] = dist.sample()
        return samp

    def sample_raw_dist(self):
        '''
        Randomly sample from the raw distributions
        without applying transforms.
        '''
        samp = {}
        for name, dist in self.distributions.items():
            if dist.is_numerical:
                s = dist.sample_raw_dist()
            else:
                s = dist.sample()
            samp[name] = s
        return samp

    def apply_transform(self, samp):
        '''
        Apply transforms to a sample drawn from the space
        using sample_dist.

        samp: Dictionary of hyperparameter values, e.g.
                    samp = {'hp1': 7, 'hp2': 3}
        '''
        transformed = {}
        for var_name in samp.keys():
            dist = self.dists[self.name2idx[var_name]]
            if dist.is_numerical:
                t = dist.transform_fn(samp[var_name])
            else:
                t = samp[var_name]
            transformed[var_name] = t
        return transformed

    def round_to_space(self, hp_dict):
        '''
        Round hyperparameters to possible choices for ordered variables.

        hp_dict: Dictionary of hyperparameter values.
        '''
        rounded = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.var_type == 'discrete':
                val = dist.round_to_options(val)

            rounded[name] = val
        return rounded

    def normalize_to_space(self, hp_dict, return_as_list=False):
        '''
        Normalize hyperparameters to [0, 1] based on range for
        numerical variables and convert them to integers for
        categorical variables.

        hp_dict: Dictionary of hyperparameter values.
        return_as_list: Whether or not to return the hyperparameters as a list.
        '''
        normalized = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.is_numerical:
                val = dist.normalize_to_range(val)
            else:
                val = dist.option_to_int(val)
            normalized[name] = val

        if return_as_list:
            lst = []
            for name in self.var_names:
                if name in normalized.keys():
                    lst.append(normalized[name])
            return lst
        else:
            return normalized

    def unnormalize(self, hp_dict):
        '''
        Reverse the normalize_to_space function.

        hp_dict: Dictionary of normalized hyperparameter values.
        '''
        unnormalized = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.is_numerical:
                val = dist.unnormalize(val)
            else:
                val = dist.int_to_option(val)
            unnormalized[name] = val
        return unnormalized
