from abc import ABC, abstractmethod


class Searcher(ABC):
    '''
    Base searcher class.  Searchers handle hyperparameter
    configuration selection and only  have access to a) the
    hyperparameters that need to be tuned and b) the performance of
    these hyperparameters.
    '''

    def __init__(self):
        self.space = None

    def assign_space(self, space):
        self._check_space(space)
        self.space = space
        self.n_configs_proposed = 0

    @classmethod
    def _check_space(cls, space):
        pass

    def propose_new_params(self, **kwargs):
        '''
        Proposes a new hyperparameter configuration based on
        internal searching algorithm.  All
        subclasses must implement this method.
        '''
        if self.space is None:
            raise ValueError('Space has not been assigned to searcher.')
        params_id = self.n_configs_proposed
        params = self._propose_new_params(**kwargs)
        if params is None:
            return None, None
        else:
            self.n_configs_proposed += 1
            return params_id, params

    @abstractmethod
    def _propose_new_params(self, **kwargs):
        pass

    @abstractmethod
    def register_results(self, results, **kwargs):
        pass


class ModelBasedSearcher(Searcher):

    def __init__(self, min_configs_in_model):
        '''
        space: A hypertune.space.Space object.
        min_points_in_model: Minimum number of points
        to collect before model building.
        '''
        super().__init__()
        self.min_configs_in_model = min_configs_in_model
        self.data = {}

    @property
    def n_configs_in_model(self):
        return sum([1 for datum in self.data.values() if datum['result'] is not None])

    @property
    def has_enough_configs_for_model(self):
        '''
        Check if the searcher has enough points to build a model.
        '''
        return self.n_configs_in_model >= self.min_configs_in_model

    def _propose_new_params(self, **kwargs):
        '''
        Propose a new hyperparameter configuration.
        Calls internal self._propose_new_model_hp
        function and applies the transforms from the space.
        Note that model building happens
        in the distribution space (not the transformed space).
        '''
        params_in_model_space = self._propose_new_params_in_model_space(**kwargs)
        params_id = self.n_configs_proposed
        self.data[params_id] = {'params_in_model_space': params_in_model_space,
                                'result': None}

        params = self._apply_transform(params_in_model_space)
        return params

    @abstractmethod
    def _propose_new_params_in_model_space(self, **kwargs):
        '''
        Subclasses must override this method that
        generates samples in the distribution space.
        '''
        if not self.has_enough_configs_for_model:
            return self.space.sample_raw_dist()

    @abstractmethod
    def _apply_transform(self, params_in_model_space):
        pass

    def register_results(self, results, **kwargs):
        '''
        Record results.

        hp_results: List of hyperparameters and associated results.
        '''
        for params_id, result in results.items():
            self.data[params_id]['result'] = result
