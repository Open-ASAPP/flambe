import numpy as np

from flambe.search.searcher.searcher import ModelBasedSearcher


class MultiFidSearcher(ModelBasedSearcher):
    '''
    A searcher that allows for multiple fidelities --
    meant to be paired with HyperBand scheduler.
    '''

    def __init__(self, searcher_class, search_kwargs):
        '''
        searcher_class: A subclass of ModelBasedSearcher.
        search_kwargs: The parameters of the searcher_class.
        '''

        self.searcher_class = searcher_class
        self.search_kwargs = search_kwargs
        self.searchers = {}
        self.max_fid = -np.inf
        self.max_fid_searcher = self.searcher_class(**self.search_kwargs)

        super().__init__(min_configs_in_model=0)

    def _check_space(self, space):
        '''
        Check if the space is valid for this algorithm.

        space: A hypertune.space.Space object.
        '''
        self.max_fid_searcher._check_space(space)

    def assign_space(self, space):
        super().assign_space(space)
        self.max_fid_searcher.assign_space(space)

    def _propose_new_params_in_model_space(self):
        return self.max_fid_searcher._propose_new_params_in_model_space()

    def _apply_transform(self, params_in_model_space):
        return self.max_fid_searcher._apply_transform(params_in_model_space)

    def register_results(self, results, fids_dict):

        # Group param_ids by fidelity
        fids_groups = {}
        for param_id, fid in fids_dict.items():
            if fid in fids_groups:
                fids_groups[fid].add(param_id)
            else:
                fids_groups[fid] = set([param_id])

        for fid, params_ids in fids_groups.items():

            # Do not need to train the model if fidelity is less than max
            if fid < self.max_fid:
                continue

            if fid in self.searchers.keys():
                searcher = self.searchers[fid]
            else:
                searcher = self.searcher_class(**self.search_kwargs)
                searcher.assign_space(self.space)
                self.searchers[fid] = searcher

            # Transfer data from multifidelity searcher to subsearcher
            for params_id in params_ids:
                params_in_model_space = self.data[params_id]['params_in_model_space']
                searcher.data[params_id] = {'params_in_model_space': params_in_model_space,
                                            'result': None}

            filtered_results = {p_id: res for p_id, res in results.items() if p_id in params_ids}
            searcher.register_results(filtered_results)

            # Check if max fidelity searcher should be updated
            if fid > self.max_fid and searcher.has_enough_configs_for_model:
                self.max_fid = fid
                self.max_fid_searcher = searcher
