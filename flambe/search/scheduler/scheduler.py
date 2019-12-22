from abc import ABC, abstractmethod

from flambe.search.trial import Trial


class Scheduler(ABC):
    '''
    Base scheduler class.
    '''

    def __init__(self, max_steps=1, verbose=False):
        '''
        results_file: String to path that results will be logged.
        results_keys: The names of the results that will be stored.
        target_result: The key result for searchers to focus on.
        n_workers: The maximum number of trials to be released in parallel.
        verbose: Whether or not to print out the results.
        '''
        self.trials = {}
        self.max_steps = max_steps
        self.verbose = verbose

        self._propose_new_params = None
        self._register_results = None

    def link_searcher_fns(self, propose_new_params_fn, register_results_fn):
        '''
        Link the searcher's propose_new_hp and register_hps functions to the scheduler.
        '''
        self._propose_new_params = propose_new_params_fn
        self._register_results = register_results_fn

    @abstractmethod
    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
            ``True`` if the algorithm has terminated.

        """
        pass

    @abstractmethod
    def release_trials(self, n, trials_paused):
        '''
        Return a new trial for the user.

        worker_id: The id of the worker.
        '''
        pass

    @abstractmethod
    def update_trials(self, trial_ids):
        '''
        Update results from the user.

        worker_id: The id of the worker.
        trial_id: The id of the trial.
        results: Dictionary of results.
        '''
        pass

    def _create_trial(self, n_steps, **kwargs):
        '''
        Create a new trial.
        '''
        # Create trial
        trial_id, params = self._propose_new_params(**kwargs)
        if params is None:
            return None, None
        else:
            # TODO: pass n_steps?
            new_trial = Trial(params)
            return trial_id, new_trial
