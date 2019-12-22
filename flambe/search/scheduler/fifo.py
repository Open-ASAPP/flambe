from flambe.search.scheduler.scheduler import Scheduler


class BlackBoxScheduler(Scheduler):
    '''
    Base scheduler class.
    '''

    def __init__(self, trial_budget, max_steps=1, verbose=False):
        '''
        budget: Maximum number of trials to create.
        results_file: String to path that results will be logged.
        results_keys: The names of the results that will be stored.
        target_result: The key result for searchers to focus on.
        n_workers: The maximum number of trials to be released in parallel.
        verbose: Whether or not to print out the results.
        '''
        super().__init__(max_steps, verbose)
        self.trial_budget = trial_budget
        self.num_released = 0
        self.done = False

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
            ``True`` if the algorithm has terminated.

        """
        return self.done or (self.num_released >= self.trial_budget)

    def release_trials(self, n, trials):
        '''
        Release a new trial with the corresponding worker id.

        worker_id: The id of the worker in {0, ..., n_workers}.
        '''
        while len(trials) < self.trial_budget and n > 0:
            trial_id, trial = self._create_trial(self.max_steps)
            if trial is None:
                self.done = True
                break
            else:
                assert trial_id not in trials.keys()
                trials[trial_id] = trial
                n -= 1
                self.num_released += 1
        return trials

    def update_trials(self, trials):
        '''
        Update the algorithm with trial results.

        worker_id: The id of the worker in {0, ..., n_workers}.
        trial_id: String denoting the id of the trial.
        results: Dictionary of results to log.
        '''
        trials_with_result = {trial_id: trial for trial_id,
                              trial in trials.items() if trial.has_result()}
        results_dict = {t_id: trial.best_metric for t_id, trial in trials_with_result.items()}
        self._register_results(results_dict)
        for trial in trials_with_result.values():
            trial.set_terminated()
        return trials
