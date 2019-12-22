import numpy as np

from flambe.search.scheduler.scheduler import Scheduler


class HyperBandScheduler(Scheduler):

    def __init__(self, step_budget, max_steps, min_steps=1, drop_rate=3, verbose=False):
        '''
        n_workers: The maximum number of trials to be released in
            parallel.
        results_file: String to path that results will be logged.
        results_keys: The names of the results that will be stored.
        target_result: The key result for searchers to focus on.
        max_steps: Maximum number of resources to allocate to a single
            trial.
        min_steps: Minimum number of resources to allocate to a single
            trial.
        n_bracket_runs: Number of iterations of the outer loop of
        HyperBandto run.
        drop_rate: The factor by which trials are ended after each
        successive halving iteration.
        res_is_int: Whether or not to truncate resources to nearest
        integer.
        verbose: Whether or not to print out results.
        '''
        super().__init__(max_steps, verbose)
        self.min_steps = min_steps
        self.drop_rate = drop_rate

        # Initialize HyperBand params
        self.max_drops = int(np.log(max_steps / min_steps) / np.log(drop_rate))
        n_drops_grid = np.arange(self.max_drops + 1)
        self.n_trials_grid = np.ceil((self.max_drops + 1) / (n_drops_grid + 1) *
                                     (drop_rate**n_drops_grid)).astype('int')
        self.n_res_grid = max_steps / (drop_rate**n_drops_grid)

        self.step_budget = step_budget
        self.n_bracket_runs = self.step_budget // ((self.max_drops + 1) * self.max_steps)

        # Keep track of each bracket
        self.brackets = []
        for br in range(self.n_bracket_runs):
            s = self.max_drops - br % (self.max_drops + 1)
            self.brackets.append(Bracket(self.n_trials_grid[s], s))

        self.bracket_ids = {}

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
            ``True`` if the algorithm has terminated.

        """
        return all(bracket.has_finished for bracket in self.brackets)

    def release_trials(self, n, trials):
        '''
        Release a new trial with the corresponding worker id.

        worker_id: The id of the worker in {0, ..., n_workers}.
        '''
        for br, bracket in enumerate(self.brackets):
            s = bracket.max_halvings

            while n > 0 and (not bracket.has_finished) and bracket.has_pending:
                trial_id = bracket.get_pending()
                n_res_init = self.n_res_grid[s]

                if trial_id is None:
                    # Create new trial
                    trial_id, trial = self._create_trial(int(n_res_init))
                    trials[trial_id] = trial
                else:
                    # Resume previously paused trial
                    trial = trials[trial_id]
                    # n_res_curr = len(trial.metrics)
                    # n_res_total = n_res_init * (self.drop_rate**bracket.n_halvings)
                    # TODO trial.append_steps(int(n_res_total - n_res_curr))
                    trial.set_resume()

                self.bracket_ids[trial_id] = br
                n -= 1

        return trials

    def update_trials(self, trials):
        '''
        Update the algorithm with trial results.

        worker_id: The id of the worker in {0, ..., n_workers}.
        trial_id: String denoting the id of the trial.
        results: Dictionary of results to log.
        '''

        # Send results back to searcher
        trials_with_result = {trial_id: trial for trial_id,
                              trial in trials.items() if trial.has_result()}
        results_dict = {t_id: trial.best_metric for t_id, trial in trials_with_result.items()}
        fids_dict = {t_id: len(trial.metrics) for t_id, trial in trials_with_result.items()}
        self._register_results(results_dict, fids_dict=fids_dict)

        # Update brackets with results
        for trial_id, trial in trials_with_result.items():
            br = self.bracket_ids[trial_id]
            bracket = self.brackets[br]
            s = bracket.max_halvings
            bracket.record_finished(trial_id, trial.get_metric(-1))
            trial.set_paused()

            if bracket.is_ready_for_halving:
                n_trials_init = self.n_trials_grid[s]
                i = bracket.n_halvings + 1
                n_active = int(n_trials_init / (self.drop_rate**i))
                non_active_trial_ids = bracket.execute_halving(n_active)
                for trial_id in non_active_trial_ids:
                    trials[trial_id].set_terminated()

            elif bracket.has_finished:
                trial_ids = [x[0] for x in bracket.finished]
                for trial_id in trial_ids:
                    trials[trial_id].set_terminated()
        return trials


class Bracket:
    '''
    Internal hyperband data structure.
    '''

    def __init__(self, n_trials, max_halvings):
        self.n_trials = n_trials
        self.n_halvings = 0
        self.max_halvings = max_halvings
        self.pending = [None] * n_trials
        self.finished = []

    def get_pending(self):
        return self.pending.pop()

    def record_finished(self, trial, result):
        self.finished.append((trial, result))

    @property
    def has_pending(self):
        return len(self.pending) > 0

    @property
    def is_ready_for_halving(self):
        return (self.n_halvings < self.max_halvings) & (self.n_trials == len(self.finished))

    @property
    def has_finished(self):
        return (self.n_halvings == self.max_halvings) & (self.n_trials == len(self.finished))

    def execute_halving(self, n_active):
        trials_sorted = sorted(self.finished, key=lambda x: x[1], reverse=True)
        trials_sorted = [x[0] for x in trials_sorted]
        active_trials = trials_sorted[:n_active]
        non_active_trials = trials_sorted[n_active:]

        self.n_trials = n_active
        self.pending = active_trials
        np.random.shuffle(self.pending)

        self.finished = []

        self.n_halvings += 1
        return non_active_trials
