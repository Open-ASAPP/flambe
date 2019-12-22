
from typing import Tuple, Optional, Callable, Dict, List, Any
import os
import subprocess

import ray

import flambe as fl
from flambe.compile import Schema
from flambe.search.trial import Trial
from flambe.search.algorithm import Algorithm
from flambe.search.distribution import Grid


class Task(object):
    """Temporary dummy."""

    def step(self) -> bool:
        pass

    def metric(self) -> float:
        pass


class TaskAdapter:

    """Perform computation of a task."""

    def __init__(self, schema: Schema) -> None:
        """Initialize the Trial.

        Parameters
        ----------
        partial: partial[Task]

        """
        self.schema = schema
        self.task: Optional[Task] = None

    def step(self) -> Tuple[bool, Optional[float]]:
        """Run a step of the Trial"""
        if self.task is None:
            self.task = self.schema.compile()

        continue_ = self.task.step()
        metric = self.task.metric()
        return continue_, metric

    def kill(self) -> None:
        ray.actor.exit_actor()


class CallableAdapter:

    """Perform computation of a function."""

    def __init__(self, schema: Schema) -> None:
        """Initialize the Trial.

        Parameters
        ----------
        partial: partial[Task]

        """
        self.schema = schema
        self.callable: Optional[Callable] = None

    def step(self) -> Tuple[bool, Optional[float]]:
        """Run a step of the Trial"""
        self.callable = self.callable or self.schema()
        continue_ = False
        metric = self.callable()
        return continue_, metric

    def kill(self) -> None:
        ray.actor.exit_actor()


class Checkpointer(object):

    def __init__(self,
                 path: str,
                 host: str,
                 memory: bool = False,
                 local_dir: Optional[str] = None):

        self.object_id = None
        self.memory = memory
        self.path = os.path.join(path, '')
        self.remote = f"{host}:{self.path}"

    def get_checkpoint(self) -> Optional[Task]:
        if self.memory:
            if self.object_id is None:
                return None
            return ray.get(self.object_id)
        else:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            subprocess.run(f'rsync -a {self.remote} {self.path}')
            return fl.load(self.path)

    def set_checkpoint(self, task: Task):
        if self.memory:
            del self.object_id
            self.object_id = ray.put(task)
        else:
            fl.save(task, self.path)
            subprocess.run(f'rsync -a {self.path} {self.remote}')


class Search(object):
    """Implement a hyperparameter search over any schema.

    Use a Search to construct a hyperparameter search over any
    objects. Takes as input a schema to search, and algorithm,
    and resources to allocate per variant.

    Example
    -------

    >>> task = Schema(Task, arg1=uniform(0, 1), arg2=choice('a', 'b'))
    >>> search = Search(task, algorithm=Hyperband())
    >>> search.run()

    """
    def __init__(self,
                 schema: Schema,
                 algorithm: Algorithm,
                 cpus_per_trial: int = 1,
                 gpus_per_trial: int = 0,
                 ouput_dir: Optional[str] = None,
                 refresh_waitime: float = 1.0) -> None:
        """Initialize a hyperparameter search.

        Parameters
        ----------
        schema : Schema[Task]
            A schema of the callable or Task to search
        algorithm : Algorithm
            The hyperparameter search algorithm
        cpus_per_trial : int
            The number of cpu's to allocate per trial
        gpus_per_trial : int
                The number of gpu's to allocate per trial
        refresh_waitime : float, optional
            The minimum amount of time to wait before a refresh.
            Defaults ``10`` seconds.

        """
        self.output_dir = ouput_dir
        self.n_cpus = cpus_per_trial
        self.n_gpus = gpus_per_trial
        self.refresh_waitime = refresh_waitime

        total_resources = ray.cluster_resources()
        if self.n_cpus > 0 and self.n_cpus > total_resources['CPU']:
            raise ValueError("# of CPUs required per trial is larger than the total available.")
        elif self.n_gpus > 0 and self.n_gpus > total_resources['GPU']:
            raise ValueError("# of CPUs required per trial is larger than the total available.")

        # Check schema
        if any(isinstance(dist, Grid) for dist in schema.search_space()):
            raise ValueError("Schema cannot contain grid options, please split first.")

        self.schema = schema
        self.algorithm = algorithm

        if issubclass(self.schema._cls, Task):
            self.adatper = ray.remote(num_cpus=self.n_cpus, num_gpus=self.n_gpus)(TaskAdapter)
        elif isinstance(self.schema._cls, Callable):  # type: ignore
            self.adatper = ray.remote(num_cpus=self.n_cpus, num_gpus=self.n_gpus)(CallableAdapter)
        else:
            raise ValueError("Only functions and Task objects supported.")

    def run(self) -> List[Trial]:
        """Execute the search.

        Returns
        -------
        List[Trial]

        """
        self.algorithm.initialize(self.schema.search_space())

        running: List[int] = []
        finished: List[int] = []

        trials: Dict[str, Trial] = dict()
        id_to_trial_name: Dict[str, str] = dict()
        trial_name_to_actor: Dict[str, Any] = dict()

        while running or (not self.algorithm.is_done()):
            # Get all the current object ids running
            if running:
                finished, running = ray.wait(running, timeout=self.refresh_waitime)

            # Process finished trials
            for object_id in finished:
                _continue, metric = ray.get(object_id)
                trial = trials[id_to_trial_name[str(object_id)]]

                if _continue:
                    trial.set_metric(metric)
                    trial.set_has_result()
                else:
                    trial.set_terminated()

            # Update the algorithm
            max_queries = 0
            current_resources = ray.available_resources()
            if self.n_cpus > 0 and self.n_gpus > 0:
                n_possible_cpu = current_resources.get('CPU', 0) // self.n_cpus
                n_possible_gpu = current_resources.get('GPU', 0) // self.n_gpus
                max_queries = min(n_possible_cpu, n_possible_gpu)
            elif self.n_cpus > 0:
                max_queries = current_resources.get('CPU', 0) // self.n_cpus
            elif self.n_gpus > 0:
                max_queries = current_resources.get('GPU', 0) // self.n_gpus

            trials = self.algorithm.update(trials, maximum=max_queries)

            # Update based on trial status
            for name, trial in trials.items():
                # Handle creation and termination
                if trial.is_paused() or trial.is_running():
                    continue
                elif trial.is_terminated() and name in trial_name_to_actor:
                    del trial_name_to_actor[name]
                elif trial.is_created():
                    partial = self.schema.set_params(trial.parameters)
                    actor = self.adatper.remote(partial)
                    trial_name_to_actor[name] = actor
                    trial.set_resume()

                # Launch created and resumed
                if trial.is_resuming():
                    actor = trial_name_to_actor[name]
                    object_id = actor.step.remote()
                    running.append(object_id)
                    id_to_trial_name[str(object_id)] = name
                    trial.set_running()

        return list(trials.values())
