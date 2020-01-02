import logging
from copy import deepcopy
from typing import Optional, Dict, Sequence, Union, Callable, List, NamedTuple

import ray

from flambe.compile import Schema
from flambe.search import Algorithm, Search, Trial


logger = logging.getLogger(__name__)


class Reduction(NamedTuple):
    """A reduction of the variants for source stage to k variants"""

    source: str
    k: int


class Pipeline(Schema):
    # move this to its own file if it gets big

    def __init__(self, tasks: Dict[str, Task]):
        self.tasks = tasks
        self.task = list(tasks.keys())[-1]

    def sub_pipeline(self, stage_name):
        """Return subset of the pipeline stages ending in stage_name

        The subset of pipeline stages will include all dependencies
        needed for stage the given stage and all their dependencies
        and so on.
        """
        # TODO actually prune not-needed deps
        sub_stages = {}
        for k, v in self.tasks:
            sub_stages[k] = v
            if k == stage_name:
                break
        return Pipeline(sub_stages)

    def dependencies(self):
        return self.keys()[:-1]

    def step(self):
        return self.task.step()

    def save_state(self):
        self.task.save_state()

    def load_state(self):
        self.task.load_state()


class Stage(object):

    def __init__(self,
                 name: str,
                 full_pipeline_id: int,
                 algorithm: Algorithm,
                 cpus_per_trial: int,
                 gpus_per_trial: int,
                 dependencies: List[List[Trial]],
                 reductions: List[Tuple[str, int]]):
        self.name = name
        self.full_pipeline_id = full_pipeline_id
        self.algorithm = algorithm
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.dependencies = dependencies
        self.reductions = reductions

    def run(self):
        pipeline = ray.get(self.full_pipeline_id)
        # Fetch dependencies
        results = ray.get(self.dependencies)

        # Mask out failed trials and apply reductions
        successful = []
        for trials in results:
            # Filter errors out
            trials = filter(lambda t: not t.is_error(), trials)
            # Select topk
            filtered = [r for r in self.reductions if r.source == result.stage_name]
            if filtered:
                min_reduction = min(r.k for r in filtered)
                trials = sorted(trials, key=lambda x: trials.best_metric(), reverse=True)[:k]
                successful.append(result.topk(min_reduction))
            else:
                successful.append(result)

        # Perform merging for conditional dependencies
        for result in successful:
            pass

        # Run remaining searches in parallel
        out_ids = []
        for pipeline in successful:
            # Construct and execute the search
            search = ray.remote(Search)(pipeline,
                                        self.algorithm,
                                        self.cpus_per_trial,
                                        self.gpus_per_trial)
            result = search.remote().run()

            # Upload results
            result_id = ray.put(result)
            out_ids.append(result_id)
        return out_ids


class Experiment(object):

    def __init__(self,
                 name: str,
                 pipeline: Optional[Dict[str, Schema]] = None,
                 resources: Optional[Dict[str, Union[str, ClusterResource]]] = None,
                 resume: Optional[Union[str, Sequence[str]]] = None,
                 debug: bool = False,
                 devices: Dict[str, int] = None,
                 save_path: Optional[str] = None,
                 algorithm: Optional[Dict[str, Algorithm]] = None,
                 reduce: Optional[Dict[str, int]] = None,
                 env: RemoteEnvironment = None,
                 max_failures: int = 1,
                 stop_on_failure: bool = True,
                 merge_plot: bool = True,
                 user_provider: Callable[[], str] = None) -> None:
        self.name = name
        self.save_path = save_path
        self.requirements = requirements or []

        self.pipeline = Pipeline(pipeline or dict())
        self.algorithms = algorithms
        self.reduce = reduce

    def add_stage(self,
                  name: str,
                  schema: Schema,
                  algorithm: Optional[Algorithm] = None,
                  reduce: Optional[int] = None,
                  n_cpus_per_trial: int = 1,
                  n_gpus_per_trial: int = 0) -> None:
        self.pipeline[name] = schema
        self.algorithms[name] = deepcopy(algorithm)
        self.reduce[name] = reduce
        self.resources[name] = {'cpu': n_cpus_per_trial, 'gpu': n_gpus_per_trial}

    def run(self, resume: bool = False) -> None:
        logger.info('Experiment started.')
        stage_to_result: Dict[str, int] = {}
        full_pipeline_id = ray.put(self.pipeline)

        for name in self.pipeline:
            # Get dependencies as a list of result object ids
            pipeline = self.pipeline.sub_pipeline(name)
            depedency_ids = [stage_to_result[d] for d in pipeline.dependencies()]

            stage = ray.remote(Stage).remote(name,
                                             full_pipeline_id,
                                             self.reduce[name],
                                             self.resources[name],
                                             dependencies=depedency_ids,
                                             resume=resume)
            result = stage.run.remote()
            stage_to_result[name] = result

        # Wait until the extperiment is done
        ray.wait(stage_to_result.values(), num_returns=1)
        logger.info('Experiment ended.')
