import os
import copy
from typing import Dict, List, Optional
import logging

import ray

from flambe.search import Algorithm, Search, Choice
from flambe.experiment.pipeline import Pipeline
from flambe.runner import Environment


logger = logging.getLogger(__name__)


@ray.remote
def run_search(variant,
               algorithm,
               cpus_per_trial,
               gpus_per_trial,
               environment):
    search = Search(
        variant,
        algorithm,
        cpus_per_trial,
        gpus_per_trial
    )
    return search.run(environment)


class Stage(object):
    """A stage in the Experiment pipeline.

    This object is a wrapper around the Search object, which adds
    logic to support hyperparameter searches as nodes in a directed
    acyclic graph. In particular, it handles applying dependency
    resolution, running searches, and reducing to the best trials.

    """

    def __init__(self,
                 name: str,
                 pipeline: Pipeline,
                 dependencies: List[Dict[str, Pipeline]],
                 cpus_per_trial: int,
                 gpus_per_trial: int,
                 environment: Environment,
                 algorithm: Optional[Algorithm] = None,
                 reductions: Optional[Dict[str, int]] = None):
        """Initialize a Stage.

        Parameters
        ----------
        name : str
            A name for this stage in the experiment pipeline.
        pipeline : Pipeline
            The sub-pipeline to execute in this stage.
        algorithm : Algorithm
            A search algorithm.
        dependencies : List[Pipeline]
            A list of previously executed pipelines.
        reductions : Dict[str, int]
            Reductions to apply between stages.
        cpus_per_trial : int
            The number of CPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 1.
        gpus_per_trial : int
            The number of GPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 0.

        """
        self.name = name
        self.pipeline = pipeline
        self.algorithm = algorithm
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.reductions = reductions if reductions is not None else dict()
        self.env = environment

        # Flatten out dependencies
        self.dependencies = {name: p for dep in dependencies for name, p in dep.items()}

    def filter_dependencies(self, pipelines: Dict[str, 'Pipeline']) -> Dict[str, 'Pipeline']:
        """Filter out erros, and apply reductions on dependencies.

        Parameters
        ----------
        pipelines: Dict[str, Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        Dict[str, Pipeline]
            An updated list of pipelines with error filtering and
            reductions applied.

        """
        # Filter out error trials
        pipelines = {k: p for k, p in pipelines.items() if not p.error}

        for stage, reduction in self.reductions.items():
            # Find all pipelines with the given stage name
            reduce = {k: p for k, p in pipelines.items() if p.task == stage}
            ignore = {k: p for k, p in pipelines.items() if p.task != stage}

            # Apply reductions
            keys = sorted(reduce.keys(), key=lambda k: reduce[k].metric, reverse=True)
            pipelines = {k: reduce[k] for k in keys[:reduction]}
            pipelines.update(ignore)

        return pipelines

    def merge_variants(self, pipelines: Dict[str, 'Pipeline']) -> Dict[str, 'Pipeline']:
        """Filter out erros, and apply reductions on dependencies.

        Parameters
        ----------
        pipelines: Dict[str, Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        Dict[str, Pipeline]
            An updated list of pipelines with error filtering and
            reductions applied.

        """
        variants: Dict[str, Pipeline] = dict()

        for name, pipe in pipelines.items():
            match_found = False
            for var_name, var in variants.items():
                # If all the matching stages have matching schemas
                if pipe.matches(var):
                    match_found = True
                    variants[var_name] = pipe.merge(var)

            # If no match was found, then just add to variants
            if not match_found:
                variants[name] = pipe

        return variants

    def construct_pipeline(self, pipelines: Dict[str, 'Pipeline']) -> Optional['Pipeline']:
        """Filter out erros, and apply reductions on dependencies.

        Parameters
        ----------
        pipelines: List[Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        Dict[str, Pipeline]
            An updated list of pipelines with error filtering and
            reductions applied.

        """
        schemas = copy.deepcopy(self.pipeline)
        task = schemas[self.name]

        # Here we nest the pipelines so that we can search over
        # cross-stage parameter configurations
        if len(pipelines) == 0:
            pipeline = Pipeline({
                self.name: task
            })
        elif len(pipelines) == 1:
            pipeline = Pipeline({
                'dependencies': pipelines[0],  # type: ignore
                self.name: task
            })
        else:
            pipeline = Pipeline({
                'dependencies': Choice(pipelines),  # type: ignore
                self.name: task
            })

        # Check that the pipeline is complete
        if pipeline.is_subpipeline:
            return pipeline
        else:
            return None

    def run(self) -> Dict[str, Pipeline]:
        """Execute the stage.

        Proceeds as follows:

        1. Filter out errored trials and apply reductions
        2. Construct dependency variants
        3. Get the link types between this stage and its dependencies
        4. Construct the pipelines to execute
        5. For every pipeline launch a Search remotely
        6. Aggregate results, and return executed pipelines

        Returns
        -------
        Dict[str, Pipeline]
            A list of pipelines each containing the schema, checkpoint,
            variant id, and error status for the respective trial.

        """
        # Each dependency is the output of stage, which is
        # a pipeline object with all the variants that ran
        filtered = self.filter_dependencies(self.dependencies)

        # Take an intersection with the other sub-pipelines
        merged = self.merge_variants(filtered)

        # Construct pipelines to execute
        pipeline = self.construct_pipeline(merged)
        if pipeline is None:
            logger.warn(f"Stage {self.name} did not have any variants to execute.")
            return dict()

        # Exectue remotely
        pipeline_env = self.env.clone(
            output_path=os.path.join(
                self.env.output_path,
                self.name
            )
        )
        object_id = run_search.remote(
            pipeline,
            self.algorithm,
            self.cpus_per_trial,
            self.gpus_per_trial,
            pipeline_env
        )

        return object_id

        # Get results and construct output pipelines
        results = ray.get(object_ids)
        pipelines: Dict[str, Pipeline] = dict()
        for variants in results:
            # Each variants object is a dictionary from variant name
            # to a dictionary with schema, params, and checkpoint
            for name, var_dict in variants.items():
                # Flatten out the pipeline schema
                pipeline: Pipeline = var_dict['schema']
                # Add search results to the pipeline
                pipeline.var_ids[self.name] = var_dict['var_id']
                pipeline.checkpoints[self.name] = var_dict['checkpoint']
                pipeline.error = var_dict['error']
                pipeline.metric = var_dict['metric']
                pipelines[name] = pipeline

        return pipelines
