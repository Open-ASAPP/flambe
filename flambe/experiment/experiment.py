# from __future__ import annotations
import os
import re
import logging
from copy import deepcopy
from typing import Dict, Optional, Union, Sequence, cast, Callable
from collections import OrderedDict
import shutil
import tempfile

from tqdm import tqdm
from io import StringIO
import ray
from ray.tune.suggest import SearchAlgorithm
from ray.tune.schedulers import TrialScheduler
from ray.tune.logger import DEFAULT_LOGGERS, TFLogger, tf2_compat_logger

from flambe.compile import Schema, Component, yaml
from flambe.runnable import ClusterRunnable
from flambe.compile.downloader import download_manager
from flambe.cluster import errors as man_errors
from flambe.cluster import const
from flambe.cluster import Cluster
from flambe.experiment import utils, wording
from flambe.experiment.options import ClusterResource
from flambe.runnable import RemoteEnvironment
from flambe.runnable import error
from flambe.runnable import utils as run_utils
from flambe.runner.utils import get_files
from flambe.experiment.progress import ProgressState
from flambe.experiment.tune_adapter import TuneAdapter
from flambe.logging import coloredlogs as cl

logger = logging.getLogger(__name__)

OptionalSearchAlgorithms = Optional[Dict[str, Union[SearchAlgorithm, Schema]]]
OptionalTrialSchedulers = Optional[Dict[str, Union[TrialScheduler, Schema]]]


class Experiment(ClusterRunnable):
    """A Experiment object.

    The Experiment object is the top level module in the Flambé
    workflow. The object is responsible for starting workers,
    assiging the orchestrator machine, as well as converting the
    input blocks into Ray Tune Experiment objects.

    Parameters
    ----------
    name: str
        A name for the experiment
    pipeline: OrderedDict[str, Schema[Component]]
        Ordered mapping from block id to a schema of the block
    force: bool
        When running a local experiment this flag will make flambe
        override existing results from previous experiments. When
        running remote experiments this flag will reuse an existing
        cluster (in case of any) that is running an experiment
        with the same name in the same cloud service.
        The use of this flag is discouraged as you may lose useful data.
    resume: Union[str, List[str]]
        If a string is given, resume all blocks up until the given
        block_id. If a list is given, resume all blocks in that list.
    save_path: Optional[str]
        A directory where to save the experiment.
    devices: Dict[str, int]
        Tune's resources per trial. For example: {"cpu": 12, "gpu": 2}.
    resources: Optional[Dict[str, Dict[str, Any]]]
        Variables to use in the pipeline section with !@ notation.
        This section is splitted into 2 sections: local and remote.
    search : Mapping[str, SearchAlgorithm], optional
        Map from block id to hyperparameter search space generator. May
        have Schemas of SearchAlgorithm as well.
    schedulers : Mapping[str, TrialScheduler], optional
        Map from block id to search scheduler. May have Schemas of
        TrialScheduler as well.
    reduce: Mapping[str, int], optional
        Map from block to number of trials to reduce to.
    env: RemoteEnvironment
        Contains remote information about the cluster. This object will
        be received in case this Experiment is running remotely.
    max_failures: int
        Number of times to retry running the pipeline if it hits some
        type of failure, defaults to one.
    merge_plot: bool
        Display all tensorboard logs in the same plot (per block type).
        Defaults to True.
    user_provider: Callable[[], str]
        The logic for specifying the user triggering this
        Runnable. If not passed, by default it will pick the computer's
        user.

    """

    def __init__(self,
                 name: str,
                 pipeline: Dict[str, Schema],
                 resume: Optional[Union[str, Sequence[str]]] = None,
                 devices: Dict[str, int] = None,
                 save_path: Optional[str] = None,
                 resources: Optional[Dict[str, Union[str, ClusterResource]]] = None,
                 search: OptionalSearchAlgorithms = None,
                 schedulers: OptionalTrialSchedulers = None,
                 reduce: Optional[Dict[str, int]] = None,
                 env: RemoteEnvironment = None,
                 max_failures: int = 1,
                 stop_on_failure: bool = True,
                 merge_plot: bool = True,
                 user_provider: Callable[[], str] = None) -> None:
        super().__init__(env=env, user_provider=user_provider)
        self.name = name

        self.original_save_path = save_path

        if save_path is None or len(save_path) == 0:
            save_path = os.path.join(os.getcwd(), "flambe-output")
        else:
            save_path = os.path.abspath(os.path.expanduser(save_path))

        # Prepending 'output' to the name in the output folder
        # is a basic security mechanism to avoid removing user
        # folders when using --force (if for example, save_path
        # is '/$HOME/Desktop' and name is "nlp", and user has folder
        # '$HOME/Desktop/nlp', then there is a risk of accidentally
        # removing it when using --force)
        self.output_folder_name = f"output__{name}"

        self.full_save_path = os.path.join(
            save_path,
            self.output_folder_name
        )

        self.resume = resume
        self.devices = devices
        self.resources = resources or dict()
        self.pipeline = pipeline
        # Compile search algorithms if needed
        self.search = search or dict()
        for stage_name, search_alg in self.search.items():
            if isinstance(search_alg, Schema):
                self.search[stage_name] = search_alg()
        # Compile schedulers if needed
        self.schedulers = schedulers or dict()
        for stage_name, scheduler in self.schedulers.items():
            if isinstance(scheduler, Schema):
                self.schedulers[stage_name] = scheduler()
        self.reduce = reduce or dict()
        self.max_failures = max_failures
        self.stop_on_failure = stop_on_failure
        self.merge_plot = merge_plot
        if pipeline is None or not isinstance(pipeline, (Dict, OrderedDict)):
            raise TypeError("Pipeline argument is not of type Dict[str, Schema]. "
                            f"Got {type(pipeline).__name__} instead")
        self.pipeline = pipeline

    def process_resources(
        self,
        resources: Dict[str, Union[str, ClusterResource]],
        folder: str
    ) -> Dict[str, Union[str, ClusterResource]]:
        """Download resources that are not tagged with '!cluster'
        into a given directory.

        Parameters
        ----------
        resources: Dict[str, Union[str, ClusterResource]]
            The resources dict
        folder: str
            The directory where the remote resources
            will be downloaded.

        Returns
        -------
        Dict[str, Union[str, ClusterResource]]
            The resources dict where the remote urls that
            don't contain '!cluster' point now to the local
            path where the resource was downloaded.

        """
        # Keep the resources temporary dict for later cleanup
        ret = {}
        for k, v in resources.items():
            if not isinstance(v, ClusterResource):
                with download_manager(v, os.path.join(folder, k)) as path:
                    ret[k] = path
            else:
                ret[k] = v

        return ret

    def run(self, force: bool = False, verbose: bool = False, debug: bool = False, **kwargs):
        """Run an Experiment"""

        logger.info(cl.BL("Launching local experiment"))

        # Check if save_path/name already exists + is not empty
        # + force and resume are False
        if not self.resume and not force and os.path.exists(self.full_save_path) \
                and list(get_files(self.full_save_path)):
            raise error.ParsingRunnableError(
                f"Results from an experiment with the same name were located in the save path " +
                f"{self.full_save_path}. To overide this results, please use '--force' " +
                "To use these results and resume the experiment, pick 'resume: True' " +
                "If not, just pick another save_path/name."
            )

        full_save_path = self.full_save_path

        if not self.env:
            wording.print_useful_local_info(full_save_path)

        # If running remotely then all folders were already created.
        # in the 'setup' method.
        if not self.env:
            if os.path.exists(full_save_path) and force:
                shutil.rmtree(full_save_path)  # This deleted the folder also
                logger.info(
                    cl.RE(f"Removed previous existing from {full_save_path} " +
                          "results as --force was specified"))

            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path)
                logger.debug(f"{full_save_path} created to store output")

        self._dump_experiment_file()

        if any(map(lambda x: isinstance(x, ClusterResource), self.resources.values())):
            raise ValueError(
                f"Local experiments doesn't support resources with '!cluster' tags. " +
                "The '!cluster' tag is used for those resources that need to be handled " +
                "in the cluster when running remote experiments.")

        if not self.env:
            self.tmp_resources_dir = tempfile.TemporaryDirectory()
            resources_folder = self.tmp_resources_dir.name
        else:
            resources_folder = f"{self.full_save_path}/_resources"

        resources = self.process_resources(self.resources, resources_folder)

        # rsync downloaded resources
        if self.env:
            run_utils.rsync_hosts(self.env.orchestrator_ip,
                                  self.env.factories_ips,
                                  self.env.user,
                                  self.full_save_path,
                                  self.env.key,
                                  exclude=["state.pkl"])

        # Check that links are in order (i.e topologically in pipeline)
        utils.check_links(self.pipeline, resources)

        # Check that only computable blocks are given
        # search algorithms and schedulers
        utils.check_search(self.pipeline, self.search, self.schedulers)

        # Initialize ray cluster
        kwargs = {"logging_level": logging.ERROR, "include_webui": False}
        if debug:
            kwargs['local_mode'] = True

        if self.env:
            ray.init(redis_address=f"{self.env.orchestrator_ip}:{const.RAY_REDIS_PORT}", **kwargs)
        else:
            ray.init(**kwargs)
            logger.debug(f"Ray cluster up")

        # Initialize map from block to list of checkpoints
        # This is used whe resolving links over other computable blocks
        # TODO: in python 3.7 we can replace these with dict() or {}
        checkpoints: OrderedDict = OrderedDict()
        schemas: OrderedDict = OrderedDict()
        success: OrderedDict = OrderedDict()

        # By default use all CPUs if no GPU is present
        devices = self.devices if self.devices else None
        if devices is None and utils.local_has_gpu():
            devices = {"cpu": 4, "gpu": 1}

        to_resume = None
        if isinstance(self.resume, str):
            index = list(self.pipeline.keys()).index(self.resume)
            to_resume = list(self.pipeline.keys())[:index + 1]
        elif isinstance(self.resume, Sequence):
            to_resume = list(self.resume)

        # Make experiment_tag easier to extract
        def trial_name_creator(trial):
            identifier = ""
            if "env" in trial.config:
                env = trial.config["env"]
                if isinstance(env, type):
                    env = env.__name__
                identifier += f"{env}"
            if trial.experiment_tag:
                hyper_params = {}
                if "_" in trial.experiment_tag:
                    num, tunable_params = trial.experiment_tag.split("_", 1)
                    identifier += tunable_params
                    param_list = [p.split("=") for p in tunable_params.split(",")]
                    hyper_params = {p[0]: p[1] for p in param_list}
                else:
                    identifier += trial.experiment_tag
                trial.config['hyper_params'] = hyper_params
            return identifier.replace("/", "_")

        trial_name_creator = ray.tune.function(trial_name_creator)

        # Compute depedencies DAG
        dependency_dag = {}
        schemas_dag: OrderedDict = OrderedDict()
        for block_id, schema_block in self.pipeline.items():
            schemas_dag[block_id] = schema_block
            relevant_ids = utils.extract_needed_blocks(schemas_dag, block_id, resources)
            dependencies = deepcopy(relevant_ids)
            dependencies.discard(block_id)

            dependency_dag[block_id] = list(dependencies)

        if self.env:
            self.progress_state = ProgressState(
                self.name, full_save_path, dependency_dag,
                self.content, len(self.env.factories_ips))
        else:
            self.progress_state = ProgressState(self.name, full_save_path,
                                                dependency_dag, self.content)

        for block_id, schema_block in tqdm(self.pipeline.items()):
            schema_block.add_extensions_metadata(self.extensions)
            logger.debug(f"Starting {block_id}")

            # Add the block to the configuration so far
            schemas[block_id] = schema_block
            success[block_id] = True

            self.progress_state.checkpoint_start(block_id)
            relevant_ids = utils.extract_needed_blocks(schemas, block_id, resources)
            relevant_schemas = {k: v for k, v in deepcopy(schemas).items() if k in relevant_ids}

            # Set resume
            resume = False if to_resume is None else (block_id in to_resume)

            # If computable, convert to tune.Trainable
            # Each Component block is an Experiment in ray.tune
            if not isinstance(schema_block, Schema):
                raise ValueError('schema block not of correct type Schema')
            if issubclass(schema_block.component_subclass, Component):

                # Returns is a list non-nested configuration
                divided_schemas = list(utils.divide_nested_grid_search_options(relevant_schemas))
                divided_dict = [utils.extract_dict(x) for x in divided_schemas]
                # Convert options and links
                divided_dict_tune = [utils.convert_tune(x) for x in divided_dict]
                # Execute block
                tune_experiments = []
                for param_dict, schemas_dict in zip(divided_dict_tune, divided_schemas):
                    config = {'name': block_id,
                              'merge_plot': self.merge_plot,
                              'params': param_dict,
                              'schemas': Schema.serialize(schemas_dict),
                              'checkpoints': checkpoints,
                              'to_run': block_id,
                              'global_vars': resources,
                              'verbose': verbose,
                              'custom_modules': list(self.extensions.keys()),
                              'debug': debug}
                    # Filter out the tensorboard logger as we handle
                    # general and tensorboard-specific logging ourselves
                    tune_loggers = list(filter(lambda l: l != tf2_compat_logger and  # noqa: E741
                                               not issubclass(l, TFLogger), DEFAULT_LOGGERS))
                    tune_experiment = ray.tune.Experiment(name=block_id,
                                                          run=TuneAdapter,
                                                          trial_name_creator=trial_name_creator,
                                                          config=deepcopy(config),
                                                          local_dir=full_save_path,
                                                          checkpoint_freq=1,
                                                          checkpoint_at_end=True,
                                                          max_failures=self.max_failures,
                                                          resources_per_trial=devices,
                                                          loggers=tune_loggers)
                    logger.debug(f"Created tune.Experiment for {param_dict}")
                    tune_experiments.append(tune_experiment)

                trials = ray.tune.run_experiments(tune_experiments,
                                                  search_alg=self.search.get(block_id, None),
                                                  scheduler=self.schedulers.get(block_id, None),
                                                  queue_trials=True,
                                                  verbose=False,
                                                  resume=resume,
                                                  raise_on_failed_trial=False)
                logger.debug(f"Finish running all tune.Experiments for {block_id}")

                any_error = False
                for t in trials:
                    if t.status == t.ERROR:
                        logger.error(cl.RE(f"Variant {t} of '{block_id}' ended with ERROR status."))
                        success[block_id] = False
                        any_error = True
                if any_error and self.stop_on_failure:
                    self.teardown()
                    self.progress_state.checkpoint_end(block_id, success[block_id])
                    raise error.UnsuccessfulRunnableError(
                        f"Stopping experiment at block '{block_id}' "
                        "because there was an error and stop_on_failure == True."
                    )

                # Save checkpoint location
                # It should point from:
                # block_id -> hash(variant) -> checkpoint
                hashes = []
                for t in trials:
                    schema_with_params: Dict = OrderedDict()
                    for b in schemas_dict:
                        schema_copy = deepcopy(schemas_dict[b])
                        utils.update_schema_with_params(schema_copy, t.config['params'][b])
                        schema_with_params[b] = schema_copy
                    hashes.append(repr(schema_with_params))

                paths = [t._checkpoint.value for t in trials]

                # Mask out error trials
                mask = [True] * len(trials)
                for i, trial in enumerate(trials):
                    if trial.status == ray.tune.trial.Trial.ERROR:
                        mask[i] = False

                # Mask out on reduce
                reduce_k = self.reduce.get(block_id, None)
                if reduce_k is not None and int(reduce_k) > 0:
                    # Get best
                    best_trials = utils.get_best_trials(trials, topk=int(reduce_k))
                    best_trial_ids = set([t.trial_id for t in best_trials])
                    # Mask out
                    for i, trial in enumerate(trials):
                        if trial.trial_id not in best_trial_ids:
                            mask[i] = False

                trial_checkpoints = {t_hash: path for t_hash, path in zip(hashes, paths)}
                trial_mask = {t_hash: mask_value for t_hash, mask_value in zip(hashes, mask)}
                checkpoints[block_id] = {'paths': trial_checkpoints, 'mask': trial_mask}

                # Rsync workers to main machine and back to all workers
                # TODO specify callbacks. If not remote will not work
                if self.env:
                    run_utils.rsync_hosts(self.env.orchestrator_ip,
                                          self.env.factories_ips,
                                          self.env.user,
                                          self.full_save_path,
                                          self.env.key,
                                          exclude=["state.pkl"])

            self.progress_state.checkpoint_end(block_id, success[block_id])
            logger.debug(f"Done running {block_id}")

        self.teardown()

        if all(success.values()):
            logger.info(cl.GR("Experiment ended successfully"))
        else:
            raise error.UnsuccessfulRunnableError(
                "Not all trials were successful. Check the logs for more information"
            )

    def teardown(self):
        # Disconnect process from ray cluster
        ray.shutdown()

        # Shutdown ray cluster.
        if self.env:
            ret = utils.shutdown_ray_node()
            logger.debug(f"Node shutdown {'successful' if ret == 0 else 'failed'} in Orchestrator")
            for f in self.env.factories_ips:
                ret = utils.shutdown_remote_ray_node(f, "ubuntu", self.env.key)
                logger.debug(f"Node shutdown {'successful' if ret == 0 else 'failed'} in {f}")

        self.progress_state.finish()

        if hasattr(self, 'tmp_resources_dir'):
            self.tmp_resources_dir.cleanup()

    def setup(self, cluster: Cluster, extensions: Dict[str, str], force: bool, **kwargs) -> None:
        """Prepare the cluster for the Experiment remote execution.

        This involves:

        1) [Optional] Kill previous flambe execution
        2) [Optional] Remove existing results
        3) Create supporting dirs (exp/synced_results, exp/resources)
        4) Install extensions in all factories
        5) Launch ray cluster
        6) Send resources
        7) Launch Tensorboard + Report site

        Parameters
        ----------
        cluster: Cluster
            The cluster where this Runnable will be running
        extensions: Dict[str, str]
            The ClusterRunnable extensions
        force: bool
            The force value provided to Flambe

        """

        if cluster.existing_flambe_execution() or cluster.existing_ray_cluster():
            if not force:
                raise man_errors.ClusterError("This cluster is currently used by other " +
                                              "experiment. Use --force flag to reuse it. Aborting.")
            else:
                cluster.shutdown_flambe_execution()
                cluster.shutdown_ray_cluster()
                logger.info(cl.YE("Forced resource to become available..."))

        output_dir_remote = f"{self.name}/{self.output_folder_name}"
        if cluster.existing_dir(output_dir_remote):
            logger.debug("This cluster already ran an experiment " +
                         "with the same name.")

            if self.resume:
                logger.info(cl.YE("Resuming previous experiment..."))
            elif force:
                cluster.remove_dir(output_dir_remote, content_only=True, all_hosts=True)
            else:
                raise man_errors.ClusterError(
                    "This cluster already has results for the same experiment name. " +
                    "If you wish to reuse them, use resume: True or if you want to override them " +
                    "use --force. Aborting."
                )

        cluster.install_extensions_in_factories(extensions)
        logger.info(cl.YE("Extensions installed in all factories"))

        # Add redundant check for typing
        if not cluster.orchestrator:
            raise man_errors.ClusterError("The orchestrator needs to exist at this point")

        cluster.create_dirs([self.name,
                             f"{self.name}/{self.output_folder_name}",
                             f"{self.name}/{self.output_folder_name}/_resources"])
        logger.info(cl.YE("Created supporting directories"))

        cluster.launch_ray_cluster()

        if not cluster.check_ray_cluster():
            raise man_errors.ClusterError("Ray cluster not launched correctly.")

        local_resources = {k: v for k, v in self.resources.items()
                           if not isinstance(v, ClusterResource)}

        tmp_resources_dir = tempfile.TemporaryDirectory()

        # This will download remote resources.
        local_resources = self.process_resources(
            local_resources, tmp_resources_dir.name)  # type: ignore

        local_resources = cast(Dict[str, str], local_resources)

        if local_resources:
            new_resources = cluster.send_local_content(
                local_resources,
                os.path.join(cluster.orchestrator.get_home_path(), self.name,
                             self.output_folder_name, "_resources"),
                all_hosts=True
            )
        else:
            new_resources = dict()

        tmp_resources_dir.cleanup()

        # Add the cluster resources without the tag
        new_resources.update({k: v.location for k, v in self.resources.items()
                              if isinstance(v, ClusterResource)})

        if cluster.orchestrator.is_tensorboard_running():
            if force:
                cluster.orchestrator.remove_tensorboard()
            else:
                raise man_errors.ClusterError("Tensorboard was running on the orchestrator.")

        cluster.orchestrator.launch_tensorboard(output_dir_remote, const.TENSORBOARD_PORT)

        if cluster.orchestrator.is_report_site_running():
            if force:
                cluster.orchestrator.remove_report_site()
            else:
                raise man_errors.ClusterError("Report site was running on the orchestrator")

        cluster.orchestrator.launch_report_site(
            f"{output_dir_remote}/state.pkl",
            port=const.REPORT_SITE_PORT,
            output_log=f"output.log",
            output_dir=output_dir_remote,
            tensorboard_port=const.TENSORBOARD_PORT
        )

        self.set_serializable_attr("resources", new_resources)
        self.set_serializable_attr("devices", cluster.get_max_resources())
        self.set_serializable_attr(
            "save_path", f"{cluster.orchestrator.get_home_path()}/{self.name}")

    def parse(self) -> None:
        """Parse the experiment.

        Parse the Experiment in search of errors that won't allow the
        experiment to run successfully. If it finds any error, then it
        raises an ParsingExperimentError.

        Raises
        ------
        ParsingExperimentError
            In case a parsing error is found.

        """
        # Check if name is None:
        if self.name is None or len(self.name) == 0:
            raise error.ParsingRunnableError(
                "Experiment should declare a name and it must not be empty"
            )

        # Check if name is valid
        else:
            if re.match('^([a-zA-Z0-9]+[_-]*)+$', self.name) is None:
                raise error.ParsingRunnableError(
                    "Experiment name should contain only alphanumeric characters " +
                    "(with optional - or _ in between)"
                )

    def get_user(self) -> str:
        """Get the user that triggered this experiment.

        Returns
        -------
        str:
            The user as a string.

        """
        return self.env.local_user if self.env else self.user_provider()

    def _dump_experiment_file(self) -> None:
        """Dump the experiment YAML representation
        to the output folder.

        """
        destination = os.path.join(self.full_save_path, "experiment.yaml")
        with open(destination, 'w') as f:
            with StringIO() as s:
                yaml.dump_all([self.extensions, self], s)
                f.write(s.getvalue())
            f.flush()
        logger.debug(f"Saved experiment file in {destination}")
