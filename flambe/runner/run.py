"""Local run script for flambe.
"""

import argparse
import logging

import torch
import ray
import os

from flambe.const import FLAMBE_GLOBAL_FOLDER
from flambe.compile import make_component
from flambe.compile.extensions import download_extensions
from flambe.optim import LRScheduler
from flambe.runnable.utils import is_dev_mode, get_flambe_repo_location
from flambe.logging import setup_global_logging
from flambe.logging import coloredlogs as cl
from flambe.runnable.context import SafeExecutionContext
from flambe.runnable import ClusterRunnable
from flambe.cluster import Cluster
from flambe.runner.utils import check_system_reqs
from flambe.logo import ASCII_LOGO, ASCII_LOGO_DEV
import flambe

from typing import cast


TORCH_TAG_PREFIX = "torch"
TUNE_TAG_PREFIX = "tune"


def main(args: argparse.Namespace) -> None:
    """Execute command based on given config"""
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    if args.debug:
        print(cl.YE(f"Debug mode activated\n"))
        if args.cluster is not None:
            raise ValueError('Will not run on cluster in debug mode. ' +
                             'Please disable debug mode or run locally.')

    # Pass original module for ray / pickle
    exclude = ['torch.nn.quantized', 'torch.nn.qat']
    make_component(torch.nn.Module, only_module='torch.nn', exclude=exclude)
    # torch.optim.Optimizer exists, ignore mypy
    make_component(torch.optim.Optimizer,  # type: ignore
                   only_module='torch.optim')
    make_component(torch.optim.lr_scheduler._LRScheduler,
                   only_module='torch.optim.lr_scheduler', parent_component_class=LRScheduler)
    make_component(ray.tune.schedulers.TrialScheduler)
    make_component(ray.tune.suggest.SearchAlgorithm)

    # TODO check first if there is cluster as if there is there
    # is no need to install extensions
    check_system_reqs()
    with SafeExecutionContext(args.config) as ex:
        if args.cluster is not None:
            with SafeExecutionContext(args.cluster) as ex_cluster:
                cluster, _ = ex_cluster.preprocess(secrets=args.secrets,
                                                   install_ext=args.install_extensions)
                runnable, extensions = ex.preprocess(import_ext=False,
                                                     check_tags=False,
                                                     secrets=args.secrets)
                cluster.run(force=args.force, debug=args.debug)
                if isinstance(runnable, ClusterRunnable):
                    cluster = cast(Cluster, cluster)

                    # This is independant to the type of ClusterRunnable
                    destiny = os.path.join(cluster.get_orch_home_path(), "extensions")

                    # Before sending the extensions, they need to be
                    # downloaded (locally).
                    t = os.path.join(FLAMBE_GLOBAL_FOLDER, "extensions")
                    extensions = download_extensions(extensions, t)

                    # At this point, all remote extensions
                    # (except pypi extensions)
                    # have local paths.
                    new_extensions = cluster.send_local_content(extensions,
                                                                destiny, all_hosts=True)

                    new_secrets = cluster.send_secrets()

                    # Installing the extensions is crutial as flambe
                    # will execute without '-i' flag and therefore
                    # will assume that the extensions are installed
                    # in the orchestrator.
                    cluster.install_extensions_in_orchestrator(new_extensions)
                    logger.info(cl.GR("Extensions installed in Orchestrator"))

                    runnable.setup_inject_env(cluster=cluster,
                                              extensions=new_extensions,
                                              force=args.force)
                    cluster.execute(runnable, new_extensions, new_secrets, args.force)
                else:
                    raise ValueError("Only ClusterRunnables can be executed in a cluster.")
        else:
            runnable, _ = ex.preprocess(secrets=args.secrets,
                                        install_ext=args.install_extensions)
            runnable.run(force=args.force, verbose=args.verbose, debug=args.debug)


if __name__ == '__main__':
    # torch.multiprocessing exists, ignore mypy
    torch.multiprocessing.set_start_method('fork', force=True)  # type: ignore

    parser = argparse.ArgumentParser(description='Run a flambé!')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('-i', '--install-extensions', action='store_true', default=False,
                        help='Install extensions automatically using pip. WARNING: ' +
                             'This could potentially override already installed packages.')
    parser.add_argument('-c', '--cluster', type=str, default=None,
                        help='Specify the cluster that will run the experiment. This option ' +
                             'works if the main config is an Experiment')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Override existing runnables. Be careful ' +
                             'when using this flag as it could have undesired effects.')
    parser.add_argument('-s', '--secrets',
                        type=str, default=os.path.join(FLAMBE_GLOBAL_FOLDER, "secrets.ini"))
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode. Each runnable specifies the debug behavior. ' +
                             'For example for an Experiment, Ray will run in a single thread ' +
                             'allowing user breakpoints')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose console output')
    args = parser.parse_args()

    setup_global_logging(logging.INFO if not args.verbose else logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        main(args)
        logger.info(cl.GR("------------------- Done -------------------"))
    except KeyboardInterrupt:
        logger.info(cl.RE("---- Exiting early (Keyboard Interrupt) ----"))
