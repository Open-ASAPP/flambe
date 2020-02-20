import logging
import click
import os
import sys
import shutil
import traceback

import torch

import flambe
from flambe.const import FLAMBE_GLOBAL_FOLDER, ASCII_LOGO, ASCII_LOGO_DEV
from flambe.const import FLAMBE_CLUSTER_DEFAULT_FOLDER, FLAMBE_CLUSTER_DEFAULT_CONFIG
from flambe.logging import coloredlogs as cl
from flambe.utils.path import is_dev_mode, get_flambe_repo_location
from flambe.compile.downloader import download_manager
from flambe.compile.extensions import is_package, is_installed_module
from flambe.runner.environment import load_env_from_config
from flambe.runner.protocol import load_runnable_from_config
from flambe.cluster.cluster import load_cluster_config


logging.getLogger('tensorflow').disabled = True


@click.group()
def cli():
    pass


# ----------------- flambe up ------------------ #
@click.command()
@click.argument('name', type=str, required=True)
@click.option('--create', is_flag=True, default=False,
              help='Create a new cluster.')
@click.option('--config', type=str, default=FLAMBE_CLUSTER_DEFAULT_CONFIG,
              help="Cluster template config.")
@click.option('--min-workers', type=int, default=None,
              help="Required name for a new cluster.")
@click.option('--max-workers', type=int, default=None,
              help="Optional max number of workers.")
def up(name, create, config, min_workers, max_workers):
    """Launch / update the cluster."""
    os.makedirs(FLAMBE_CLUSTER_DEFAULT_FOLDER)
    cluster_path = os.path.join(FLAMBE_CLUSTER_DEFAULT_FOLDER, f"{name}.yaml")

    # Check update or install
    if create and os.path.exists(cluster_path):
        raise ValueError(f"Cluster {name} already exists.")
    elif not create and not os.path.exists(cluster_path):
        raise ValueError(f"Cluster {name} does not exist.")
    elif create and not os.path.exists(config):
        raise ValueError(f"Config {config} does not exist.")
    elif create:
        yaml = YAML()
        # Load cluster template config
        with open(config, 'r') as f:
            cluster = load_cluster_config(cluster_path)
        # 
        with open(cluster_path, 'w') as f:
            yaml = YAML()
            yaml.dump(f)
    else:
        cluster = load_cluster_config(cluster_path)

    # Run update
    cluster.up(min_workers, max_workers)


# ----------------- flambe down ------------------ #
@click.command()
@click.argument('name', type=str, required=True)
@click.option('-y', '--yes', is_flag=True, default=False,
              help='Run without confirmation.')
@click.option('--workers-only', is_flag=True, default=False,
              help='Only teardown the worker nodes.')
@click.option('--terminate', is_flag=True, default=False,
              help='Terminate the instances instead of stopping them (AWS only).')
@click.option('--destroy', is_flag=True, default=False,
              help='Destroys this cluster permanently.')
def down(name, yes, workers_only, terminate, destroy):
    """Take down the cluster, optionally destroy it permanently."""
    cluster = load_cluster_config(cluster)
    cluster.down(yes, workers_only, terminate)


# ----------------- flambe rsync up ------------------ #
@click.command()
@click.argument('source', type=str, required=True)
@click.argument('target', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def rsync_up(source, target, cluster):
    """Upload files to the cluster."""
    cluster = load_cluster_config(cluster)
    cluster.rsync_up(source, target)


# ----------------- flambe rsync down ------------------ #
@click.command()
@click.argument('source', type=str, required=True)
@click.argument('target', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def rsync_down(source, target, cluster):
    """Download files from the cluster."""
    cluster = load_cluster_config(cluster)
    cluster.rsync_down(source, target)


# ----------------- flambe list ------------------ #
@click.command()
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-a', '--all', is_flag=True, default=False,
              help="Cluster name.")
def list_cmd(cluster, all):
    """List the jobs (i.e tmux sessions) running on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config(cluster)
    cluster.list()


# ----------------- flambe exec ------------------ #
@click.command()
@click.argument('command', type=str)
@click.option('-p', '--port-forward', type=int, default=None,
              help='Port in which the site will be running url')
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def exec_cmd(command, port_forward, cluster):
    """Execute a command on the cluster head node."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config(cluster)
    cluster.exec(command=command, port_forward=port_forward)


# ----------------- flambe attach ------------------ #
@click.command()
@click.argument('name', required=False, type=str, default=None)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def attach(name, cluster):
    """Attach to a running job (i.e tmux session) on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config(cluster)
    cluster.attach(name)


# ----------------- flambe kill ------------------ #
@click.command()
@click.argument('name', type=str)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('--clean', is_flag=True, default=False,
              help='Clean the artifacts of the job.')
def kill(name, cluster, clean):
    """Kill a job (i.e tmux session) running on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config(cluster)
    cluster.kill(name=name)
    if clean:
        cluster.clean(name=name)


# ----------------- flambe clean ------------------ #
@click.command()
@click.argument('name', type=str)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def clean(name, cluster):
    """Clean the artifacts of a job on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config(cluster)
    cluster.clean(name=name)


# ----------------- flambe submit ------------------ #
@click.command()
@click.argument('config', type=str, required=True)
@click.argument('name', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True, default=False,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Experiment, Ray will run in a single thread \
                    allowing user breakpoints')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Verbose console output')
@click.option('-a', '--attach', is_flag=True, default=False,
              help='Attach after submitting the job.')
def submit(runnable, name, cluster, force, debug, verbose, attach):
    """Submit a job to the cluster, as a YAML config."""
    if debug:
        logging.disable(logging.INFO)
    else:
        logging.disable(logging.ERROR)
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    cluster = load_cluster_config(cluster)
    cluster.submit(runnable, name, force=force, debug=debug)
    if attach:
        cluster.attach(name)


# ----------------- flambe site ------------------ #
@click.command()
@click.argument('name', type=str, required=False, default='')
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-p', '--port', type=int, default=49558,
              help='Port in which the site will be running url')
def site(name, cluster, port):
    """Launch a Web UI to monitor the activity on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config(cluster)
    try:
        cluster.launch_site(port=port, name=name)
    except KeyboardInterrupt:
        logging.disable(logging.ERROR)


# ----------------- flambe run ------------------ #
@click.command()
@click.argument('config', type=str, required=True)
@click.option('-o', '--output', default='./',
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Experiment, Ray will run in a single thread \
                    allowing user breakpoints')
def run(runnable, output, force, debug):
    """Execute a runnable config."""
    # Load environment
    env = load_env_from_config(runnable)
    if not env:
        env = flambe.get_env()

    # Check if previous job exists
    output = os.path.join(os.path.expanduser(output), 'flambe_output')
    if os.path.exists(output):
        if force:
            shutil.rmtree(output)
        else:
            raise ValueError(f"{output} already exists. Use -f, --force to override.")

    os.makedirs(output)

    # torch.multiprocessing exists, ignore mypy
    # TODO: investigate if this is actually needed
    torch.multiprocessing.set_start_method('fork', force=True)  # type: ignore

    # Check if dev mode
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    # Check if debug
    if debug:
        print(cl.YE(f"Debug mode activated\n"))

    # Check that all extensions are importable
    message = "Module ({}) from package ({}) is not installed."
    for module, package in env.extensions.items():
        package = os.path.expanduser(package)
        if not is_installed_module(module):
            # Check if the package exsists locally
            if os.path.exists(package):
                if is_package(package):
                    # Package exsists locally but is not installed
                    print(message.format(module, package) + " Attempting to add to path.")
                # Try to add to the python path
                sys.path.append(package)
            else:
                raise ValueError(message.format(module, package))

    # Download resources
    resources_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'resources')
    updated_resources: Dict[str, str] = dict()
    for name, resource in env.local_files.items():
        with download_manager(resource, os.path.join(resources_dir, name)) as path:
            updated_resources[name] = path

    try:
        # Execute runnable
        flambe.set_env(
            output_path=output,
            debug=debug,
            local_files=updated_resources
        )

        runnable_obj = load_runnable_from_config(runnable)
        runnable_obj.run()
        print(cl.GR("------------------- Done -------------------"))
    except KeyboardInterrupt:
        print(cl.RE("---- Exiting early (Keyboard Interrupt) ----"))
    except Exception:
        print(traceback.format_exc())
        print(cl.RE("------------------- Error -------------------"))


if __name__ == '__main__':
    cli.add_command(up)
    cli.add_command(down)
    cli.add_command(kill)
    cli.add_command(clean)
    cli.add_command(list_cmd, name='ls')
    cli.add_command(exec_cmd, name='exec')
    cli.add_command(attach)
    cli.add_command(run)
    cli.add_command(submit)
    cli.add_command(site)
    cli.add_command(rsync_up, name='rsync-up')
    cli.add_command(rsync_down, name='rsync-down')
    cli(prog_name='flambe')