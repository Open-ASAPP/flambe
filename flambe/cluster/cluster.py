"""This module contains the base implementation of a Cluster.

A Cluster is in charge of dealing with the different Instance
objects that will be part of the remote runnable.

"""
from flambe.compile import yaml
from flambe.runnable import Runnable
from flambe.runnable.error import MissingSecretsError
from flambe.runnable.utils import is_dev_mode
from flambe.runner.utils import get_size_MB
from flambe.cluster import const
from flambe.cluster.instance import errors
from flambe.cluster import errors as man_errors
from flambe.cluster.instance.instance import OrchestratorInstance, GPUFactoryInstance, \
    CPUFactoryInstance, Instance
from flambe.logging import coloredlogs as cl

from flambe.runnable.environment import RemoteEnvironment
from concurrent.futures import ThreadPoolExecutor

from typing import Optional, Type, List, TypeVar, Union, Dict, Callable
from types import TracebackType

import logging
import os
import traceback
import sys
import configparser

import errno

from ruamel.yaml.compat import StringIO

import tempfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

GPUFactoryInsT = TypeVar("GPUFactoryInsT", bound=GPUFactoryInstance)
CPUFactoryInsT = TypeVar("CPUFactoryInsT", bound=CPUFactoryInstance)
FactoryInsT = Union[GPUFactoryInsT, CPUFactoryInsT]


UPLOAD_WARN_LIMIT_MB = 10


class Cluster(Runnable):
    """Basic implementation of a Cluster.

    The cluster is in charge of creating the cluster of instances where
    one host is the Orchestrator while the other ones are Factories.

    This implementation should not be used by an end user.
    In order to give support to a cloud service provider (ex: AWS),
    a child class must be implemented inheriting from the Cluster class.

    *Important: when possible, Clusters should context managers*

    Parameters
    ----------
    name: str
        The name of the cluster, used to name the remote instances.
    factories_num : int
        The amount of factories to use. Note that this differs from
        the number of workers,  as each factories can contain multiple
        GPUs and therefore, multiple workers.
    key: str
        The path to the ssh key used to communicate to all instances.
        IMPORTANT: all instances must be accessible with the same key.
    username: str
        The username of the instances the cluster will handle.
        IMPORTANT: for now all instances need to have the same username.
    setup_cmds: Optional[List[str]]
        A list of commands to be run on all hosts for setup purposes.
        These commands can be used to mount volumes, install software,
        etc. Defaults to None.
        IMPORTANT: the commands need to be idempotent and they shouldn't
        expect user input.

    """
    def __init__(self,
                 name: str,
                 factories_num: int,
                 key: str,
                 username: str,
                 setup_cmds: Optional[List[str]] = None) -> None:
        super().__init__()
        self.name = name
        self.factories_num = factories_num
        self.setup_cmds = setup_cmds

        self.key = os.path.abspath(os.path.expanduser(key))
        self.username = username

        self.debug = is_dev_mode()

        self.orchestrator: Optional[OrchestratorInstance] = None
        self.factories: List[FactoryInsT] = []

    def __enter__(self) -> "Cluster":
        """A Cluster should be used with a context cluster
        to handle all possible errors in a clear way.

        Examples
        --------

        >>> with cluster as cl:
        >>>     cl.launch_orchestrator()
        >>>     cl.build_cluster()
        >>>     ...

        """
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 tb: Optional[TracebackType]):
        """Exit method for the context cluster.

        This method will catch any exception, log it and return True.
        This means that all exceptions produced in a Cluster
        (used with the context cluster) will not continue to raise.

        """
        if exc_type is not None:
            self.rollback_env()
            traceback.print_exception(exc_type, exc_value, tb)
            if exc_type == man_errors.ClusterError:
                sys.exit(errno.EREMOTE)
            if exc_type == MissingSecretsError:
                sys.exit("Double check the information provided in the secrets file. " +
                         "Consider that it can be provided in the 'secrets' parameter or if " +
                         "not provided it looks in ~/.flambe.ini")
            else:
                sys.exit(-1)

        return False

    def get_orchestrator_name(self) -> str:
        """Get the orchestrator name.

        The name is given by `name` with the '_orchestrator' suffix.
        For example, if name is 'seq2seq-en-fr', then the orchestrator
        name will be 'seq2seq-en-fr_orchestrator'.

        This is an auxiliary method that can be used in child classes.

        Returns
        -------
        str
            The orcehstrator name

        """
        return f"{self.name}_orchestrator"

    def get_factory_basename(self) -> str:
        """Get the factory base name.

        The name is `name` with the '_factory' suffix.
        For example, if name is 'seq2seq-en-fr', then the factory
        basename will be 'seq2seq-en-fr_factory'.

        The base name can be used to generate all the factories' names
        (for example, by also appending an index to the basename).

        This is an auxiliary method that can be used in child classes.

        Returns
        -------
        str
            The factory basename

        """
        return f"{self.name}_factory"

    def load_all_instances(self) -> None:
        """Method to make all hosts accessible.

        Depending on the Cluster type, it behaves differently.
        For example, AWSCluster or GCPCluster can create the instances
        in this step. The SSHCluster does nothing (the machines are
        already created).

        """
        raise NotImplementedError

    def _get_all_hosts(self):
        """Auxiliary method to get all the hosts in a list.append(

        """
        instances = self.factories[:]
        instances.append(self.orchestrator)
        return instances

    def create_dirs(self, relative_dirs: List[str]) -> None:
        """Create folders in all hostss.

        If some of the already exist, it will do nothing.

        Parameters
        ----------
        relative_dirs: List[str]
            The directories to create. They should be relative paths
            and $HOME of each host will be used to add the prefix.

        """
        with ThreadPoolExecutor() as executor:
            futures = {}

            for ins in self._get_all_hosts():
                futures[executor.submit(ins.create_dirs, relative_dirs)] = ins

            for f in futures.keys():
                try:
                    f.result()
                except errors.RemoteCommandError:
                    raise
                except Exception as exc:
                    logger.error(f'Generated an exception: {exc}')
                    raise
                else:
                    logger.debug(f'{futures[f].host} ready')

        logger.info(cl.GR("All instances prepared"))

    def prepare_all_instances(self) -> None:
        """Prepare all the instances (both orchestrator and factories).

        This method assumes that the hosts are running and accesible.
        It will call the 'prepare' method from all hosts.

        """
        with ThreadPoolExecutor() as executor:
            futures = {}

            for ins in self._get_all_hosts():
                futures[executor.submit(ins.prepare)] = ins

            for f in futures.keys():
                try:
                    f.result()
                except errors.RemoteCommandError:
                    raise
                except Exception as exc:
                    logger.error(f'Generated an exception: {exc}')
                    raise
                else:
                    logger.debug(f'{futures[f].host} ready')

        logger.info(cl.GR("All instances prepared"))

    def run(self, force: bool = False, **kwargs) -> None:
        """Run a cluster and load all the instances.

        After this metho runs, the orchestrator and factories
        objects will be populated.

        If a runnable is provided, then the cluster will execute
        the runnable remotely in the cluster. Currently, only
        ClusterRunnable is supported.

        This method should be idempotent (ie if called N times with
        the same configuration, only one cluster will be created.)

        Parameters
        ----------
        force: bool, defaults to False
            If true, current executions of the same runnable in the
            cluster will be overriden by a new execution.

        """
        self.load_all_instances()
        logger.info(cl.GR("Cluster loaded"))

        for ins in self._get_all_hosts():
            ins.wait_until_accessible()

        logger.debug("All instances accessible.")
        self.distribute_keys()

        self.create_dirs(["extensions"])
        logger.debug("Created flambe folder to store content")

        if self.setup_cmds is not None:
            self.run_cmds(self.setup_cmds)

        self.prepare_all_instances()
        logger.info(cl.GR("Flambe installed in all hosts"))

    def run_cmds(self, setup_cmds: List[str]) -> None:
        """Run setup commands in all hosts

        Parameters
        ----------
        setup_cmds: List[str]
            The list of commands

        Raises
        ------
        errors.RemoteCommandError
            If at least one commands is not successful in at
            least one host.

        """
        with ThreadPoolExecutor() as executor:
            futures = []

            for ins in self._get_all_hosts():
                futures.append(executor.submit(ins.run_cmds, setup_cmds))

            for f in futures:
                try:
                    f.result()
                except errors.RemoteCommandError:
                    raise
                except Exception as exc:
                    logger.error('Generated an unknown exception: {}'.format(exc))
                    raise

        logger.info(cl.GR("Custom commands ran successfully in all hosts"))

    def get_orchestrator(self, ip: str, private_ip: str = None,
                         use_public: bool = True) -> OrchestratorInstance:
        """Get an orchestrator instance"""
        return OrchestratorInstance(ip, private_ip if private_ip else ip,
                                    self.username, self.key, self.config, self.debug, use_public)

    def get_orch_home_path(self) -> str:
        """Return the orchestrator home path

        Returns
        -------
        str

        """
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        return self.orchestrator.get_home_path()

    def get_factory(self, ip: str, private_ip: str = None,
                    use_public: bool = True) -> CPUFactoryInstance:
        """Get an CPU factory instance"""
        return CPUFactoryInstance(ip, private_ip if private_ip else ip,
                                  self.username, self.key, self.config,
                                  self.debug, use_public)

    def get_gpu_factory(self, ip: str, private_ip: str = None,
                        use_public: bool = True) -> GPUFactoryInstance:
        """Get an GPU factory instance"""
        return GPUFactoryInstance(ip, private_ip if private_ip else ip,
                                  self.username, self.key, self.config,
                                  self.debug, use_public)

    def launch_ray_cluster(self) -> None:
        """Create a ray cluster.

        The main node is going to be located in the orchestrator machine
        and all other nodes in the factories.

        The main node is executed with --num-cpus=0 flag so that
        it doesn't do any work and all work is done by the factories.

        """
        for ins in self._get_all_hosts():
            if ins.is_node_running():
                raise man_errors.ClusterError(
                    f"Node {ins.host} is running in an existing cluster. Aborting.")

        port = const.RAY_REDIS_PORT

        # The orchestator needs to exist at this point
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        self.orchestrator.launch_node(port)

        redis_address = f"{self.orchestrator.private_host}:{port}"

        with ThreadPoolExecutor(max_workers=self.factories_num) as executor:
            futures = {}

            for ins in self.factories:
                futures[executor.submit(ins.launch_node, redis_address)] = ins

            for f in futures.keys():
                try:
                    f.result()
                except errors.RemoteCommandError:
                    raise
                except Exception as exc:
                    logger.error('Generated an exception: {}'.format(exc))
                    raise
                else:
                    logger.debug('{} Ray worker ready'.format(futures[f].host))

        logger.info(cl.GR("Ray cluster launched"))

    def check_ray_cluster(self) -> bool:
        """Check if ray cluster was build successfully.

        Compares the name of workers available with the requested ones.

        Returns
        -------
        bool
            Whether the number of workers in the node
            matches the number of factories

        """
        # The orchestator needs to exist at this point
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        workers = self.orchestrator.worker_nodes()
        return len(workers) == self.factories_num

    def shutdown_ray_cluster(self) -> None:
        """Shut down the ray cluster.

        Shut down the main node running in the orchestrator.

        """
        for f in self.factories:
            f.shutdown_node()

        if self.orchestrator:
            self.orchestrator.shutdown_node()

        logger.debug("Ray cluster shutdown")

    def existing_ray_cluster(self) -> List[Instance]:
        """Return a list of the nodes in the Ray cluster.

        Returns
        -------
        List[Instance]
            The list of nodes

        """
        ret = []
        for h in self._get_all_hosts():
            if h.is_node_running():
                ret.append(h)

        return ret

    def existing_flambe_execution(self) -> List[Instance]:
        """Return a list of the hosts that are running flambe.

        Returns
        -------
        List[Instance]
            The list of nodes

        """
        ret = []
        for h in self._get_all_hosts():
            if h.is_flambe_running():
                ret.append(h)

        return ret

    def shutdown_flambe_execution(self) -> None:
        """Shut down any flambe execution in the hosts.

        """
        # The orchestator needs to exist at this point
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        # The orchestator needs to exist at this point
        if not self.factories:
            raise man_errors.ClusterError("Factories instances were not loaded.")

        for f in self.factories:
            f.shutdown_flambe()

        self.orchestrator.shutdown_flambe()

        # Flambe runs in tmux, so killing the session may be also
        # needed
        if self.orchestrator.existing_tmux_session("flambe"):
            self.orchestrator.kill_tmux_session("flambe")

        logger.debug("Flambe execution shutdown")

    def existing_dir(self, _dir: str) -> bool:
        """Determine if _dir exists in at least one host

        """
        for h in self._get_all_hosts():
            if h.existing_dir(_dir):
                return True

        return False

    def is_ray_cluster_up(self) -> bool:
        """Return if the ray cluster is running.

        Returns
        -------
        bool

        """
        # The orchestator needs to exist at this point
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        return self.orchestrator.is_node_running()

    def rollback_env(self) -> None:
        """Rollback the enviornment.

        When an error occures during the local stage of the remote
        runnable (i.e. creating the cluster, sending the data and
        submitting jobs), this method may be used to destroy the
        cluster that has been built.

        """
        raise NotImplementedError()

    def parse(self) -> None:
        """Parse the cluster object.

        Look for configurations mistakes that don't allow the remote
        runnable to run. Each different cluster will have it's own
        policies. For example, AWSCluster could check the instance
        types that are allowed. By default, checks nothing.

        Raises
        ------
        man_errors.ClusterConfigurationError
            In case the Runnable is not able to run.

        """
        pass

    def send_local_content(self,
                           content: Dict[str, str],
                           dest: str,
                           all_hosts: bool = False) -> Dict[str, str]:
        """Send local content to the cluster

        Parameters
        ----------
        content: Dict[str, str]
            The dict of resources key -> local path
        dest: str
            The orchestator's destination folder
        all_hosts: bool
            If False, only send the content to the orchestrator.
            If True, send to all factories.

        Returns
        -------
        Dict[str, str]
            The new dict of content with orchestrator's paths.

        """
        ret = {}

        # The orchestator needs to exist at this point
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        for k, c in content.items():
            c = os.path.expanduser(c)
            base: str = ""
            if os.path.exists(c):

                size = get_size_MB(c)
                if size > UPLOAD_WARN_LIMIT_MB:
                    logger.info(cl.YE(
                        f"Uploading '{c}' ({int(size)} MB) which may take a while. " +
                        "Double check you want to be transferring this file " +
                        "(note we automatically sync extensions, experiment resources " +
                        "and potentially the flambe repo if installed in dev mode)"))

                if os.path.isdir(c):
                    if not c.endswith(os.sep):
                        c = f"{c}{os.sep}"
                    base = os.path.basename(os.path.dirname(c))
                elif os.path.isfile(c):
                    base = os.path.basename(c)

                new_c = os.path.join(dest, f"{k}__{base}")
                self.orchestrator.send_rsync(c, new_c)
                logger.debug(f"Content {k}: {c} sent to cluster")

                ret[k] = new_c
            else:
                ret[k] = c

        if all_hosts:
            self.rsync_orch(dest)

        return ret

    def rsync_orch(self, folder):
        """Rsync the orchestrator's folder with all factories

        Parameters
        ----------
        folder: str
            The folder to rsync. It should be a relative path.
            $HOME value will be automatically added.

        """
        orch = self.orchestrator
        content = os.path.join(orch.get_home_path(), folder)

        for f in self.factories:
            f_path = os.path.join(f.get_home_path(), folder)
            f_loc = f"{f.username}@{f.private_host}:{f_path}"
            orch.rsync_folder(content, f_loc)

    def send_secrets(self, whitelist: List[str] = None) -> str:
        """Send the secrets file to the orchestrator.

        This file will be located in $HOME/secrets.ini
        The injected secrets file will be used.

        Parameters
        ----------
        whitelist: List[str]
            A list of sections to filter. For example: ["AWS", "GITHUB"]

        """
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        config = configparser.ConfigParser()
        for section, section_dict in self.config.items():
            if not whitelist or section in whitelist:
                config[section] = {k: v for k, v in section_dict.items()}

        secrets_path = (
            f"{self.orchestrator.get_home_path()}/secret.ini"
        )

        with tempfile.NamedTemporaryFile("w") as t:
            config.write(t)
            t.flush()
            self.orchestrator.send_rsync(t.name, secrets_path)
            logger.debug("New secrets file sent to cluster")

        return secrets_path

    def execute(self,
                cluster_runnable,
                extensions: Dict[str, str],
                new_secrets: str,
                force: bool) -> None:
        """Execute a ClusterRunnable in the cluster.

        It will first upload the runnable file + extensions to the
        orchestrator (under $HOME/flambe.yaml) and then it will
        execute it based on the provided secrets

        Parameters
        ----------
        cluster_runnable: ClusterRunnable
            The ClusterRunnable to run in the cluster
        extensions: Dict[str, str]
            The extensions for the ClusterRunnable
        new_secrets: str
            The path (relative to the orchestrator) where
            the secrets are located.
            IMPORTANT: previous to calling this method, the secrets
            should have been uploaded to the orchestrator
        force: bool
            The force parameter provided when running flambe locally

        """
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        orch_exp = (
            f"{self.orchestrator.get_home_path()}/flambe.yaml"
        )

        with tempfile.NamedTemporaryFile("w") as t:
            with StringIO() as s:
                yaml.dump_all([extensions, cluster_runnable], s)
                t.write(s.getvalue())
            t.flush()
            self.orchestrator.send_rsync(t.name, orch_exp)
            logger.info(cl.BL("Remote runnable file sent to orchestrator"))

        self.orchestrator.launch_flambe(orch_exp, new_secrets, force)

    def remove_dir(self, _dir: str, content_only: bool = True, all_hosts: bool = True) -> None:
        """ Remove a directory in the ClusterError

        Parameters
        ----------
        _dir: str
            The directory to remove
        content_only: bool
            To remove the content only or the folder also.
            Defaults to True.
        all_hosts: bool
            To remove it in all hosts or only in the Orchestrator.
            Defaults to True (in all hosts).

        """
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        if all_hosts:
            for ins in self._get_all_hosts():
                ins.remove_dir(_dir, content_only)

        else:
            self.orchestrator.remove_dir(_dir, content_only)

    def cluster_has_key(self) -> bool:
        """Whether the cluster already contains a valid common key.

        The key must be in all hosts.

        Returns
        -------
        bool
            If the cluster has a key in all hosts.

        """
        pub_key_content = None
        for ins in self._get_all_hosts():
            private_key = f"{ins.get_home_path()}/{const.PRIVATE_KEY}"
            public_key = f"{ins.get_home_path()}/{const.PUBLIC_KEY}"
            pub_ret = ins._run_cmd(f"ls {public_key}")
            priv_ret = ins._run_cmd(f"ls {private_key}")

            if not (pub_ret.success and priv_ret.success):
                return False

            pub_key_ret = ins._run_cmd(f"cat {public_key}")
            if not pub_key_ret.success:
                logger.debug(f"Not able to read file from {ins.host}")
                return False  # Not able to read key

            if pub_key_ret.success:
                curr_key_content = pub_key_ret.msg
                if pub_key_content is None or pub_key_content == curr_key_content:
                    pub_key_content = curr_key_content
                else:
                    logger.debug(f"Key in {ins.host} differs from others")
                    return False  # Keys mismatch

        logger.debug(f"All hosts contain same key pair")
        return True

    def distribute_keys(self) -> None:
        """Create a new key pair and distributes it to all hosts.

        Ensure that the hosts have a safe communication.
        The name of the key is the cluster's name

        """
        if self.cluster_has_key():
            logger.info(cl.GR("Cluster has already configured key pair"))
            return

        # generate private/public key pair
        key = rsa.generate_private_key(backend=default_backend(), public_exponent=65537,
                                       key_size=2048)

        # get public key in OpenSSH format
        public_key = key.public_key().public_bytes(serialization.Encoding.OpenSSH,
                                                   serialization.PublicFormat.OpenSSH)

        # get private key in PEM container format
        pem = key.private_bytes(encoding=serialization.Encoding.PEM,
                                format=serialization.PrivateFormat.TraditionalOpenSSL,
                                encryption_algorithm=serialization.NoEncryption())

        # decode to printable strings
        private_key_str = pem.decode('utf-8')
        public_key_str = public_key.decode('utf-8')
        logger.debug("New key pair generated")

        def m(ins):
            ins._run_cmd(f"rm -rf {ins.get_home_path()}/{const.PUBLIC_KEY}")
            ins._run_cmd(f"rm -rf {ins.get_home_path()}/{const.PRIVATE_KEY}")

            ret = ins._run_cmd(
                f"echo '{public_key_str}' >> {ins.get_home_path()}/.ssh/authorized_keys",
                retries=3
            )
            if not ret.success:
                raise man_errors.ClusterError("Could not send key to authorized_keys")

            with tempfile.NamedTemporaryFile("w") as t:
                t.write(private_key_str)
                t.flush()
                ins.send_rsync(t.name, f"{ins.get_home_path()}/{const.PRIVATE_KEY}")
                ins._run_cmd(f"chmod 600 {ins.get_home_path()}/{const.PRIVATE_KEY}")

            with tempfile.NamedTemporaryFile("w") as t:
                t.write(public_key_str)
                t.flush()
                ins.send_rsync(t.name, f"{ins.get_home_path()}/{const.PUBLIC_KEY}")
                logger.debug(f"New key pair sent to {ins.host}")

        with ThreadPoolExecutor() as executor:
            futures = {}

            for ins in self._get_all_hosts():
                futures[executor.submit(m, ins)] = ins

            for f in futures.keys():
                try:
                    f.result()
                except errors.RemoteCommandError:
                    raise
                except Exception as exc:
                    logger.error('Generated an exception: {}'.format(exc))
                    raise

        logger.info(cl.GR("Distributed keys"))

    def contains_gpu_factories(self) -> bool:
        """Return if the factories contain GPU.

        For now, all factories are same machine type,
        so as soon as a GPU is found, then this method returns.

        """
        for f in self.factories:
            if f.contains_gpu():
                return True

        return False

    def get_max_resources(self) -> Dict[str, int]:
        """Return the max common CPU/GPU devices in the factories

        For example, if one factory contains 32 CPU + 1 GPU
        and the other factory contains 16 CPU + 2 GPU, this
        method will return {"cpu": 16, "gpu": 1} available

        Returns
        -------
        Dict[str, int]
            The devices, in {"cpu": N, "gpu": M} format

        """
        ret: Dict[str, int] = {}
        for f in self.factories:
            cpus = f.num_cpus()

            if 'cpu' not in ret:
                ret['cpu'] = cpus
            else:
                ret['cpu'] = min(ret['cpu'], cpus)

            gpus = f.num_gpus()

            if 'gpu' not in ret:
                ret['gpu'] = gpus
            else:
                ret['gpu'] = min(ret['gpu'], gpus)

        return ret

    def install_extensions_in_orchestrator(self, extensions: Dict[str, str]) -> None:
        """Install local + pypi extensions in the orchestrator

        Parameters
        ----------
        extension: Dict[str, str]
            The extensions, as a dict from module_name to location

        Raises
        ------
        errors.RemoteCommandError
            If could not install an extension.
        man_errors.ClusterError
            If the orchestrator was not loaded.

        """
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        self.orchestrator.install_extensions(extensions)

    def install_extensions_in_factories(self, extensions: Dict[str, str]) -> None:
        """Install local + pypi extensions in all the factories.

        Parameters
        ----------
        extension: Dict[str, str]
            The extensions, as a dict from module_name to location

        Raises
        ------
        errors.RemoteCommandError
            If could not install an extension

        """
        for f in self.factories:
            f.install_extensions(extensions)

    def get_remote_env(self, user_provider: Callable[[], str]) -> RemoteEnvironment:
        """Get the RemoteEnvironment for this cluster.

        The IPs stored will be the private IPs

        Returns
        -------
        RemoteEnvironment
            The RemoteEnvironment with information about this cluster.

        """
        if not self.orchestrator:
            raise man_errors.ClusterError("Orchestrator instance was not loaded.")

        # Use compile method so that is serializable
        return RemoteEnvironment(
            key=f"{self.orchestrator.get_home_path()}/{const.PRIVATE_KEY}",
            orchestrator_ip=self.orchestrator.private_host,
            factories_ips=[f.private_host for f in self.factories],
            user=self.orchestrator.username,
            local_user=user_provider(),
            public_orchestrator_ip=self.orchestrator.host,
            public_factories_ips=[f.host for f in self.factories]
        )
