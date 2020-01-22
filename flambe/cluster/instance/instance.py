"""This modules includes base Instance classes to represent machines.

All Instance objects will be managed by Cluster objects
(`flambe.cluster.cluster.Cluster`).

This base implementation is independant to the type of instance used.

Any new instance that flambe should support should inherit from the
classes that are defined in this module.

"""
# from __future__ import annotations

import time
import os
import paramiko
from paramiko.client import SSHClient
import socket
import contextlib
import subprocess

import logging
import uuid

from configparser import ConfigParser
from typing import Optional, Type, Generator, TypeVar, List, Dict
from types import TracebackType

import flambe
from flambe.cluster import const
from flambe.cluster.utils import RemoteCommand
from flambe.cluster.instance import errors

from flambe.runnable.utils import get_flambe_repo_location
from flambe.logging import coloredlogs as cl

logger = logging.getLogger(__name__)

InsT = TypeVar("InsT", bound="Instance")


class Instance(object):
    """Encapsulates remote instances.

    In this context, the instance is a running computer.

    All instances used by flambe remote mode will inherit
    `Intance`. This class provides high-level methods to deal with
    remote instances (for example, sending a shell command over SSH).

    *Important: Instance objects should be pickeable.* Make sure that
    all child classes can be pickled.

    The flambe local process will communicate with the remote instances
    using SSH. The authentication mechanism will be using private keys.

    Parameters
    ----------
    host : str
        The public DNS host of the remote machine.
    private_host : str
        The private DNS host of the remote machine.
    username : str
        The machine's username.
    key: str
        The path to the ssh key used to communicate to the instance.
    config : ConfigParser
        The config object that contains useful information for the
        instance. For example, `config['SSH']['SSH_KEY']` should
        contain the path of the ssh key to login the remote instance.
    debug : bool
        True in case flambe was installed in dev mode, False otherwise.
    use_public : bool
        Wether this instance should use public or private IP. By
        default, the public IP is used. Private host is used when
        inside a private LAN.

    """

    def __init__(self,
                 host: str,
                 private_host: str,
                 username: str,
                 key: str,
                 config: ConfigParser,
                 debug: bool,
                 use_public: bool = True) -> None:
        self.host = host
        self.private_host = private_host
        self.username = username
        self.key = key

        self.config = config
        self.fix_relpaths_in_config()

        self.use_public = use_public
        self.debug = debug

        # Uses only one ssh client per instance object
        self._cli: SSHClient = None

    def fix_relpaths_in_config(self) -> None:
        """Updates all paths to be absolute.
        For example, if it contains "~/a/b/c" it will be change to
        /home/user/a/b/c (the appropiate $HOME value)

        """
        for section in self.config:
            for k, v in self.config[section].items():
                if os.path.exists(v) and not os.path.isabs(v):
                    self.config[section][k] = os.path.abspath(v)
                if v.startswith("~"):
                    self.config[section][k] = os.path.expanduser(v)

    def __enter__(self):
        """Method to use `Instance` instances with context managers

        Returns
        -------
        Instance
            The current instance

        """
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]):
        """Exit method for the context manager.

        This method will catch any uprising exception and raise it.

        """
        if exc_value is not None:
            print(f"{exc_value}")

        return False

    def prepare(self) -> None:
        """Runs all neccessary processes to prepare the instances.

        The child classes should implement this method
        according to the type of instance.

        """
        raise NotImplementedError()

    def wait_until_accessible(self) -> None:
        """Waits until the instance is accesible through SSHClient

        It attempts `const.RETRIES` time to ping SSH port to See
        if it's listening for incoming connections. In each attempt,
        it waits `const.RETRY_DELAY`.

        Raises
        ------
        ConnectionError
            If the instance is unaccesible through SSH

        """
        retry_count = 0

        while retry_count <= const.RETRIES:
            if self.is_up():
                logger.debug(f"Instance {self.host} is UP & accessible on port 22")
                return

            time.sleep(const.RETRY_DELAY)
            retry_count += 1
            logger.debug(f"Instance {self.host} not accesible. Retrying")

        raise ConnectionError(f"{self.host} is unreachable through ssh.")

    def is_up(self) -> bool:
        """Tests wether port 22 is open to incoming SSH connections

        Returns
        -------
        bool
            True if instance is listening in port 22. False otherwise.

        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(const.RETRY_DELAY)
        result = sock.connect_ex((self.host if self.use_public else self.private_host, 22))
        return result == 0

    def _get_cli(self) -> paramiko.SSHClient:
        """Get an `SSHClient` in order to execute commands.

        This will cache an existing SSHClient to optimize resource.
        This is a private method and should only be used in this module.

        Returns
        -------
        paramiko.SSHClient
            The client for latter use.

        Raises
        ------
        SSHConnectingError
            In case opening an SSH connection fails.

        """
        try:
            if (self._cli is None or self._cli.get_transport() is None or
                    not self._cli.get_transport().is_active()):
                # Set cli in case it was not set or if it was closed
                cli = paramiko.SSHClient()
                cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                hostname = self.host if self.use_public else self.private_host
                cli.connect(hostname=hostname,
                            username=self.username, key_filename=self.key,
                            allow_agent=False, look_for_keys=False)
                self._cli = cli
            return self._cli
        except paramiko.ssh_exception.SSHException:
            raise errors.SSHConnectingError(f"Error opening SSH connection with {hostname}. "
                                            "Double check information provided in the secrets file")

    def _run_cmd(self, cmd: str, retries: int = 1, wd: str = None) -> RemoteCommand:
        """Runs a single shell command in the instance through SSH.

        The command will be executed in one ssh connection.
        Don't expect calling several time to `_run_cmd` expecting to
        keep state between commands. To use mutliple commands, use:
        `_run_script`

        *Important: when running docker containers, don't use -it flag!*

        This is a private method and should only be used in this module.

        Parameters
        ----------
        cmd : str
            The command to execute.
        retries : int
            The amount of attempts to run the command if it fails.
            Default to 1.
        wd : str
            The working directory to 'cd' before running the command

        Returns
        -------
        RemoteCommand
            A `RemoteCommand` instance with success boolean and message.

        Examples
        --------
        To get $HOME env

        >>> instance._run_cmd("echo $HOME")
        RemoteCommand(True, "/home/ubuntu")

        This will not work

        >>> instance._run_cmd("export var=10")
        >>> instance._run_cmd("echo $var")
        RemoteCommand(False, "")

        This will work

        >>> instance._run_cmd("export var=10; echo $var")
        RemoteCommand(True, "10")

        Raises
        ------
        RemoteCommandError
            In case the `cmd` failes after `retries` attempts.

        """
        if retries <= 0:
            raise ValueError("'retries' parameter should be > 0")

        for i in range(retries):
            cli = self._get_cli()

            try:
                if wd:
                    cmd = f"cd {wd}; {cmd}"

                status, stdout, stderr = cli.exec_command(cmd)

                # Blocks until done
                while not stdout.channel.exit_status_ready():
                    status = stdout.channel.recv_exit_status()

                out, err = stdout.read(), stderr.read()

                success = status == 0

                if not success:
                    logger.debug(f"Retry {i}. {cmd} failed with message: {err}")
                else:
                    logger.debug(f"'{cmd}' ran successfully")
                    return RemoteCommand(success, out if success else err)

            except errors.SSHConnectingError:
                raise
            except Exception as err:
                raise errors.RemoteCommandError(err)

        logger.debug(f"'{cmd}' returning after {retries} intents returning != 0")
        return RemoteCommand(success, out if success else err)

    def _run_script(self, fname: str, desc: str) -> RemoteCommand:
        """Runs a script by copyinh the script to the instance and
        executing it.

        This is a private method and should only be used in this module.

        Parameters
        ----------
        fname : str
            The script filename
        desc : str
            A description for the script purpose. This will be used
            for the copied filename

        Returns
        -------
        RemoteCommand
            A `RemoteCommand` instance with success boolean and message.

        Raises
        ------
        RemoteCommandError
            In case the script fails.

        """
        # TODO it can exist
        with self._remote_script(fname, desc) as rs:
            return self._run_cmd(f"./{rs}")

    @contextlib.contextmanager
    def _remote_script(self, host_fname: str, desc: str) -> Generator[str, None, None]:
        """Sends a local file containing a script to the instance
        using Paramiko SFTP.

        It should be used as a context manager for latter execution of
        the script. See `_run_script` on how to use it.

        After the context manager exists, then the file is removed from
        the instance.

        This is a private method and should only be used in this module.

        Parameters
        ----------
        host_fname : str
            The local script filename
        desc : str
            A description for the script purpose. This will be used
            for the copied filename

        Yields
        -------
        str
            The remote filename of the copied local file.

        Raises
        ------
        RemoteCommandError
            In case sending the script fails.

        """
        random_fname = f"{desc}_{uuid.uuid4().hex}.sh"

        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cli.connect(hostname=self.host, username=self.username,
                    key_filename=self.key)
        sftp = cli.open_sftp()

        try:
            random_fname = f"{desc}_{uuid.uuid4().hex}.sh"
            sftp.put(host_fname, random_fname)
            cmd = self._run_cmd(f"chmod +x {random_fname}")

            if cmd.success:
                yield random_fname
            else:
                raise errors.RemoteCommandError(f"Error sending local script. {cmd.msg}")

        finally:
            sftp.remove(random_fname)
            sftp.close()
            cli.close()

    def run_cmds(self, setup_cmds: List[str]) -> None:
        """Execute a list of sequential commands

        Parameters
        ----------
        setup_cmds: List[str]
            The list of commands

        Returns
        -------
        RemoteCommandError
            In case at least one command is not successful

        """
        for s in setup_cmds:
            ret = self._run_cmd(s, retries=3)
            if not ret.success:
                raise errors.RemoteCommandError(f"Error executing {s} in {self.host}. " +
                                                f"{ret.msg}")

    def send_rsync(self, host_path: str, remote_path: str, params: List[str] = None) -> None:
        """Send a local file or folder to a remote instance with rsync.

        Parameters
        ----------
        host_path : str
            The local filename or folder
        remote_path : str
            The remote filename or folder to use
        params : List[str], optional
            Extra parameters to be passed to rsync.
            For example, ["--filter=':- .gitignore'"]

        Raises
        ------
        RemoteFileTransferError
            In case sending the file fails.

        """
        if not os.path.exists(host_path):
            raise errors.RemoteFileTransferError(f"{host_path} does not exist.")

        _from = host_path
        if os.path.isdir(host_path) and not host_path.endswith(os.sep):
            _from = f"{host_path}{os.sep}"

        _to = f"{self.username}@{self.host if self.use_public else self.private_host}:{remote_path}"

        rsync_params = ""
        if params:
            rsync_params = " ".join(params)
        cmd = (
            f'rsync {rsync_params} -ae "ssh -i {self.key} -o StrictHostKeyChecking=no" '
            f'{_from} {_to}'
        )
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  shell=True)
            logger.debug(f"rsync {host_path} -> {remote_path} successful")
        except subprocess.CalledProcessError as e:
            raise errors.RemoteFileTransferError(e)

    def get_home_path(self) -> str:
        """Return the $HOME value of the instance.

        Returns
        -------
        str
            The $HOME env value.

        Raises
        ------
        RemoteCommandError
            If after 3 retries it is not able to get $HOME.

        """
        cmd = self._run_cmd("echo $HOME", retries=3)

        if cmd.success:
            return cmd.msg.decode("utf-8").strip()

        raise errors.RemoteCommandError(f"Could not access $HOME env variable. {cmd.msg}")

    def clean_containers(self) -> None:
        """Stop and remove all containers running

        Raises
        ------
        RemoteCommandError
            If command fails

        """
        cmd = f'''
        docker stop $(docker ps -a -q);
        docker rm $(docker ps -a -q);
        '''

        ret = self._run_cmd(cmd)

        if not ret.success:
            raise errors.RemoteCommandError("Could not clean containers")

    def clean_container_by_image(self, image_name: str) -> None:
        """Stop and remove all containers given an image name.

        Parameters
        ----------
        image_name : str
            The name of the image for which all containers
            should be stopped and removed.

        Raises
        ------
        RemoteCommandError
            If command fails

        """
        cmd = f"docker rm $(docker stop "\
              f"$(docker ps -a -q --filter ancestor={image_name} --format='{{{{.ID}}}}'))"
        res = self._run_cmd(cmd)

        if not res.success:
            raise errors.RemoteCommandError(f"Could not clean container {image_name}. {res.msg}")

    def clean_container_by_command(self, command: str) -> None:
        """Stop and remove all containers with the given command.

        Parameters
        ----------
        command : str
            The command used to stop and remove the containers

        Raises
        ------
        RemoteCommandError
            If command fails

        """
        cmd = f"docker rm -f $(docker inspect -f '{{{{.ID}}}} "\
              f"{{{{.Config.Cmd}}}}' $(docker ps -a -q) | grep {command} | awk '{{print $1}}')"
        res = self._run_cmd(cmd)

        if not res.success:
            raise errors.RemoteCommandError(
                f"Could not clean container with cmd {command}. {res.msg}")

    def install_docker(self) -> None:
        """Install docker in a Ubuntu 18.04 distribution.

        Raises
        ------
        RemoteCommandError
            If it's not able to install docker.
            ie. then the installation script fails

        """
        fname = os.path.join(os.path.dirname(__file__), "scripts/install_docker.sh")
        cmd = self._run_script(fname, "install_docker")
        if not cmd.success:
            raise errors.RemoteCommandError(f"Could not install docker. {cmd.msg}")

    def install_extensions(self, extensions: Dict[str, str]) -> None:
        """Install local + pypi extensions.

        Parameters
        ----------
        extension: Dict[str, str]
            The extensions, as a dict from module_name to location

        Raises
        ------
        errors.RemoteCommandError
            If could not install an extension

        """
        cmd = ['python3', '-m', 'pip', 'install', '-U', '--user']
        for ext, resource in extensions.items():
            curr_cmd = cmd[:]

            if 'PIP' in self.config:
                host = self.config['PIP'].get('HOST', None)
                if host:
                    curr_cmd.extend(["--trusted-host", host])

                host_url = self.config['PIP'].get('HOST_URL', None)
                if host_url:
                    curr_cmd.extend(["--extra-index-url", host_url])

            if os.path.exists(resource):
                # Package is local
                if os.sep not in resource:
                    resource = f"./{resource}"
            else:
                # Package follows pypi notation: "torch>=0.4.1,<1.1"
                resource = f"{resource}"

            curr_cmd.append(resource)

            ret = self._run_cmd(" ".join(curr_cmd))
            if not ret.success:
                raise errors.RemoteCommandError(
                    f"Could not install package {resource} in {self.host}"
                )

    def install_flambe(self) -> None:
        """Pip install Flambe.

        If dev mode is activated, then it rsyncs the local flambe
        folder and installs that version. If not, downloads from pypi.

        Raises
        ------
        RemoteCommandError
            If it's not able to install flambe.

        """
        flags = []
        if 'PIP' in self.config:
            host = self.config['PIP'].get('HOST', None)
            if host:
                flags.append(f"--trusted-host {host}")

            host_url = self.config['PIP'].get('HOST_URL', None)
            if host_url:
                flags.append(f"--extra-index-url {host_url}")

        if not self.debug:
            pip_flambe = "flambe" if not self.contains_gpu() else "flambe[cuda]"
            logger.debug(f"Installing flambe in {self.host} using pypi")
            ret = self._run_cmd(
                f"python3 -m pip install --user --upgrade "
                f"{' '.join(flags)} {pip_flambe}=={flambe.__version__}",
                retries=3
            )

        else:
            origin = get_flambe_repo_location()
            destination = os.path.join(self.get_home_path(), "extensions", "flambe")

            self.send_rsync(origin, destination, params=["--exclude='.*'", "--exclude='docs/*'"])
            logger.debug(f"Sent flambe {origin} -> {destination}")
            pip_destination = destination if not self.contains_gpu() else f"{destination}[cuda]"
            ret = self._run_cmd(
                f"python3 -m pip install --user --upgrade {' '.join(flags)} {pip_destination}",
                retries=3
            )

        if not ret.success:
            raise errors.RemoteCommandError(f"Could not install flambe. {ret.msg}")
        else:
            logger.debug(f"Installed flambe in {self.host} successfully")

    def is_docker_installed(self) -> bool:
        """Check if docker is installed in the instance.

        Executes command "docker --version" and expect it not to fail.

        Returns
        -------
        bool
            True if docker is installed. False otherwise.

        """
        cmd = self._run_cmd("docker --version")
        return cmd.success

    def is_flambe_installed(self, version: bool = True) -> bool:
        """Check if flambe is installed and if it matches version.

        Parameters
        ----------
        version: bool
            If True, also the version will be used. That is, if flag
            is True and the remote flambe version is different from the
            local flambe version, then this method will return False.
            If they match, then True. If version is False this method
            will return if there is ANY flambe version in the host.

        Returns
        ------
        bool

        """
        # First check if a version of flambe is installed
        ret = self._run_cmd("bash -lc 'flambe --help'")
        if not ret.success:
            return False

        if version:
            cmd = "python3 -c 'import flambe; print(flambe.__version__)'"
            ret = self._run_cmd(cmd)

            if not ret.success:
                raise errors.RemoteCommandError(
                    f"Could not run flambe in python at {self.host} even if binary was found."
                )

            return ret.msg.strip() == bytes(flambe.__version__, 'utf-8')

        return True

    def is_docker_running(self) -> bool:
        """Check if docker is running in the instance.

        Executes the command "docker ps" and expects it not to fail.

        Returns
        -------
        bool
            True if docker is running. False otherwise.

        """
        cmd = self._run_cmd("docker ps")
        return cmd.success

    def start_docker(self) -> None:
        """Restart docker.

        Raises
        ------
        RemoteCommandError
            If it's not able to restart docker.

        """
        cmd = self._run_cmd("sudo systemctl restart docker")
        if not cmd.success:
            raise errors.RemoteCommandError(f"Could not start docker. {cmd.msg}")

    def is_node_running(self) -> bool:
        """Return if the host is running a ray node

        Returns
        -------
        bool

        """
        cmd = "ps -e | grep ray"
        ret = self._run_cmd(cmd)
        return ret.success

    def is_flambe_running(self) -> bool:
        """Return if the host is running flambe

        Returns
        -------
        bool

        """
        cmd = "ps axco command | grep -P ^flambe$"
        ret = self._run_cmd(cmd)
        return ret.success

    def existing_dir(self, _dir: str) -> bool:
        """Return if a directory exists in the host

        Parameters
        ----------
        _dir: str
            The name of the directory. It needs to be relative to $HOME

        Returns
        -------
        bool
            True if exists. Otherwise, False.

        """
        cmd = f"[ -d {self.get_home_path()}/{_dir} ]"
        ret = self._run_cmd(cmd)
        return ret.success

    def shutdown_node(self) -> None:
        """Shut down the ray node in the host.

        If the node is also the main node, then the entire
        cluster will shut down

        """
        if not self.is_node_running():
            logger.debug("Tried to shutdown a non existing node")
            return

        cmd = self._run_cmd(
            "bash -lc 'ray stop'")

        if cmd.success:
            logger.debug(f"Ray node stopped at {self.host}")
        else:
            raise errors.RemoteCommandError(f"Ray node failed to stop. {cmd.msg}")

    def shutdown_flambe(self) -> None:
        """Shut down flambe in the host

        """
        if not self.is_flambe_running():
            logger.debug("Tried to shutdown flambe in a host that it's not runing flambe")
            return

        cmd = self._run_cmd("killall -9 flambe")

        if cmd.success:
            logger.debug(f"Flambe killed in {self.host}")
        else:
            raise errors.RemoteCommandError(f"Flambe failed to be shutdown. {cmd.msg}")

    def create_dirs(self, relative_dirs: List[str]) -> None:
        """Create the necessary folders in the host.

        Parameters
        ----------
        relative_dirs: List[str]
            The directories to create. They should be relative paths
            and $HOME of each host will be used to add the prefix.

        """
        # Create user's folders to deal with the experiment
        for d in relative_dirs:
            ret = self._run_cmd(f"[ -d {d} ]")
            if not ret.success:
                ret = self._run_cmd(f"mkdir -p {d}")
                if ret.success:
                    logger.debug(f"Foldcreate_dirs {d} created in {self.host}")
            else:
                logger.debug(f"Existing folder {d} in {self.host}")

    def remove_dir(self, _dir: str, content_only: bool = True) -> None:
        """Delete the specified dir result folder.

        Parameters
        ----------
        _dir: str
            The directory. It needs to be relative to the $HOME path as
            it will be prepended as a prefix.
        content_only: bool
            If True, the folder itseld will not be erased.

        """
        path = f"{self.get_home_path()}/{_dir}"
        if content_only:
            cmd = self._run_cmd(f"rm -rf {path}/*")
        else:
            cmd = self._run_cmd(f"rm -rf {path}/")
        if cmd.success:
            logger.debug(f"Removed {path} at {self.host}.")
        else:
            raise errors.RemoteCommandError(
                f"Failed to remove {path} on {self.host}. {cmd.msg}"
            )

    def contains_gpu(self) -> bool:
        """Return if this machine contains GPU.

        This method will be used to possibly upgrade
        this factory to a GPUFactoryInstance.

        """
        cmd = "python3 -c 'import torch; print(torch.cuda.is_available())'"
        ret = self._run_cmd(cmd)

        if not ret.success:
            raise errors.RemoteCommandError("Factory does not contain torch installed")

        return ret.msg.strip() == b"True"


class CPUFactoryInstance(Instance):
    """This class represents a CPU Instance in the Ray cluster.

    CPU Factories are instances that can run only one worker
    (no GPUs available). This class is mostly useful debugging.

    Factory instances will not keep any important information.
    All information is going to be sent to an orchestrator machine.
    """

    def prepare(self) -> None:
        """Prepare a CPU machine to be a worker node.

        Checks if flambe is installed, and if not, installs it.

        Raises
        ------
        RemoteCommandError
            In case any step of the preparing process fails.

        """
        self.install_flambe()

    def launch_node(self, redis_address: str) -> None:
        """Launch the ray worker node.

        Parameters
        ----------
        redis_address : str
            The URL of the main node. Must be IP:port

        Raises
        ------
        RemoteCommandError
            If not able to run node.

        """
        # https://stackoverflow.com/a/18665363.
        # `ray` is in ~/.local/bin that is not in $PATH in paramiko.
        # For this, use bash and -lc flags
        cmd = f"bash -lc 'ray start --redis-address {redis_address}'"
        ret = self._run_cmd(cmd)

        if ret.success:
            logger.debug(f"Ray worker node launched at {self.host}")
        else:
            raise errors.RemoteCommandError(f"Could not launch worker node. {ret.msg}")

    def num_cpus(self) -> int:
        """Return the number of CPUs this host contains.

        """
        cmd = self._run_cmd(f"python3 -c 'import multiprocessing; " +
                            "print(multiprocessing.cpu_count())'")

        if cmd.success:
            return int(cmd.msg)

        raise errors.RemoteCommandError(f"Could not find out the number of CPUs. {cmd.msg}")

    def num_gpus(self) -> int:
        """Get the number of GPUs this host contains

        Returns
        -------
        int
            The number of GPUs

        Raises
        ------
        RemoteCommandError
            If command to get the number of GPUs fails.

        """
        cmd = self._run_cmd(f"python3 -c 'import torch; print(torch.cuda.device_count())'")

        if cmd.success:
            return int(cmd.msg)

        raise errors.RemoteCommandError(f"Could not find out how many GPUs available. {cmd.msg}")


class GPUFactoryInstance(CPUFactoryInstance):
    """This class represents an Nvidia GPU Factory Instance.

    Factory instances will not keep any important information.
    All information is going to be sent to an Orchestrator machine.

    """

    def prepare(self) -> None:
        """Prepare a GPU instance to run a ray worker node. For this, it
        installs CUDA and flambe if not installed.

        Raises
        ------
        RemoteCommandError
            In case any step of the preparing process fails.

        """
        if not self.is_cuda_installed():
            logger.debug(f"Installing CUDA at {self.host}")
            self.install_cuda()

        super().prepare()

    def install_cuda(self) -> None:
        """Install CUDA 10.0 drivers in an Ubuntu 18.04 distribution.

        Raises
        ------
        RemoteCommandError
            If it's not able to install drivers. ie if script fails

        """
        fname = os.path.join(os.path.dirname(__file__), "scripts/install_cuda_ubuntu1804.sh")
        cmd = self._run_script(fname, "install_cuda")

        if not cmd.success:
            raise errors.RemoteCommandError(f"Could not install CUDA. {cmd.msg}")

    def is_cuda_installed(self) -> bool:
        """Check if CUDA is installed trying to execute `nvidia-smi`

        Returns
        -------
        bool
            True if CUDA is installed. False otherwise.

        """
        cmd = self._run_cmd("nvidia-smi")
        return cmd.success


class OrchestratorInstance(Instance):
    """The orchestrator instance will be the main machine in a cluster.

    It is going to be the main node in the ray cluster and it will
    also host other services. TODO: complete

    All services besides ray will run in docker containers.

    This instance does not needs to be a GPU machine.

    """

    def prepare(self) -> None:
        """Install docker and flambe

        Raises
        ------
        RemoteCommandError
            In case any step of the preparing process fails.

        """
        if not self.is_docker_installed():
            self.install_docker()

        if not self.is_docker_running():
            self.start_docker()

        self.install_flambe()

    def launch_report_site(self, progress_file: str,
                           port: int,
                           output_log: str,
                           output_dir: str,
                           tensorboard_port: int) -> None:
        """Launch the report site.

        The report site is a Flask web app.

        Raises
        ------
        RemoteCommandError
            In case the launch process fails

        """
        tensorboard_url = f"http://{self.host}:{tensorboard_port}"

        cmd = (
            f"tmux new-session -d -s 'flambe-site' 'bash -lc \"flambe-site {progress_file} "
            f"--tensorboard_url {tensorboard_url} "
            f"--host 0.0.0.0 --port {port} "
            f"--output-dir {output_dir} "
            f"--output-log {output_log} &>> outputsite.log\"'"
        )

        res = self._run_cmd(cmd)

        # Sometimes tmux command returns failure (because of some
        # timeout) but website is running.
        # Adding this extra check in that case.
        if res.success or self.is_report_site_running():
            logger.info(cl.BL(f"Report site at http://{self.host}:{port}"))
        else:
            raise errors.RemoteCommandError(f"Report site failed to run. {res.msg}")

    def is_tensorboard_running(self) -> bool:
        """Return wether tensorboard is running in the host as docker.

        Returns
        -------
        bool
            True if Tensorboard is running, False otherwise.

        """
        cmd = "docker ps | grep tensorboard"
        ret = self._run_cmd(cmd)
        return ret.success

    def is_report_site_running(self) -> bool:
        """Return wether the report site is running in the host

        Returns
        -------
        bool

        """
        cmd = "ps axco command | grep -P ^flambe-site$"
        ret = self._run_cmd(cmd)
        return ret.success

    def remove_tensorboard(self) -> None:
        """Removes tensorboard from the orchestrator.

        """
        self.clean_container_by_command("tensorboard")

    def remove_report_site(self) -> None:
        """Remove report site from the orchestrator.

        """
        cmd = "pkill flambe-site"
        ret = self._run_cmd(cmd)

        if self.existing_tmux_session("flambe-site"):
            self.kill_tmux_session("flambe-site")

        return ret.success

    def launch_tensorboard(self,
                           logs_dir: str,
                           tensorboard_port: int) -> None:
        """Launch tensorboard.

        Parameters
        ----------
        logs_dir : str
            Tensorboard logs directory
        tensorboard_port: int
            The port where tensorboard will be available

        Raises
        ------
        RemoteCommandError
            In case the launch process fails

        """
        if not self.is_docker_installed():
            logger.error("Can't run tensorboard. Docker not installed.")
            return

        cmd = self._run_cmd(
            f"docker run -d -p {tensorboard_port}:6006 -v " +
            f"{os.path.join(self.get_home_path(), logs_dir)}:" +
            f"/tensorboard_logs {const.TENSORBOARD_IMAGE} tensorboard --logdir /tensorboard_logs")
        if cmd.success:
            logger.debug(f"Tensorboard running at http://{self.host}:{tensorboard_port} . " +
                         "Be aware that it can take a while until it starts showing results.")
        else:
            raise errors.RemoteCommandError(f"Tensorboard stable failed to run. {cmd.msg}")

    def existing_tmux_session(self, session_name: str) -> bool:
        """Return if there is an existing tmux session with the same
        name

        Parameters
        ----------
        session_name: str
            The exact name of the searched tmux session

        Returns
        -------
        bool

        """
        cmd = f'tmux ls -F "#{{session_name}}" | grep -P ^{session_name}$'
        ret = self._run_cmd(cmd)
        return ret.success

    def kill_tmux_session(self, session_name: str) -> None:
        """Kill an existing tmux session

        Parameters
        ----------
        session_name: str
            The exact name of the tmux session to be removed

        """
        cmd = f'tmux kill-session -t {session_name}'
        ret = self._run_cmd(cmd)

        if ret.success:
            logger.debug(f"Remove existing tmux session {session_name}")
        else:
            raise errors.RemoteCommandError(f"Tried to remove a session. {ret.msg}")

    def launch_flambe(self,
                      config_file: str,
                      secrets_file: str,
                      force: bool) -> None:
        """Launch flambe execution in the remote host

        Parameters
        ----------
        config_file: str
            The config filename relative to the orchestrator
        secrets_file: str
            The filepath containing the secrets for the orchestrator
        force: bool
            The force parameters that was originally passed to flambe

        """
        force_params = "--force" if force else ""
        cmd = (
            f"tmux new-session -d -s 'flambe' " +
            f"'bash -lc \"flambe {config_file} --secrets {secrets_file} " +
            f"{force_params} &> output.log\"'"
        )

        ret = self._run_cmd(cmd)

        # Sometimes tmux command returns failure (because of some
        # timeout) but flambe is running.
        # Adding this extra check in that case.
        if ret.success or self.is_flambe_running():
            logger.info(cl.GR("Running flambe in Orchestrator"))
        else:
            raise errors.RemoteCommandError(f"Not able to run flambe. {ret.msg}")

    def launch_node(self, port: int) -> None:
        """Launch the main ray node in given sftp server in port 49559.

        Parameters
        ----------
        port: int
            Available port to launch the redis DB of the main ray node

        Raises
        ------
        RemoteCommandError
            In case the launch process fails

        """
        # https://stackoverflow.com/a/18665363.
        # `ray` is in ~/.local/bin that is not in $PATH in paramiko.
        # For this, use bash and -lc flags
        cmd = self._run_cmd(
            f"bash -lc 'ray start --head --num-cpus=0 --redis-port={port}'")

        if cmd.success:
            logger.debug(f"Ray main node running in {self.host}")
        else:
            raise errors.RemoteCommandError(f"Ray main node failed to run. {cmd.msg}")

    def worker_nodes(self) -> List[str]:
        """Returns the list of worker nodes

        Returns
        -------
        List[str]
            The list of worker nodes identified by their hostname

        """
        redis_address = f"\"{self.private_host}:{const.RAY_REDIS_PORT}\""
        cmd = "python3 -c '\n"\
              "import time\n"\
              "import ray\n"\
              f"ray.init(redis_address={redis_address})\n"\
              "@ray.remote\n"\
              "def f():\n"\
              "    time.sleep(0.01)\n"\
              "    return ray.services.get_node_ip_address()\n"\
              "print(set(ray.get([f.remote() for _ in range(1000)])))\n'"

        ret = self._run_cmd(cmd)

        if not ret.success:
            raise errors.RemoteCommandError(f"Failed to run Python script. {ret.msg}")

        return [s[1:-1] for s in (ret.msg[1:-2]).decode("utf-8").split(',')]

    def rsync_folder(self, _from, _to, exclude=None):
        """Rsyncs folders or files.

        One of the folders NEEDS to be local. The remaining one can
        be remote if needed.

        """
        # -o StrictHostKeyChecking=no if not rsync asks for accepting
        # key and it fails (the cmd is not providing the 'yes' answer)
        # with message Host keys verification failed. \r\nrsync:
        # connection unexpectedly closed (0 bytes received so far)
        exc = ""
        if exclude:
            for x in exclude:
                exc += f" --exclude {x} "

        if not _from.endswith(os.sep):
            _from = f"{_from}{os.sep}"

        cmd = (
            f"rsync -ae 'ssh -i {self.get_home_path()}/{const.PRIVATE_KEY} "
            f"-o StrictHostKeyChecking=no' {exc} "
            f"{_from} {_to}"
        )

        ret = self._run_cmd(cmd)

        if not ret.success:
            logger.debug(f"Could not rsync between {self.private_host} and {_to}")
        else:
            logger.debug(f"Rsync successful between {_from} -> {_to}")
