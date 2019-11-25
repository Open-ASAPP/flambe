from abc import abstractmethod
from typing import Optional, Dict, Callable

from flambe.runnable import Runnable
from flambe.cluster import Cluster
from flambe.runnable.environment import RemoteEnvironment


class ClusterRunnable(Runnable):
    """Base class for all runnables that are able to run on cluster.

    This type of Runnables must include logic in the 'run' method to
    deal with the fact that they could be running in a distributed
    cluster of machines.

    To provide useful information about the cluster, a RemoteEnvironment
    object will be injected when running remotely.

    Attributes
    ----------
    config: configparser.ConfigParser
        The secrets that the user provides. For example,
        'config["AWS"]["ACCESS_KEY"]'
    env: RemoteEnvironment
        The remote environment has information about the cluster
        where this ClusterRunnable will be running.
        IMPORTANT: this object will be available only when
        the ClusterRunnable is running remotely.
    user_provider: Callable[[], str]
        The logic for specifying the user triggering this
        Runnable. If not passed, by default it will pick the computer's
        user.

    """
    def __init__(self, user_provider: Callable[[], str] = None,
                 env: Optional[RemoteEnvironment] = None, **kwargs) -> None:
        super().__init__(user_provider=user_provider, **kwargs)
        self.env = env

    @abstractmethod
    def setup(self, cluster: Cluster,
              extensions: Dict[str, str],
              force: bool, **kwargs) -> None:
        """Setup the cluster.

        Parameters
        ----------
        cluster: Cluster
            The cluster where this Runnable will be running
        extensions: Dict[str, str]
            The ClusterRunnable extensions
        force: bool
            The force value provided to Flambe

        """
        raise NotImplementedError()

    def setup_inject_env(self, cluster: Cluster,
                         extensions: Dict[str, str],
                         force: bool, **kwargs) -> None:
        """Call setup and inject the RemoteEnvironment

        Parameters
        ----------
        cluster: Cluster
            The cluster where this Runnable will be running
        extensions: Dict[str, str]
            The ClusterRunnable extensions
        force: bool
            The force value provided to Flambe

        """
        self.setup(cluster=cluster, extensions=extensions, force=force, **kwargs)
        self.set_serializable_attr("env", cluster.get_remote_env(self.user_provider))

    def set_serializable_attr(self, attr, value):
        """Set an attribute while keep supporting serializaton.
        """
        setattr(self, attr, value)
        if hasattr(self, "_saved_kwargs"):
            self._saved_kwargs[attr] = value
