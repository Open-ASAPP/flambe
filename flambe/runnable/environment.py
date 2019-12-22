from typing import List, Any, Optional
from flambe.compile import Registrable


class RemoteEnvironment(Registrable):
    """This objects contains information about the cluster

    This object will be available on the remote execution of
    the ClusterRunnable (as an attribute).

    IMPORTANT: this object needs to be serializable, hence it Needs
    to be created using 'compile' method.

    Attributes
    ----------
    key: str
        The key that communicates the cluster
    orchestrator_ip: str
        The orchestrator visible IP for the factories (usually
        the private IP)
    factories_ips: List[str]
        The list of factories IPs visible for other factories and
        orchestrator (usually private IPs)
    user: str
        The username of all machines. This implementations assumes
        same username for all machines
    local_user: str
        The username of the local process that launched the cluster
    public_orchestrator_ip: Optional[str]
        The public orchestrator IP, if available.
    public_factories_ips: Optional[List[str]]
        The public factories IPs, if available.

    """

    def __init__(self,
                 key: str,
                 orchestrator_ip: str,
                 factories_ips: List[str],
                 user: str,
                 local_user: str,
                 public_orchestrator_ip: Optional[str] = None,
                 public_factories_ips: Optional[List[str]] = None,
                 **kwargs) -> None:
        self.key = key
        self.orchestrator_ip = orchestrator_ip
        self.factories_ips = factories_ips
        self.user = user
        self.local_user = local_user

        self.public_orchestrator_ip = public_orchestrator_ip
        self.public_factories_ips = public_factories_ips

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        kwargs = {'key': node.key,
                  'orchestrator_ip': node.orchestrator_ip,
                  'factories_ips': node.factories_ips,
                  'public_orchestrator_ip': node.public_orchestrator_ip,
                  'public_factories_ips': node.public_factories_ips,
                  'user': node.user,
                  'local_user': node.local_user}
        return representer.represent_mapping(tag, kwargs)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> Any:
        """Use constructor to create an instance of cls"""
        # NOTE: construct_yaml_map is a generator that yields the
        # constructed data and then updates it
        kwargs, = list(constructor.construct_yaml_map(node))
        return cls(**kwargs)
