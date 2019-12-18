import logging

from typing import Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.common.google import GoogleBaseError

from flambe.cluster.errors import ClusterError
from flambe.cluster.cluster import Cluster
from flambe.cluster.instance import OrchestratorInstance, CPUFactoryInstance, GPUFactoryInstance
from flambe.logging import coloredlogs as cl


logger = logging.getLogger(__name__)


class GCPCluster(Cluster):

    def __init__(self,
                 name: str,
                 factory_type: str,
                 factories_num: int,
                 orchestrator_type: str,
                 ssh_key: str,
                 ssh_username: str,
                 service_account_email: str,
                 service_account_key: str,
                 project_id: str,
                 zone: str = 'us-central1-a',
                 factory_image: Optional[str] = None,
                 gpu_type: Optional[str] = None,
                 gpu_count: int = 1,
                 orchestrator_image: Optional[str] = None,
                 disk_type: str = 'pd-standard',
                 disk_size: int = 100,
                 setup_cmds: Optional[List[str]] = None) -> None:
        if setup_cmds is None:
            setup_cmds = []
        if factory_image is None or orchestrator_image is None:
            # we're using GCP's pytorch images which do not activate
            # its conda environment for non-login and non-interactive
            # shells like the ssh shell Paramiko gives us, so we need
            # to source the script correctly so that we get the correct
            # python version.
            # NOTE: we need to prepend b/c there's a check for skipping
            # non-interactive shells
            # we also need to add the local pip bin path and
            # install google_compute_engine otherwise commands are not
            # available and boto3 complains about a missing package
            setup_cmds = ["if ! grep '\\. /etc/profile.d/anaconda.sh' ~/.bashrc; "
                          "then sed -i '1s;^;. /etc/profile.d/anaconda.sh\\n;' ~/.bashrc; fi",
                          "if ! grep 'PATH=~/\\.local/bin:$PATH' ~/.bashrc; "
                          "then sed -i '1s;^;PATH=~/\\.local/bin:$PATH\\n;' ~/.bashrc; fi",
                          "python3 -m pip install --user --upgrade google_compute_engine",
                          "sudo apt-get install psmisc"]
        super().__init__(name, factories_num, ssh_key, ssh_username, setup_cmds)
        self.factory_type = factory_type
        self.orchestrator_type = orchestrator_type
        self.driver = get_driver(Provider.GCE)
        self.service_account_email = service_account_email
        self.service_account_key = service_account_key
        self.project_id = project_id
        self.zone = zone
        self.conn = self.driver(
            self.service_account_email,
            key=self.service_account_key,
            datacenter=self.zone,
            project=self.project_id
        )

        self.factory_metadata = None
        if factory_image is None:
            self.factory_image = self.conn.ex_get_image_from_family(
                'pytorch-1-1-cpu', ex_project_list=['deeplearning-platform-release'])
            self.factory_metadata = {'install-nvidia-driver': 'True'}
        else:
            self.factory_image = factory_image
        self.orchestrator_image = orchestrator_image
        if self.orchestrator_image is None:
            self.orchestrator_image = self.conn.ex_get_image_from_family(
                'pytorch-1-1-cpu', ex_project_list=['deeplearning-platform-release'])
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.disk_type = disk_type
        self.disk_size = disk_size

    def _create_cpu_factory(self, name: str) -> CPUFactoryInstance:
        node = self.conn.create_node(
            name,
            self.factory_type,
            self.factory_image,
            ex_disk_type=self.disk_type,
            ex_disk_size=self.disk_size,
            ex_tags='flambe-factory',
        )
        return CPUFactoryInstance(
            node.public_ips[0],
            node.private_ips[0],
            self.username,
            self.key,
            self.config,
            self.debug,
        )

    def _create_gpu_factory(self, name: str) -> GPUFactoryInstance:
        node = self.conn.create_node(
            name,
            self.factory_type,
            self.factory_image,
            ex_disk_type=self.disk_type,
            ex_disk_size=self.disk_size,
            ex_metadata=self.factory_metadata,
            ex_accelerator_type=self.gpu_type,
            ex_accelerator_count=self.gpu_count,
            ex_on_host_maintenance='TERMINATE',
            ex_automatic_restart=True,
            ex_tags='flambe-factory',
        )
        return GPUFactoryInstance(
            node.public_ips[0],
            node.private_ips[0],
            self.username,
            self.key,
            self.config,
            self.debug,
        )

    def _get_existing_instances(self) -> Tuple[
        Optional[OrchestratorInstance],
            List[Union[CPUFactoryInstance, GPUFactoryInstance]]]:
        nodes = self.conn.list_nodes()
        orchestrator = None
        factories: List[Union[CPUFactoryInstance, GPUFactoryInstance]] = []

        for node in nodes:
            if node.name == self.get_orchestrator_name():
                if orchestrator:
                    raise ClusterError(
                        f'Found more than one orchestrator with the same name: {node.name}')
                orchestrator = OrchestratorInstance(
                    node.public_ips[0],
                    node.private_ips[0],
                    self.username,
                    self.key,
                    self.config,
                    self.debug,
                )
            elif node.name.startswith(self.get_factory_basename()):
                factory = CPUFactoryInstance(
                    node.public_ips[0],
                    node.private_ips[0],
                    self.username,
                    self.key,
                    self.config,
                    self.debug,
                )
                if factory.contains_gpu():
                    factories.append(GPUFactoryInstance(
                        factory.host, factory.private_host, factory.username,
                        self.key, self.config, self.debug))
                else:
                    factories.append(factory)

        return orchestrator, factories

    def load_all_instances(self) -> None:
        orchestrator, factories = self._get_existing_instances()
        if orchestrator is not None and len(factories) == self.factories_num:
            logger.info(cl.GR('Found an existing cluster'))
            self.orchestrator = orchestrator
            self.factories = factories
            return

        # TODO: handle partially launched cluster

        with ThreadPoolExecutor() as executor:
            # launch the orchestrator
            logger.info(cl.GR("Launching the orchestrator"))
            future_orchestrator_node = executor.submit(
                self.conn.create_node,
                self.get_orchestrator_name(),
                self.orchestrator_type,
                self.orchestrator_image,
                ex_disk_type=self.disk_type,
                ex_disk_size=self.disk_size,
                ex_tags='flambe-orchestrator',
            )

            # launch factories
            if self.gpu_type is None:
                logger.info(cl.GR("Launching the CPU factories"))
                future_factories = executor.map(
                    self._create_cpu_factory,
                    [self.get_factory_basename() + f'-{i+1}' for i in range(self.factories_num)],
                )
            else:
                logger.info(cl.GR("Launching the GPU factories"))
                future_factories = executor.map(
                    self._create_gpu_factory,
                    [self.get_factory_basename() + f'-{i+1}' for i in range(self.factories_num)],
                )
            try:
                orchestrator_node = future_orchestrator_node.result()
                self.orchestrator = OrchestratorInstance(
                    orchestrator_node.public_ips[0],
                    orchestrator_node.private_ips[0],
                    self.username,
                    self.key,
                    self.config,
                    self.debug,
                )

                self.factories = [f for f in future_factories]

            except GoogleBaseError as e:
                raise ClusterError(f"Error creating nodes. Original error: {e}")

    def get_orchestrator_name(self) -> str:
        return f"{self.name}-orchestrator"

    def get_factory_basename(self) -> str:
        return f"{self.name}-factory"

    def rollback_env(self) -> None:
        return super().rollback_env()
