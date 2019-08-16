"""Implementation of a Cluster with AWS EC2 as the cloud provider

"""

import boto3
import botocore
import logging

from typing import Generator, Dict, Tuple, List, TypeVar, Type, Any, Optional

from flambe.cluster import instance, const
from flambe.cluster.cluster import Cluster, FactoryInsT
from flambe.cluster import errors
from flambe.logging import coloredlogs as cl

from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)

T = TypeVar("T", instance.OrchestratorInstance, instance.GPUFactoryInstance,
            instance.CPUFactoryInstance)


class AWSCluster(Cluster):
    """This Cluster implementation uses AWS EC2 as the cloud provider.

    This cluster works with AWS Instances that are defined in:
    `flambe.remote.instance.aws`

    Parameters
    ----------
    name: str
        The unique name for the cluster
    factories_num : int
        The amount of factories to use. This is not the amount of
        workers, as each factories can contain multiple GPUs and
        therefore, multiple workers.
    factories_type : str
        The type of instance to use for the Factory Instances.
        GPU instances are required for AWS the AWSCluster.
        "p2" and "p3" instances are recommended.
    factory_ami : str
        The AMI to be used for the Factory instances. Custom Flambe AMI
        are provided based on Ubuntu 18.04 distribution.
    orchestrator_type : str
        The type of instance to use for the Orchestrator Instances.
        This may not be a GPU instances. At least a "t2.small" instance
        is recommended.
    key_name: str
        The key name that will be used to connect into the instance.
    creator: str
        The creator should be a user identifier for the instances.
        This information will create a tag called 'creator' and it will
        also be used to retrieve existing hosts owned by the user.
    key: str
        The path to the ssh key used to communicate to all instances.
        IMPORTANT: all instances must be accessible with the same key.
    region_name: Optional[str]
        The region name to use. If not specified, it uses the locally
        configured region name or 'us-east-1' in case it's not
        configured.
    username: str
        The username of the instances the cluster will handle. Defaults
        to 'ubuntu'.
        IMPORTANT: for now all instances need to have the same username.
    tags: Dict[str, str]
        A dictionary with tags that will be added to all created hosts.
    security_group: str
        The security group to use to create the instances.
    subnet_id: str
        The subnet ID to use.
    orchestrator_ami : str
        The AMI to be used for the Factory instances. Custom Flambe
        AMI are provided based on Ubuntu 18.04 distribution.
    dedicated: bool
        Wether all created instances are dedicated instances or shared.
    orchestrator_timeout: int
        Number of consecutive hours before terminating the orchestrator
        once the experiment is over (either success of failure).
        Specify -1 to disable automatic shutdown (the orchestrator
        will stay on until manually terminated) and 0 to shutdown when
        the experiment is over. For example, if specifying 24, then the
        orchestrator will be shut down one day after the experiment is
        over. ATTENTION: This also applies when the experiment ends
        with an error. Default is -1.
    factories_timeout: int
        Number of consecutive hours to automatically terminate factories
        once the experiment is over (either success or failure).
        Specify -1 to disable automatic shutdown (the factories will
        stay on until manually terminated) and 0 to shutdown when the
        experiment is over. For example, if specifying 10, then the
        factories will be shut down 10 hours after the experiment is
        over. ATTENTION: This also applies when the experiment ends
        with an error. Default is 1.
    volume_size: int
        The disk size in GB that all hosts will contain. Defaults to
        100 GB.
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
                 factories_type: str,
                 orchestrator_type: str,
                 key_name: str,
                 security_group: str,
                 subnet_id: str,
                 creator: str,
                 key: str,
                 region_name: Optional[str] = None,
                 username: str = "ubuntu",
                 tags: Dict[str, str] = None,
                 orchestrator_ami: str = None,
                 factory_ami: str = None,
                 dedicated: bool = False,
                 orchestrator_timeout: int = -1,
                 factories_timeout: int = 1,
                 volume_size: int = 100,
                 setup_cmds: Optional[List[str]] = None) -> None:
        super().__init__(name, factories_num, key, username, setup_cmds)

        self.factories_type = factories_type
        self.orchestrator_type = orchestrator_type

        self.region_name = region_name
        self.sess = self._get_boto_session(region_name)

        self.ec2_resource = self.sess.resource('ec2')
        self.ec2_cli = self.sess.client('ec2')
        self.cloudwatch = self.sess.client('cloudwatch')

        self.factory_ami = factory_ami
        self.orchestrator_ami = orchestrator_ami

        self.key_name = key_name

        self.creator = creator
        self.tags = tags

        self.security_group = security_group
        self.subnet_id = subnet_id
        self.volume_size = volume_size

        self.dedicated = dedicated

        self.orchestrator_timeout = orchestrator_timeout
        self.factories_timeout = factories_timeout

        self.created_instances_ids: List[str] = []

    def _get_boto_session(self, region_name: Optional[str]) -> boto3.Session:
        """Get the boto3 Session from which the resources
        and clients will be created.

        This method is called by the contructor.

        Parameters
        ----------
        region_name: Optional[str]
            The region to use. If None, boto3 will resolve to the
            locally configured region_name or 'us-east-1' if not
            configured.

        Returns
        -------
        boto3.Session
            The boto3 Session to use

        """
        return boto3.Session(region_name=region_name)

    def load_all_instances(self) -> None:
        """Launch all instances for the experiment.

        This method launches both  the orchestrator and the factories.

        """
        boto_orchestrator, boto_factories = self._existing_cluster()

        with ThreadPoolExecutor() as executor:
            future_orch, future_factories = None, None

            if boto_orchestrator:
                self.orchestrator = self.get_orchestrator(boto_orchestrator.public_ip_address,
                                                          boto_orchestrator.private_ip_address)
                logger.info(cl.BL(
                    f"Found existing orchestrator ({boto_orchestrator.instance_type}) " +
                    f"{self.orchestrator.host}"
                ))

            else:
                future_orch = executor.submit(self._create_orchestrator)

            for f in boto_factories:
                factory = self.get_factory(f.public_ip_address, f.private_ip_address)
                if factory.contains_gpu():
                    factory = self.get_gpu_factory(f.public_ip_address, f.private_ip_address)
                self.factories.append(factory)

            if len(self.factories) > 0:
                logger.info(cl.BL(f"Found {len(self.factories)} existing factories " +
                                  f"({str([f.host for f in self.factories])})."))

            pending_new_factories = self.factories_num - len(self.factories)

            logger.debug(f"Creating {pending_new_factories} factories")
            if pending_new_factories > 0:
                future_factories = executor.submit(
                    self._create_factories,
                    number=pending_new_factories
                )
            elif pending_new_factories < 0:
                logger.info(cl.BL(f"Reusing existing {len(boto_factories)} factories."))

            try:
                if future_orch:
                    self.orchestrator = future_orch.result()
                    logger.info(cl.BL(f"New orchestrator created {self.orchestrator.host}"))

                if future_factories:
                    new_factories = future_factories.result()
                    self.factories.extend(new_factories)
                    logger.info(cl.BL(
                        f"{pending_new_factories} factories {self.factories_type} created " +
                        f"({str([f.host for f in new_factories])})."))
            except botocore.exceptions.ClientError as e:
                raise errors.ClusterError(
                    "Error creating the instances. Check that the provided configuration " +
                    f" is correct. Original error: {e}"
                )

        self.name_hosts()
        self.update_tags()
        self.remove_existing_events()
        self.create_cloudwatch_events()

    def _existing_cluster(self) -> Tuple[Any, List[Any]]:
        """Whether there is an existing cluster that matches name.

        The cluster should also match all other tags, including Creator)

        Returns
        -------
        Tuple[Any, List[Any]]
            Returns the (boto_orchestrator, [boto_factories])
            that match the experiment's name.

        """
        candidates: List[Tuple[Any, str]] = []
        for ins, role, cluster_name in self.flambe_own_running_instances():
            if role and cluster_name:
                if cluster_name == self.name:
                    candidates.append((ins, role))
                    logger.debug(f"Found existing {role} host {ins.public_ip_address}")

        orchestrator = None
        factories = []

        for ins, role in candidates:
            if role == 'Orchestrator':
                if orchestrator:
                    raise errors.ClusterError(
                        "Found 2 Orchestrator instances with same experiment name. " +
                        "This should never happen. " +
                        "Please remove manually all instances with tag " +
                        f"'Cluster-Name': '{self.name}' and retry."
                    )

                orchestrator = ins
            elif role == 'Factory':
                factories.append(ins)

        return orchestrator, factories

    def _get_tags(self, boto_instance: "boto3.resources.factory.ec2.Instance") -> Dict[str, str]:
        """Gets the tags of a EC2 instances

        Parameters
        ----------
        boto_instance : BotoIns
            The EC2 instance to access the tags.

        Returns
        -------
        Dict[str, str]
            Key, Value for the specified tags.

        """
        ret = {}
        if boto_instance.tags:
            for t in boto_instance.tags:
                ret[t['Key']] = t['Value']

        return ret

    def flambe_own_running_instances(
            self
    ) -> Generator[Tuple['boto3.resources.factory.ec2.Instance',
                         Optional[str], Optional[str]], None, None]:
        """Get running instances with matching tags.

        Yields
        -------
        Tuple['boto3.resources.factory.ec2.Instance', str]
            A tuple with the instance and the name of the EC2 instance.

        """
        boto_instances = self.ec2_resource.instances.filter(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])

        for ins in boto_instances:
            tags = self._get_tags(ins)
            if all((
                    "creator" in tags and tags["creator"] == self.creator,
                    "Purpose" in tags and tags["Purpose"] == "flambe",
            )):
                yield ins, tags.get("Role"), tags.get("Cluster-Name")

    def name_hosts(self) -> None:
        """Name the orchestrator and factories.

        """
        if not self.orchestrator:
            raise errors.ClusterError("Orchestrator instance was not loaded.")

        self.name_instance(self._get_boto_instance_by_host(self.orchestrator.host),
                           self.get_orchestrator_name())

        for i, f in enumerate(self.factories):
            self.name_instance(self._get_boto_instance_by_host(f.host),
                               f"{self.get_factory_basename()}_{i+1}")

    def update_tags(self) -> None:
        """Update user provided tags to all hosts.

        In case there is an existing cluster that do not contain all the
        tags, by executing this all hosts will have the user specified
        tags.

        This won't remove existing tags in the hosts.

        """
        if not self.orchestrator:
            raise errors.ClusterError("Orchestrator instance was not loaded.")

        if self.tags:
            self._update_tags(self._get_boto_instance_by_host(self.orchestrator.host),
                              self.tags)

            for i, f in enumerate(self.factories):
                self._update_tags(self._get_boto_instance_by_host(f.host),
                                  self.tags)

    def _update_tags(
            self,
            boto_instance: 'boto3.resources.factory.ec2.Instance',
            tags: Dict[str, str]) -> None:
        """Create/Overwrite tags on an EC2 instance

        Parameters
        ----------
        boto_instance : 'boto3.resources.factory.ec2.Instance'
            The EC2 instance
        tags : Dict[str, str]
            The tags to create/overwrite

        """
        ins_tags = [{"Key": k, "Value": v} for k, v in tags.items()]
        boto_instance.create_tags(Tags=ins_tags)

    def name_instance(
            self,
            boto_instance: 'boto3.resources.factory.ec2.Instance',
            name: str) -> None:
        """Renames a EC2 instance

        Parameters
        ----------
        boto_instance : 'boto3.resources.factory.ec2.Instance'
            The EC2 instance
        name : str
            The new name

        """
        boto_instance.create_tags(Tags=[{"Key": "Name", "Value": "{}".format(name)}])

    def _create_orchestrator(self) -> instance.OrchestratorInstance:
        """Create a new EC2 instance to be the Orchestrator instance.

        This new machine receives all tags defined in the *.ini file.

        Returns
        -------
        instance.AWSOrchestratorInstance
            The new orchestrator instance.

        """
        if not self.orchestrator_ami:
            ami = self._find_default_ami(_type="orchestrator")
            if ami is None:
                raise errors.ClusterError("Could not find matching AMI for the orchestrator.")
        else:
            ami = self.orchestrator_ami

        return self._generic_launch_instances(instance.OrchestratorInstance,
                                              1, self.orchestrator_type,
                                              ami, role="Orchestrator")[0]

    def _create_factories(self,
                          number: int = 1) -> List[FactoryInsT]:
        """Creates new AWS EC2 instances to be the Factory instances.

        These new machines receive all tags defined in the *.ini file.
        Factory instances will be named using the factory basename plus
        an index. For example, "seq2seq_factory_0", "seq2seq_factory_1".

        Parameters
        ----------
        number : int
            The number of factories to be created.

        Returns
        -------
        List[instance.AWSGPUFactoryInstance]
            The new factory instances.

        """
        if not self.factory_ami:
            ami = self._find_default_ami(_type="factory")
            if ami is None:
                raise errors.ClusterError("Could not find matching AMI for the factory.")
        else:
            ami = self.factory_ami

        factories = self._generic_launch_instances(instance.CPUFactoryInstance,
                                                   number, self.factories_type,
                                                   ami, role="Factory")

        for i, f in enumerate(factories):
            f.wait_until_accessible()
            if f.contains_gpu():
                factories[i] = instance.GPUFactoryInstance(f.host, f.private_host, f.username,
                                                           self.key, self.config, self.debug)

        return factories

    def _generic_launch_instances(
            self, instance_class: Type[T], number: int,
            instance_type: str, instance_ami: str, role: str
    ) -> List[T]:
        """Generic method to launch instances in AWS EC2 using boto3.

        This method should not be used outside this module.

        Parameters
        ----------
        instance_class: Type[T]
            The instance class.
            It can be AWSOrchestratorInstance or AWSGPUFactoryInstance.
        number : int
            The amount of instances to create
        instance_type : str
            The instance type
        instance_ami : str
            The AMI to be used. Should be an Ubuntu 18.04 based AMI.
        role: str
            Wether is 'Orchestrator' or 'Factory'

        Returns
        -------
        List[Union[AWSOrchestratorInstance, AWSGPUFactoryInstance]]
            The new Instances.

        """
        # Set the tags based on the users + custom flambe tags.
        tags: List[Dict[str, str]] = []
        if self.tags:
            tags.extend([{'Key': k, 'Value': v} for k, v in self.tags.items()])

        tags.append({'Key': 'creator', 'Value': self.creator})
        tags.append({'Key': 'Purpose', 'Value': 'flambe'})
        tags.append({'Key': 'Cluster-Name', 'Value': self.name})
        tags.append({'Key': 'Role', 'Value': role})

        bdm = [
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': self.volume_size,
                    'DeleteOnTermination': True,
                    'VolumeType': 'io1',
                    'Iops': 50 * self.volume_size
                }
            }
        ]
        tags_param = [
            {
                'ResourceType': 'instance',
                'Tags': tags
            },
        ]
        placement = {
            'Tenancy': 'dedicated' if self.dedicated else 'default',
        }
        # IMPORTANT: when using dedicated Instances
        # it should be a supported instance type (for example,
        # p2 series are NOT supported).
        # For a list of supported instance types go to:
        # https://aws.amazon.com/ec2/purchasing-options/dedicated-instances/

        boto_instances = self.ec2_resource.create_instances(
            ImageId=instance_ami,
            InstanceType=instance_type,
            KeyName=self.key_name,
            SecurityGroupIds=[self.security_group],
            MaxCount=number, MinCount=1, SubnetId=self.subnet_id,
            BlockDeviceMappings=bdm, TagSpecifications=tags_param, Placement=placement,
            EbsOptimized=True)

        self.created_instances_ids.extend(ins.id for ins in boto_instances)
        logger.debug(f"Created {len(boto_instances)} {instance_type}")

        # Blocks until all instances are running.
        for idx, ins in enumerate(boto_instances):
            ins.wait_until_running()

        logger.debug(f"Created instances running")

        ret = []
        for idx, ins in enumerate(boto_instances):
            ins.reload()  # Update instance information now that is running

            ret.append(instance_class(ins.public_ip_address, ins.private_ip_address,
                                      self.username, self.key, self.config, debug=self.debug))

        if len(boto_instances) < number:
            logger.debug(f"Less {instance_type} instances were created. "
                         f"{len(boto_instances)} out of {number}")

        return ret

    def terminate_instances(self) -> None:
        """Terminates all instances.

        """
        boto_instances = self.ec2_resource.instances.filter(
            Filters=[{
                'Name': 'instance-id',
                'Values': self.created_instances_ids
            }]
        )

        for boto_ins in boto_instances:
            boto_ins.terminate()
            logger.info(cl.RE(f"Terminating {boto_ins.id}"))

    def rollback_env(self) -> None:
        """Rollback the environment.

        This occurs when an error is caucht during the local stage of
        the remote experiment (i.e. creating the cluster, sending the
        data and submitting jobs), this method handles cleanup stages.

        """
        # If no factories are created (because of quota exceeded)
        # but orchestrator was created, terminate it.
        if self.orchestrator is not None and len(self.factories) == 0:
            self.terminate_instances()

        # If factories are created but no orchestrator was.
        if len(self.factories) > 0 and self.orchestrator is None:
            self.terminate_instances()

    def parse(self) -> None:
        """Checks if the AWSCluster configuration is valid.

        This checks that the factories are never terminated after
        the orchestrator is. Avoids the scenario where the cluster has
        only factories and no orchestrator, which is useless.

        Raises
        ------
        errors.ClusterConfigurationError
            If configuration is not valid.

        """
        if self.orchestrator_timeout > -1:
            if (
                self.factories_timeout == -1 or
                self.factories_timeout > self.orchestrator_timeout
            ):
                raise errors.ClusterConfigurationError(
                    "Factories can't be terminated after the orchestrator is terminated"
                )

        if self.tags and "creator" in [x.lower() for x in self.tags.keys()]:
            raise errors.ClusterConfigurationError(
                "AWS Cluster tags can't include a 'creator' tag. " +
                "The 'creator' attribute declared in the object will be used."
            )

        if self.tags and "name" in [x.lower() for x in self.tags.keys()]:
            raise errors.ClusterConfigurationError(
                "AWS Cluster tags can't include a 'name' tag. " +
                "The 'name' attribute declared in the object will be used."
            )

    def _get_boto_instance_by_host(
            self,
            public_host: str) -> Optional["boto3.resources.factory.ec2.Instance"]:
        """Returns the instance id given the public host

        Parameters
        ----------
        public_host: str
            The host in IP format of DNS format

        Returns
        -------
        Optional[boto3.resources.factory.ec2.Instance]
            The id if found else None

        """
        boto_instances = self.ec2_resource.instances.all()

        for ins in boto_instances:
            if ins.public_dns_name == public_host or ins.public_ip_address == public_host:
                return ins

        return None

    def _get_instance_id_by_host(self, public_host: str) -> Optional[str]:
        """Returns the instance id given the public host

        Parameters
        ----------
        public_host: str
            The host in IP format of DNS format

        Returns
        -------
        Optional[str]
            The id if found else None

        """
        ins = self._get_boto_instance_by_host(public_host)
        return ins.id if ins else None

    def _get_alarm_name(self, instance_id: str) -> str:
        """Get the alarm name to be used for the given instance.

        Parameters
        ----------
        instance_id: str
            The id of the instance

        Returns
        -------
        str
            The name of the corresponding alarm

        """
        return f"Flambe_Instance_Terminate_CPU_Utilization_{instance_id}"

    def has_alarm(self, instance_id: str) -> bool:
        """Whether the instance has an alarm set.

        Parameters
        ----------
        instance_id: str
            The id of the instance

        Returns
        -------
        bool
            True if an alarm is set. False otherwise.

        """
        try:
            ret = self.cloudwatch.describe_alarms(AlarmNames=[self._get_alarm_name(instance_id)])
            return len(ret['MetricAlarms']) > 0
        except botocore.exceptions.ParamValidationError:
            raise errors.ClusterError(f"Could not retrieve alarm for {instance_id}")

    def remove_existing_events(self) -> None:
        """Remove the current alarm.

        In case the orchestrator or factories had an alarm,
        we remove it to reset the new policies.

        """
        if not self.orchestrator:
            raise errors.ClusterError("Orchestrator instance was not loaded.")

        orch_host = self.orchestrator.host
        orch_id = self._get_instance_id_by_host(orch_host)
        if orch_id:
            self._delete_cloudwatch_event(orch_id)

        for f in self.factories:
            f_id = self._get_instance_id_by_host(f.host)
            if f_id:
                self._delete_cloudwatch_event(f_id)

    def create_cloudwatch_events(self) -> None:
        """Creates cloudwatch events for orchestrator and factories.

        """
        if not self.orchestrator:
            raise errors.ClusterError("Orchestrator instance was not loaded.")

        fact_t = self.factories_timeout
        # Create events for factories to shut down
        if fact_t >= 0:
            for f in self.factories:
                f_id = self._get_instance_id_by_host(f.host)
                if f_id:
                    mins = fact_t * 60 if fact_t > 0 else 5
                    self._create_cloudwatch_event(f_id, mins=mins, cpu_thresh=0.5)
                    logger.info(cl.YE(f"{f.host} timeout of {mins} mins set"))
        else:
            logger.info(cl.YE(f"Factories have no timeout"))

        orch_host = self.orchestrator.host
        orch_id = self._get_instance_id_by_host(orch_host)
        if orch_id:
            orch_t = self.orchestrator_timeout
            if orch_t >= 0:
                mins = orch_t * 60 if orch_t > 0 else 5
                self._create_cloudwatch_event(orch_id, mins=mins, cpu_thresh=4)
                logger.info(cl.YE(f"{self.orchestrator.host} timeout of {mins} set"))
            else:
                logger.info(cl.YE(f"Orchestrator {self.orchestrator.host} has no timeout"))

    def _delete_cloudwatch_event(self, instance_id: str) -> None:
        """Deletes the alarm related to the instance.

        """
        try:
            self.cloudwatch.delete_alarms(
                AlarmNames=[self._get_alarm_name(instance_id)]
            )
            logger.debug(f"Removed existing alarm for id {instance_id}")

        except botocore.exceptions.ParamValidationError:
            raise errors.ClusterError(f"Could not delete alarm for {instance_id}")

    def _create_cloudwatch_event(self, instance_id: str, mins: int = 60,
                                 cpu_thresh: float = 0.1) -> None:
        """Create CloudWatch alarm.

        The alrm is used to terminate an instance based on CPU usage.

        Parameters
        ----------
        instance_id: str
            The ID of the EC2 instance
        mins: int
            Number of minutes to trigger the termination event.
            The evaluation preriod will be always one minute.
        cpu_thresh: float
            Percentage specifying upper bound for triggering event.
            If mins is 60 and cpu_thresh is 0.1, then this instance
            will be deleted after 1 hour of average CPU below 0.1.

        """

        # Create alarm with actions enabled
        try:
            self.cloudwatch.put_metric_alarm(
                AlarmName=self._get_alarm_name(instance_id),
                ComparisonOperator='LessThanThreshold',
                EvaluationPeriods=mins,
                MetricName='CPUUtilization',
                Namespace='AWS/EC2',
                Period=60,
                Statistic='Average',
                Threshold=cpu_thresh,
                ActionsEnabled=True,
                AlarmActions=[
                    f'arn:aws:automate:{self.region_name}:ec2:terminate'
                ],
                AlarmDescription=f'Terminate when CPU < {cpu_thresh}%',
                Dimensions=[
                    {
                        'Name': 'InstanceId',
                        'Value': instance_id
                    },
                ],
                Unit='Percent'
            )
            logger.debug(f"Created alarm for id {instance_id}")
        except botocore.exceptions.ParamValidationError:
            raise errors.ClusterError(f"Could not setup cloudwatch for {instance_id}")

    def _get_images(self) -> Dict:
        """Get the official AWS public AMIs created by Flambe.

        ATTENTION: why not just search the tags? We need to make sure
        the AMIs we pick were created by the Flambe team. Because of
        tags values not being unique, anyone can create a public AMI
        with 'Creator: flambe@asapp.com' as a tag.
        If we pick that AMI, then we could potentially be Creating
        instances with unknown AMIs, causing potential security issues.
        By filtering by our acount id (which can be public), then we can
        make sure that all AMIs that are being scanned were created
        by Flambe team.

        Returns
        -------
        Dict:
            The boto3 API response

        """
        return self.ec2_cli.describe_images(Owners=[const.AWS_FLAMBE_ACCOUNT])

    def _get_ami(self, _type: str, version: str) -> Optional[str]:
        """Given a type and a version, get the correct Flambe AMI.


        IMPORTANT: we keep the version logic in case we add versioned
        AMIs in the future.

        Parameters
        ----------
        _type: str
            It can be either 'factory' or 'orchestrator'.
            Note that the type is lowercase in the AMI tag.
        version: str
            For example, "0.2.1" or "2.0".

        Returns
        -------
        The ImageId if it's found. None if not.

        """
        images = self._get_images()
        for i in images['Images']:
            # Name is for example 'flambe-orchestrator 0.0.0'
            if i['Name'] == f"flambe-{_type.lower()}-ami {version}":
                return i['ImageId']

        return None

    def _find_default_ami(self, _type: str) -> Optional[str]:
        """Returns an AMI with version 0.0.0, which is the default.
        This means that doesn't contain flambe itself but it has
        some heavy dependencies already installed (like pytorch).

        Parameters
        ----------
        _type: str
            Wether is "orchestrator" or "factory"

        Returns
        -------
        Optional[str]
            The ImageId or None if not found.

        """
        return self._get_ami(_type, '0.0.0')
