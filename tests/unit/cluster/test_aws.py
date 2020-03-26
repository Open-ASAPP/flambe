from flambe.cluster import AWSCluster

from moto import mock_ec2, mock_cloudwatch

import pytest
import boto3
import tempfile
import shutil
import sys
from io import StringIO
import importlib
import copy

import mock
from flambe.runnable import SafeExecutionContext, error

from flambe.cluster import instance, errors

# This fixture will run for each test method automatically
@pytest.fixture(scope='function', autouse=True)
def ec2_mock():
    mock = mock_ec2()
    mock_cw = mock_cloudwatch()

    mock.start()
    mock_cw.start()

    yield

    mock.stop()
    mock_cw.stop()

# Create a mock subnet_id
@pytest.fixture
def subnet():
    ec2 = boto3.resource('ec2')
    vpc = ec2.create_vpc(CidrBlock='192.168.0.0/16')
    yield ec2.create_subnet(CidrBlock='192.168.0.0/16', VpcId=vpc.id)

@pytest.fixture
def sec_group(subnet):
    ec2 = boto3.resource('ec2')
    return ec2.create_security_group(GroupName='slice_0', Description='slice_0 sec group', VpcId=subnet.vpc.id)


@pytest.fixture
def get_secrets():
    t = tempfile.NamedTemporaryFile(mode="w+", delete=False)

    def _get_secrets(secrets_content=None):
        secrets_content = secrets_content or """
            [SSH]
            SSH_KEY = /path/to/key
            USER = ubuntu
        """
        t.write(secrets_content)
        t.flush()
        return t.name

    yield _get_secrets
    t.close()


@pytest.fixture
def get_cluster(subnet, sec_group):

    def _get_cluster(**kwargs):
        return AWSCluster(
           name=kwargs.get('name', 'some-name'),
           factories_num=kwargs.get('factories_num', 2),
           factories_type=kwargs.get('factories_type', 'g3.4xlarge'),
           orchestrator_type=kwargs.get('orchestrator_type', 't3.large'),
           key_name=kwargs.get('key_name', 'some-key'),
           security_group=sec_group.id,
           tags=kwargs.get('tags'),
           orchestrator_ami=kwargs.get('orchestrator_ami', 'ami-di32icco45kbis9bk'),
           factory_ami=kwargs.get('factory_ami', 'ami-di32icco45kbis9bk'),
           username=kwargs.get("username", "ubuntu"),
           key=kwargs.get("key", "/path/to/key"),
           subnet_id=subnet.id,
           creator=kwargs.get('creator', 'leo.messi')
       )

    return _get_cluster


@pytest.fixture
def create_cluster(get_cluster):
    def _create_cluster(**kwargs):
        cluster = get_cluster(**kwargs)
        cluster.load_all_instances()
        return cluster

    return _create_cluster


@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
def test_missing_secrets(get_cluster):
    cluster = get_cluster()
    cluster.load_all_instances()


def test_secrets(get_cluster, get_secrets):
    cluster = get_cluster()

    secrets = """
        [SOME_SECTION]
        RANDOM = random
    """

    cluster.inject_secrets(get_secrets(secrets))
    assert cluster.config['SOME_SECTION']['RANDOM'] == 'random'


def test_launch_orchestrator(subnet, sec_group, get_cluster):
    ec2 = boto3.resource('ec2')

    cluster = get_cluster(
        name='cluster_name',
        key_name='key_name',
        orchestrator_type='t3.xlarge',
        creator='leo.messi',
        volume_size=500
    )

    cluster._create_orchestrator()

    instances = list(ec2.instances.all())

    assert len(instances) == 1

    orch = instances[0]

    assert orch.instance_type == 't3.xlarge'
    assert orch.subnet == subnet
    assert orch.key_name == 'key_name'
    assert len(orch.security_groups) == 1
    assert orch.security_groups[0]['GroupId'] == sec_group.id

    assert len(orch.tags) == 5

    keys = ['creator', 'Purpose', 'Cluster-Name', 'Role', 'Name']
    for x in orch.tags:
        k, v = x['Key'], x['Value']

        assert k in keys

        if k == 'creator':
            assert v == 'leo.messi'
        if k == 'Cluster-Name':
            assert v == 'cluster_name'
        if k == 'Role':
            assert v == 'Orchestrator'
        if k == 'Purpose':
            assert v == 'flambe'
        if k == 'Name':
            assert v == cluster._get_creation_name('Orchestrator')

    assert len(orch.block_device_mappings) == 1
    assert orch.block_device_mappings[0]['DeviceName'] == '/dev/sda1'

    # moto seems not able to mock volume size
    # volume_id = orch.block_device_mappings[0]['Ebs']['VolumeId']
    # volume = list(orch.volumes.filter(VolumeIds=[volume_id]))[0]
    # assert volume.size == 500


def test_existing_orch(get_cluster):
    ec2 = boto3.resource('ec2')

    cluster = get_cluster()

    cluster._create_orchestrator()

    instances = list(ec2.instances.all())

    assert len(instances) == 1

    orch = instances[0]
    instance_id = orch.id

    cluster2 = get_cluster()

    orch2, _ = cluster2._existing_cluster()

    assert orch2.id == instance_id


def test_existing_orch2(get_cluster):
    ec2 = boto3.resource('ec2')

    # Existing clusters are evaluated based on creator
    # and name
    cluster = get_cluster(
        name='cluster_name',
        key_name='some_other_key',
        orchestrator_type='t3.medium',
        creator='leo.messi',
        volume_size=500
    )

    cluster._create_orchestrator()

    instances = list(ec2.instances.all())

    assert len(instances) == 1

    orch = instances[0]
    instance_id = orch.id


    cluster2 = get_cluster(
        name='cluster_name',
        key_name='some_key', # Another key
        orchestrator_type='t3.xlarge', # Another type
        creator='leo.messi',
    )

    orch2, _ = cluster2._existing_cluster()

    assert orch2.id == instance_id


def test_existing_multiple_orchs(get_cluster):
    ec2 = boto3.resource('ec2')

    # Existing clusters are evaluated based on creator
    # and name
    for _ in range(3):
        cluster = get_cluster(name='cluster_name', creator='leo.messi')

        cluster._create_orchestrator()

    instances = list(ec2.instances.all())
    assert len(instances) == 3

    with pytest.raises(errors.ClusterError):
        orch2, _ = cluster._existing_cluster()


@mock.patch('flambe.cluster.instance.instance.CPUFactoryInstance.contains_gpu')
@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
def test_launch_factories(mock_wait, mock_contains_gpu, subnet, sec_group, get_cluster):
    mock_contains_gpu.return_value = True

    ec2 = boto3.resource('ec2')

    cluster = get_cluster(
        name='cluster_name',
        factories_num=5,
        factories_type='g3.4xlarge',
        creator='leo.messi',
        volume_size=500
    )

    cluster._create_factories(number=5)

    instances = list(ec2.instances.all())

    assert 0 < len(instances) <= 5
    for f in instances:
        assert f.instance_type == 'g3.4xlarge'

        assert f.subnet == subnet
        assert f.key_name == 'some-key'
        assert len(f.security_groups) == 1
        assert f.security_groups[0]['GroupId'] == sec_group.id

        assert len(f.tags) == 5

        keys = ['creator', 'Purpose', 'Cluster-Name', 'Role', 'Name']
        for x in f.tags:
            k, v = x['Key'], x['Value']

            assert k in keys

            if k == 'creator':
                assert v == 'leo.messi'
            if k == 'Cluster-Name':
                assert v == 'cluster_name'
            if k == 'Role':
                assert v == 'Factory'
            if k == 'Purpose':
                assert v == 'flambe'
            if k == 'Name':
                assert v == cluster._get_creation_name('Factory')

        assert len(f.block_device_mappings) == 1
        assert f.block_device_mappings[0]['DeviceName'] == '/dev/sda1'

        # moto seems not able to mock volume size
        # volume_id = f.block_device_mappings[0]['Ebs']['VolumeId']
        # volume = list(f.volumes.filter(VolumeIds=[volume_id]))[0]
        # assert volume.size == 500


@mock.patch('flambe.cluster.instance.instance.CPUFactoryInstance.contains_gpu')
@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
def test_launch_gpu_factories(mock_wait, mock_contains_gpu, create_cluster):
    mock_contains_gpu.return_value = True

    ec2 = boto3.resource('ec2')
    cluster = create_cluster()

    instances = list(ec2.instances.all())
    assert len(instances) == 2

    assert cluster.orchestrator.__class__ == instance.OrchestratorInstance
    assert len(cluster.factories) == 1
    assert cluster.factories[0].__class__ == instance.GPUFactoryInstance


@mock.patch('flambe.cluster.instance.instance.CPUFactoryInstance.contains_gpu')
@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
def test_launch_cpu_factories(mock_wait, mock_contains_gpu, create_cluster):
    mock_contains_gpu.return_value = False

    ec2 = boto3.resource('ec2')
    cluster = create_cluster()

    instances = list(ec2.instances.all())
    assert len(instances) == 2

    assert cluster.orchestrator.__class__ == instance.OrchestratorInstance
    assert len(cluster.factories) == 1
    assert cluster.factories[0].__class__ == instance.CPUFactoryInstance


@mock.patch('flambe.cluster.instance.instance.CPUFactoryInstance.contains_gpu')
@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
@mock.patch('flambe.cluster.aws.AWSCluster._get_boto_private_host')
@mock.patch('flambe.cluster.aws.AWSCluster._get_boto_public_host')
def test_get_host_abstractions(mock_public_host, mock_private_host,
                               mock_wait, mock_contains_gpu, create_cluster):
    """Test that _get_boto_[public|private]_host are being used instead
    of the boto attributes.

    """
    mock_contains_gpu.return_value = False

    # Check that the methods are called when creating an instance.
    cluster = create_cluster()
    mock_public_host.assert_called()
    mock_private_host.assert_called()

    mock_public_host.reset_mock()
    mock_private_host.reset_mock()

    # Check methods are called when it finds an existing cluster.
    cluster2 = create_cluster()

    boto_orchestrator, boto_factories = cluster2._existing_cluster()

    assert boto_orchestrator is not None
    assert len(boto_factories) == 2

    mock_public_host.assert_called()
    mock_private_host.assert_called()

    mock_public_host.reset_mock()
    mock_private_host.reset_mock()

    # Check behavior when getting an instance by host
    _ = cluster._get_boto_instance_by_host(cluster.orchestrator.host)
    mock_public_host.assert_called()
    mock_private_host.assert_not_called()  # The private host is not required in this case.


@mock.patch('flambe.cluster.instance.instance.CPUFactoryInstance.contains_gpu')
@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
def test_existing_factories(mock_wait, mock_contains_gpu, get_cluster):
    mock_contains_gpu.return_value = False
    ec2 = boto3.resource('ec2')

    cluster = get_cluster(factories_num=1)

    cluster._create_factories()

    instances = list(ec2.instances.all())
    assert len(instances) == 1
    instance_id = instances[0].id

    cluster2 = get_cluster(factories_num=1)

    _, factories = cluster2._existing_cluster()

    assert len(factories) == 1

    assert factories[0].id == instance_id


@mock.patch('flambe.cluster.instance.instance.CPUFactoryInstance.contains_gpu')
@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
def test_tags(mock_wait, mock_contains_gpu, create_cluster):
    mock_contains_gpu.return_value = True
    ec2 = boto3.resource('ec2')

    cluster = create_cluster(
        tags={
            'random_tag': 'random_value',
            'another_random_tag': 'another_random_value'
        }
    )

    instances = list(ec2.instances.all())

    keys = ['creator', 'Purpose', 'Cluster-Name', 'Role']
    for i in instances:
        tags = {x['Key']: x['Value'] for x in i.tags}

        assert 'random_tag' in tags.keys()
        assert tags['random_tag'] == 'random_value'

        assert 'another_random_tag' in tags.keys()
        assert tags['another_random_tag'] == 'another_random_value'


@mock.patch('flambe.cluster.instance.instance.CPUFactoryInstance.contains_gpu')
@mock.patch('flambe.cluster.instance.instance.Instance.wait_until_accessible')
def test_instances_lifecycle(mock_wait, mock_contains_gpu, create_cluster):
    mock_contains_gpu.return_value = True
    ec2 = boto3.resource('ec2')

    cluster = create_cluster()

    instances = list(ec2.instances.all())

    for i in instances:
        assert i.state['Name'] == 'running'

    cluster.terminate_instances()
    instances = list(ec2.instances.all())  # Reload instances
    for i in instances:
        assert i.state['Name'] == 'terminated'


def test_get_creation_name(get_cluster):
    cluster = get_cluster(name='my-cluster')

    assert cluster._get_creation_name('Orchestrator') == 'my-cluster_orchestrator'
    assert cluster._get_creation_name('Factory') == 'my-cluster_factory'


@pytest.mark.parametrize('invalid_role', ['', 'orch', 'orchestrator', 'factory', 'Factories'])
def test_get_creation_name_invalid_role(invalid_role, get_cluster):
    cluster = get_cluster(name='my-cluster')

    with pytest.raises(ValueError):
        cluster._get_creation_name(invalid_role)
