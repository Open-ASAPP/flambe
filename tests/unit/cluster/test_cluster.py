import pytest
import configparser

import mock
from flambe.runnable import ClusterRunnable

from flambe.cluster import instance, Cluster
from flambe.cluster import errors as man_errors


class DummyRunnable(ClusterRunnable):
    def __init__(self, name='random', env=None):
        super().__init__(env)
        self.name = name

    def run(self):
        pass

    def setup(self):
        pass

    @classmethod
    def to_yaml(cls, representer, node, tag):
        kwargs = {}
        return representer.represent_mapping(tag, kwargs)

    @classmethod
    def from_yaml(cls, constructor, node, factory_name):
        kwargs, = list(constructor.construct_yaml_map(node))
        return cls(**kwargs)


def get_instance(_class=None, **kwargs):
    host = kwargs.get('host', '1.2.3.4')
    private_host = kwargs.get('private_host', '10.0.3.4')
    username = kwargs.get('username', 'ubuntu')
    key = kwargs.get('key', '/path/to/ssh/key')
    config = kwargs.get('config', configparser.ConfigParser())
    debug = kwargs.get('debug', False)
    use_public = kwargs.get('use_public', True)
    if _class:
        return _class(host=host, private_host=private_host, username=username,
                      key=key, config=config, debug=debug, use_public=use_public)

    return instance.Instance(host=host, private_host=private_host, username=username,
                             key=key, config=config, debug=debug, use_public=use_public)


@pytest.fixture
def get_cluster():

    def _get_cluster(gpu_cluster: bool = True, fill: bool = True, **kwargs):
        name = kwargs.get('name', 'cluster')
        factories_num = kwargs.get('factories_num', 2)
        username = kwargs.get("username", "ubuntu")
        key = kwargs.get("key", "/path/to/key")
        setup_cmds = kwargs.get("setup_cmds", [])

        cluster = Cluster(name=name, factories_num=factories_num, username=username,
                          key=key, setup_cmds=setup_cmds)

        if fill:
            cluster.orchestrator = get_instance(instance.OrchestratorInstance)
            if gpu_cluster:
                cluster.factories = [get_instance(instance.GPUFactoryInstance) for _ in range(cluster.factories_num)]
            else:
                cluster.factories = [get_instance(instance.CPUFactoryInstance) for _ in range(cluster.factories_num)]

        return cluster

    return _get_cluster


def test_empty_cluster(get_cluster):
    c = get_cluster(fill=False)
    with pytest.raises(man_errors.ClusterError):
        c.get_orch_home_path()


def test_empty_cluster_2(get_cluster):
    c = get_cluster(fill=False)
    with pytest.raises(man_errors.ClusterError):
        c.execute(DummyRunnable(), {}, "", False)


@mock.patch('flambe.cluster.instance.instance.Instance.get_home_path')
def test_orch_home_path(mock_get_home_path, get_cluster):
    mock_get_home_path.return_value = "/path/to/home"
    c = get_cluster()

    assert c.get_orch_home_path() == "/path/to/home"


@mock.patch('flambe.cluster.instance.instance.Instance.send_rsync')
@mock.patch('flambe.cluster.instance.instance.Instance.get_home_path')
def test_send_secrets(mock_get_home_path, mock_rsync, get_cluster):
    mock_get_home_path.return_value = "/path/to/home"

    c = get_cluster()
    c.send_secrets()
    args, kwargs = mock_rsync.call_args

    assert "/path/to/home/secret.ini" == args[-1]


@mock.patch('flambe.cluster.instance.instance.OrchestratorInstance.rsync_folder')
@mock.patch('flambe.cluster.instance.instance.Instance.get_home_path')
def test_rsync_orch(mock_get_home_path, mock_rsync, get_cluster):
    mock_get_home_path.return_value = "/path/to/home"

    c = get_cluster()
    c.rsync_orch("/some/folder")
    assert mock_rsync.call_count == c.factories_num

    mock_rsync.assert_called_with("/some/folder", f'{c.username}@{c.factories[-1].private_host}:/some/folder')


@mock.patch('flambe.cluster.instance.instance.OrchestratorInstance.launch_flambe')
@mock.patch('flambe.cluster.instance.instance.OrchestratorInstance.send_rsync')
@mock.patch('flambe.cluster.instance.instance.Instance.get_home_path')
def test_execute(mock_get_home_path, mock_rsync, mock_launch, get_cluster):
    mock_get_home_path.return_value = "/path/to/home"
    c = get_cluster()

    c.execute(DummyRunnable(), {}, "", False)

    args, kwargs = mock_rsync.call_args
    assert "/path/to/home/flambe.yaml" == args[-1]

    mock_launch.assert_called_once()
    args, kwargs = mock_launch.call_args

    assert "/path/to/home/flambe.yaml" == args[0]


def test_get_max_resources(get_cluster):
    c = get_cluster(factories_num=2)

    c.factories[0].num_cpus = mock.Mock(return_value=100)
    c.factories[0].num_gpus = mock.Mock(return_value=10)

    c.factories[-1].num_cpus = mock.Mock(return_value=10)
    c.factories[-1].num_gpus = mock.Mock(return_value=100)

    resources = c.get_max_resources()

    assert resources['cpu'] == 10
    assert resources['gpu'] == 10


@mock.patch('flambe.cluster.instance.instance.Instance.get_home_path')
def test_get_remote_env(mock_get_home_path, get_cluster):
    mock_get_home_path.return_value = "/path/to/home"
    c = get_cluster()

    env = c.get_remote_env(user_provider=lambda: 'foobar')

    assert len(env.factories_ips) == c.factories_num
    for f in c.factories:
        assert f.private_host in env.factories_ips

    assert env.orchestrator_ip == c.orchestrator.private_host
    assert env.key.startswith("/path/to/home")
