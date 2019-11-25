from flambe.experiment import Experiment
from flambe.runnable import RemoteEnvironment

import mock
import pytest


@pytest.fixture
def get_experiment():
    def wrapped(**kwargs):
        return Experiment(name=kwargs.get('name', 'test'),
                          pipeline=kwargs.get('pipeline', {}),
                          resume=kwargs.get('resume', False),
                          debug=kwargs.get('debug', False),
                          devices=kwargs.get('devices', None),
                          save_path=kwargs.get('save_path', None),
                          resources=kwargs.get('resources', None),
                          search=kwargs.get('search', None),
                          schedulers=kwargs.get('schedulers', None),
                          reduce=kwargs.get('reduce', None),
                          env=kwargs.get('env', None),
                          max_failures=kwargs.get('max_failures', 1),
                          stop_on_failure=kwargs.get('stop_on_failure', True),
                          merge_plot=kwargs.get('merge_plot', True))
    return wrapped


@pytest.fixture
def get_env():
    def wrapped(**kwargs):
        env = RemoteEnvironment(
            key=kwargs.get('key', 'my-key'),
            orchestrator_ip=kwargs.get('orchestrator_ip', '1.1.1.1'),
            factories_ips=kwargs.get('factories_ips', ['1.1.1.1']),
            user=kwargs.get('user', 'ubuntu'),
            local_user=kwargs.get('local_user', 'some_user'),
        )
        return env
    return wrapped


@mock.patch('flambe.experiment.experiment.getpass.getuser')
def test_get_user(mock_user, get_experiment, get_env):
    mock_user.return_value = 'foobar'

    exp = get_experiment()
    assert exp.get_user() == 'foobar'
    mock_user.assert_called_once()

    mock_user.reset_mock()

    exp = get_experiment(env=get_env(local_user='barfoo'))
    assert exp.get_user() == 'barfoo'
    mock_user.assert_not_called()