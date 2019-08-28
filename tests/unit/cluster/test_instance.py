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
import configparser

import flambe

import mock
from flambe.runnable import SafeExecutionContext, error

from flambe.cluster import instance
from flambe.cluster.utils import RemoteCommand
from flambe.cluster.instance import errors


class MockSSHClient():

    def __init__(self, status, success_msg, error_msg):
        self.status = status
        self.success_msg = success_msg
        self.error_msg = error_msg

    def exec_command(self, cmd):
        stdout = mock.MagicMock()
        stderr = mock.MagicMock()
        stdout.channel.exit_status_ready = mock.MagicMock(return_value=True)

        stdout.read = mock.MagicMock(return_value=self.success_msg)
        stderr.read = mock.MagicMock(return_value=self.error_msg)

        return self.status, stdout, stderr


@pytest.fixture
def get_instance():

    def _get_instance(_class=None, **kwargs):
        host=kwargs.get('host', '1.2.3.4')
        private_host=kwargs.get('private_host', '10.0.3.4')
        username=kwargs.get('username', 'ubuntu')
        key=kwargs.get('key', '/path/to/ssh/key')
        config=kwargs.get('config', configparser.ConfigParser())
        debug=kwargs.get('debug', False)
        use_public=kwargs.get('use_public', True)
        if _class:
            return _class(host=host, private_host=private_host, username=username,
                          key=key, config=config, debug=debug, use_public=use_public)

        return instance.Instance(host=host, private_host=private_host, username=username,
                                 key=key, config=config, debug=debug, use_public=use_public)
    return _get_instance


@mock.patch('flambe.cluster.instance.instance.Instance.is_up')
def test_reachable_instance(mock_is_up, get_instance):
    mock_is_up.return_value = True
    ins = get_instance()
    ins.wait_until_accessible()


@mock.patch('flambe.cluster.instance.instance.Instance.is_up')
@mock.patch('flambe.cluster.const.RETRIES', 1)
@mock.patch('flambe.cluster.const.RETRY_DELAY', 1)
def test_unreachable_instance(mock_is_up, get_instance):
    mock_is_up.return_value = False
    ins = get_instance()

    with pytest.raises(ConnectionError):
        ins.wait_until_accessible()
    

@mock.patch('flambe.cluster.instance.instance.Instance._get_cli')
def test_run_cmd(mock_ssh_cli, get_instance):
    mock_ssh_cli.return_value = MockSSHClient(0, b"success", b"")
    ins = get_instance()

    ret = ins._run_cmd("ls")
    assert ret.success is True
    assert ret.msg == b"success"
    

    mock_ssh_cli.return_value = MockSSHClient(123, b"", b"error")
    ins = get_instance()

    ret = ins._run_cmd("ls")
    assert ret.success is False
    assert ret.msg == b"error"
    

@mock.patch('flambe.cluster.instance.instance.Instance._get_cli')
def test_run_cmd2(mock_ssh_cli, get_instance):
    mock_ssh_cli.return_value = MockSSHClient(0, b"/home/ubuntu", b"")
    ins = get_instance()

    assert ins.get_home_path() == "/home/ubuntu"


@mock.patch('flambe.cluster.instance.instance.Instance._get_cli')
def test_run_cmds(mock_ssh_cli, get_instance):
    mock_ssh_cli.return_value = MockSSHClient(0, b"/home/ubuntu", b"")
    ins = get_instance()

    ins.run_cmds(['ls', 'ls'])
    

@mock.patch('flambe.cluster.instance.instance.Instance._get_cli')
def test_failure_run_cmds(mock_ssh_cli, get_instance):
    mock_ssh_cli.return_value = MockSSHClient(1, b"", b"error")
    ins = get_instance()

    with pytest.raises(errors.RemoteCommandError):
        ins.run_cmds(['ls', 'ls'])


@mock.patch('flambe.cluster.instance.instance.Instance._run_cmd')
def test_install_flambe(mock_run_cmd, get_instance):
    mock_run_cmd.return_value = RemoteCommand(True, b"")
    ins = get_instance(debug=False)

    ins.install_flambe()

    cmd = f'python3 -m pip install --user --upgrade  flambe=={flambe.__version__}'
    mock_run_cmd.assert_called_with(cmd, retries=3)


@mock.patch('flambe.cluster.instance.instance.Instance._run_cmd')
@mock.patch('flambe.cluster.instance.instance.Instance.contains_gpu')
def test_install_flambe_gpu(mock_contains_gpu, mock_run_cmd, get_instance):
    mock_contains_gpu.return_value = True
    mock_run_cmd.return_value = RemoteCommand(True, b"")
    ins = get_instance(_class=instance.GPUFactoryInstance, debug=False)

    ins.install_flambe()

    cmd = f'python3 -m pip install --user --upgrade  flambe[cuda]=={flambe.__version__}'
    mock_run_cmd.assert_called_with(cmd, retries=3)


@mock.patch('flambe.cluster.instance.instance.get_flambe_repo_location')
@mock.patch('flambe.cluster.instance.instance.Instance.get_home_path')
@mock.patch('flambe.cluster.instance.instance.Instance.send_rsync')
@mock.patch('flambe.cluster.instance.instance.Instance._run_cmd')
@mock.patch('os.path.exists')
def test_install_flambe_debug(mock_os_exists, mock_run_cmd, mock_rsync, mock_get_home_path, mock_flambe_loc, get_instance):
    mock_os_exists.return_value = False
    mock_flambe_loc.return_value = "/home/user/flambe"

    mock_get_home_path.return_value = "/home/ubuntu"
    mock_run_cmd.return_value = RemoteCommand(True, b"")

    ins = get_instance(debug=True)

    ins.install_flambe()

    cmd = f'python3 -m pip install --user --upgrade  /home/ubuntu/extensions/flambe'
    mock_run_cmd.assert_called_with(cmd, retries=3)

    mock_flambe_loc.assert_called_once()

    mock_get_home_path.assert_called_once()
    mock_rsync.assert_called_once_with('/home/user/flambe', '/home/ubuntu/extensions/flambe', params=["--exclude='.*'", "--exclude='docs/*'"])


@mock.patch('flambe.cluster.instance.instance.get_flambe_repo_location')
@mock.patch('flambe.cluster.instance.instance.Instance.get_home_path')
@mock.patch('flambe.cluster.instance.instance.Instance.send_rsync')
@mock.patch('flambe.cluster.instance.instance.Instance._run_cmd')
@mock.patch('os.path.exists')
def test_install_flambe_debug_2(mock_os_exists, mock_run_cmd, mock_rsync, mock_get_home_path, mock_flambe_loc, get_instance):
    mock_os_exists.return_value = True
    mock_flambe_loc.return_value = "/home/user/flambe"

    mock_get_home_path.return_value = "/home/ubuntu"
    mock_run_cmd.return_value = RemoteCommand(True, b"")

    ins = get_instance(debug=True)

    ins.install_flambe()

    cmd = f'python3 -m pip install --user --upgrade  /home/ubuntu/extensions/flambe'
    mock_run_cmd.assert_called_with(cmd, retries=3)

    mock_flambe_loc.assert_called_once()

    mock_get_home_path.assert_called_once()
    mock_rsync.assert_called_once_with('/home/user/flambe', '/home/ubuntu/extensions/flambe', params=["--exclude='.*'", "--exclude='docs/*'"])


@mock.patch('flambe.cluster.instance.instance.Instance._run_cmd')
def test_install_flambe_custom_pypi(mock_run_cmd, get_instance):
    mock_run_cmd.return_value = RemoteCommand(True, b"")

    config = configparser.ConfigParser()
    config.add_section("PIP")
    config['PIP']['HOST'] = 'some_host'
    config['PIP']['HOST_URL'] = 'https://some_url'

    ins = get_instance(debug=False, config=config)

    ins.install_flambe()

    cmd = (
        'python3 -m pip install --user --upgrade --trusted-host ' +
        'some_host --extra-index-url https://some_url ' +
        f'flambe=={flambe.__version__}'
    )
    mock_run_cmd.assert_called_with(cmd, retries=3)
