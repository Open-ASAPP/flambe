import pytest
import argparse

from flambe.runner.run import main
from flambe.runnable.runnable import Runnable

import mock


@pytest.fixture(scope='function')
def args():
    args = argparse.Namespace()
    args.config = 'config.yaml'
    args.debug = False
    args.force = False
    args.verbose = False
    args.install_extensions = False
    args.cluster = None
    args.secrets = None
    return args


class DummyRunnable(Runnable):

    def run(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture(scope='function')
def runnable():
    return DummyRunnable()


def test_debug_remote(args):
    args.debug = True
    args.cluster = "cluster.yaml"

    with pytest.raises(ValueError):
        main(args)


@pytest.mark.parametrize('debug', [True, False])
@pytest.mark.parametrize('force', [True, False])
@mock.patch('flambe.runner.run.SafeExecutionContext.preprocess')
def test_runnable_args(mock_preprocess, force, debug, runnable, args):
    mock_preprocess.return_value = (runnable, None)
    args.debug = debug
    args.force = force

    main(args)

    assert runnable.kwargs['debug'] == debug
    assert runnable.kwargs['force'] == force
