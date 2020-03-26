import tempfile
import mock


from flambe.compile.utils import write_deps


def test_write_deps():
    dummy_dependencies = ['numpy==1.2.3', 'pip~=1.1.1', 'some_other-random dep']
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_deps(tmpfile.name, dummy_dependencies)

        assert tmpfile.read() == b'numpy==1.2.3\npip~=1.1.1\nsome_other-random dep'


@mock.patch('flambe.compile.utils.get_frozen_deps')
def test_write_deps_default(mock_deps):
    mock_deps.return_value = ['numpy==1.2.3', 'pip~=1.1.1', 'some_other-random dep']
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_deps(tmpfile.name)
        assert tmpfile.read() == b'numpy==1.2.3\npip~=1.1.1\nsome_other-random dep'
        mock_deps.assert_called_once()
