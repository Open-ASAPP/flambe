import tempfile


from flambe.compile.utils import write_deps


def test_write_deps():
    dummy_dependencies = ['numpy==1.2.3', 'pip~=1.1.1', 'some_other-random dep']
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_deps(tmpfile.name, dummy_dependencies)

        assert tmpfile.read() == b'numpy==1.2.3\npip~=1.1.1\nsome_other-random dep'
