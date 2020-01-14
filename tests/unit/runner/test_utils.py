import pytest
import tempfile
import os

from flambe.runner import utils


MB = 2 ** 20


def create_file(filename, size_MB=1):
    # From https://stackoverflow.com/a/8816154
    with open(filename, "wb") as out:
        out.truncate(size_MB * MB)


@pytest.mark.parametrize("mbs", [1, 2, 3, 4])
def test_size_MB_file(mbs):
    with tempfile.NamedTemporaryFile("wb") as t:
        create_file(t.name, size_MB=mbs)
        assert utils.get_size_MB(t.name) == mbs


@pytest.mark.parametrize("mbs", [1, 2, 3, 4])
def test_size_MB_folder(mbs):
    with tempfile.TemporaryDirectory() as t:
        create_file(os.path.join(t, '1.bin'), size_MB=mbs)
        create_file(os.path.join(t, '2.bin'), size_MB=mbs)
        create_file(os.path.join(t, '3.bin'), size_MB=mbs)
        create_file(os.path.join(t, '4.bin'), size_MB=mbs)
        assert utils.get_size_MB(t) == 4 * mbs


def test_get_files():
    with tempfile.TemporaryDirectory() as t:
        f1 = os.path.join(t, 'some_file.txt')
        os.mkdir(os.path.join(t, 'folder'))
        f2 = os.path.join(t, 'folder', 'some_file.txt')
        open(f1, 'w+').close()
        open(f2, 'w+').close()

        assert list(utils.get_files(t)) == [f1, f2]
