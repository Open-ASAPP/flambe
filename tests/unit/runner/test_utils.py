import pytest
import tempfile
import os

from flambe.runner import utils


MB = 2 ** 20


def create_file(filename, size_MB = 1):
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
