import os
import logging
from typing import Iterable

from flambe.const import FLAMBE_GLOBAL_FOLDER
from flambe.experiment.wording import print_extensions_cache_size_warning

logger = logging.getLogger(__name__)


MB = 2**20
WARN_LIMIT_MB = 100


def get_files(path: str) -> Iterable[str]:
    """Return the list of all files (recursively)
    a directory has.

    Parameters
    ----------
    path: str
        The directory's path

    Return
    ------
    List[str]
        The list of files (each file with its path from
        the given parameter)

    Raise
    -----
    ValueError
        In case the path does not exist

    """
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist")

    def _wrapped():
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                yield fp

    return _wrapped()


def get_size_MB(path: str) -> float:
    """Return the size of a file/folder in MB.

    Parameters
    ----------
    path: str
        The path to the folder or file

    Returns
    -------
    float
        The size in MB

    """
    accum = 0
    if os.path.isdir(path):
        for fp in get_files(path):
            if os.path.exists(fp) and not os.path.islink(fp):
                accum += os.path.getsize(fp)
    else:
        accum = os.path.getsize(path)
    return accum / MB


def check_system_reqs() -> None:
    """Run system checks and prepare the system before a run.

    This method should:
        * Create folders, files that are needed for flambe
        * Raise errors in case requirements are not met. This should
        run under the SafeExecutionContext, so errors will be handled
        * Warn the user in case something needs attention.

    """
    # Create the flambe folder if it does not exist
    if not os.path.exists(FLAMBE_GLOBAL_FOLDER):
        os.mkdir(FLAMBE_GLOBAL_FOLDER)

    # Check if extensions folder is getting big
    extensions_folder = os.path.join(FLAMBE_GLOBAL_FOLDER, "extensions")
    if os.path.exists(extensions_folder) and get_size_MB(extensions_folder) > WARN_LIMIT_MB:
        print_extensions_cache_size_warning(extensions_folder, WARN_LIMIT_MB)
