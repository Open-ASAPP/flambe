import os
import getpass
from flambe.compile.utils import _is_url
from typing import List

import flambe

try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze


DEFAULT_USER_PROVIDER = getpass.getuser


def _contains_path(nested_dict) -> bool:
    """Whether the nested dict contains any value that could be a path.

    Parameters
    ----------
    nested_dict
        The nested dict to evaluate

    Returns
    -------
    bool

    """
    for v in nested_dict.values():
        if hasattr(v, "values"):
            if _contains_path(v):
                return True
        if isinstance(v, str) and os.sep in v and not _is_url(v):
            return True

    return False


def is_dev_mode() -> bool:
    """Detects if flambe was installed in editable mode.

    For more information:
    https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs

    Returns
    -------
    bool

    """
    x = freeze.freeze()
    for pkg in x:
        if pkg.startswith("-e") and pkg.endswith("egg=flambe"):
            return True

    return False


def get_flambe_repo_location() -> str:
    """Return where flambe repository is located

    Returns
    -------
    str
        The local path where flambe is located

    Raises
    ------
    ValueError
        If flambe was not installed in editable mode

    """
    if not is_dev_mode():
        raise ValueError("Flambe repo can't be located as it was not \
                          installed in editable mode")

    # Go form the top level __init__.py to the flambe repo
    repo_location = os.path.join(flambe.__file__, os.pardir, os.pardir)
    return os.path.abspath(repo_location)


def get_commit_hash() -> str:
    """Get the commit hash of the current flambe development package.

    This will only work if flambe was install from github in dev mode.

    Returns
    -------
    str
        The commit hash

    Raises
    ------
    Exception
        In case flambe was not installed in dev mode.

    """
    x = freeze.freeze()
    for pkg in x:
        if "flambe" in pkg:
            if pkg.startswith("-e"):
                git_url = pkg.split(" ")[-1]
                commit = git_url.split("@")[-1].split("#")[0]
                return commit

    raise Exception("Tried to lookup commit hash in NOT development mode.")


def rsync_hosts(orch_ip: str,
                factories_ips: List[str],
                user: str,
                folder: str,
                key: str,
                exclude: List[str]) -> None:
    """Rsync the hosts in the cluster.

    IMPORTANT: this method is intended to be run in the cluster.

    Parameters
    ----------
    orch_ip: str
        The Orchestrator's IP that is visible by the factories
        (usually the private IP)
    factories_ips: List[str]
        The factories IPs that are visible by the Orchestrator
        (usually the private IPs)
    user: str
        The username of all machines.
        IMPORTANT: only machines with same username are supported
    key: str
        The key that communicate all machines
    exclude: List[str]
        A list of files to be excluded in the rsync

    """
    exc = ""
    if exclude:
        for x in exclude:
            exc += f" --exclude {x} "

    if not folder.endswith(os.sep):
        folder = f"{folder}{os.sep}"

    for f_ip in factories_ips:
        f_loc = f"{user}@{f_ip}:{folder}"
        cmd = (
            f"rsync -ae 'ssh -i {key} "
            f"-o StrictHostKeyChecking=no' {exc} "
            f"{f_loc} {folder}"
        )
        os.system(cmd)

    for f_ip in factories_ips:
        f_loc = f"{user}@{f_ip}:{folder}"
        cmd = (
            f"rsync -ae 'ssh -i {key} "
            f"-o StrictHostKeyChecking=no' {exc} "
            f"{folder} {f_loc}"
        )
        os.system(cmd)
