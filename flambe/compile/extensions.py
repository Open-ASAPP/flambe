"""
This module provides methods to orchestrate all extensions
"""

import os
from shutil import which
import re

from urllib.parse import urlparse
from git import Repo, NoSuchPathError
import subprocess

import importlib
import importlib.util
from typing import Dict, Optional, Iterable, Union
from flambe.logging import coloredlogs as cl
from flambe.compile.utils import _is_url

import logging

logger = logging.getLogger(__name__)


def download_extensions(extensions: Dict[str, str],
                        container_folder: str) -> Dict[str, str]:
    """Iterate through the extensions and download the remote urls.

    Parameters
    ----------
    extensions: Dict[str, str]
        The extensions that may contain both local or remote locations.
    container_folder: str
        The auxiliary folder where to download the remote repo

    Returns
    -------
    Dict[str, str]
        A new extensions dict with the local paths instead of remote
        urls. The local paths contain the downloaded remote resources.

    """
    ret = {}
    for key, inc in extensions.items():
        if _is_url(inc):
            loc = os.path.join(container_folder, key)
            new_inc = _download_remote_extension(inc, loc)
            ret[key] = new_inc
        else:
            expanded_inc = os.path.abspath(os.path.expanduser(inc))  # Could be path with ~, or rel
            if os.path.exists(expanded_inc):
                ret[key] = expanded_inc
            else:
                ret[key] = inc

    return ret


def _download_remote_extension(extension_url: str,
                               location: str) -> str:
    """Download a remote hosted extension.

    It fully supports github urls only (for now).

    Parameters
    ----------
    extension_url: str
        The github url pointing to an extension. For example:
        https://github.com/user/folder/tree/branch/path/to/ext
    location: str
        The location to download the repo

    Returns
    -------
    str
        The location of the installed package (which it could not
        match the location passed as parameter)

    """
    url = urlparse(extension_url)
    https_ext = url.scheme == 'https'
    desc = list(filter(lambda x: len(x) > 0, url.path.split('/')))

    if https_ext and 'github' not in url.netloc:
        raise ImportError("We only support Github hosted extensions for now through https.")

    if https_ext and len(desc) > 4 and _has_svn():
        # Special case: folder inside github repo
        # In this case we download with SVN (if available) as it
        # downloads only the folder instead of full repo
        # Ex: https://github.com/user/some_repo/tree/branch/path/to/ext
        user, repository, branch = desc[0], desc[1], desc[3]
        content = desc[4:]
        svn_url = (
            f"{url.scheme}://{url.hostname}/{user}/{repository}/"
            f"branches/{branch}/{'/'.join(content)}"
        )
        _download_svn(svn_url, location)
        logger.debug(f"Downloaded {extension_url} using svn")
    else:
        # Entire git repo (could be github or other)
        original_location = location

        # Add support for branch URLs in github.
        # github URL's path follow this structure:
        # {username}/{repo}/tree/{branch}
        if https_ext and len(desc) >= 4:
            user, repository, branch = desc[0], desc[1], desc[3]
            new_url = f"{url.scheme}://{url.hostname}/{user}/{repository}"
            location = f"{location}/{'/'.join(desc[4:])}"
            url_path = f"{user}/{repository}"
        else:
            # In case of ssh url, then remove the 'ssh://',
            # if not GitPython fails.
            new_url = extension_url if https_ext else extension_url[6:]
            url_path = url.path
            branch = "master"

        try:
            repo = Repo(original_location)
            logger.debug(f"{extension_url} already exists in {original_location}")

            remote_url = list(repo.remotes[0].urls)[0]  # Pick origin url

            # Previous extensions does not match this one
            if not remote_url.endswith(url_path):
                subprocess.check_call(f"rm -rf {original_location}".split(),
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
                repo = Repo.clone_from(new_url, original_location)
                logger.debug(f"{extension_url} git cloned as it had a different origin")

        except NoSuchPathError:
            # Repo was not downloaded before
            repo = Repo.clone_from(new_url, original_location)
            logger.debug(f"Downloaded {extension_url} using git clone")

        repo.remotes.origin.fetch()
        repo.git.checkout(branch)
        repo.remotes.origin.pull()
        logger.debug(f"Pulled latest changes from {extension_url}")

    logger.info(cl.YE(f"Downloaded extension {extension_url}"))
    return location


def _has_svn() -> bool:
    """Return if the host has svn installed"""
    return which('svn') is not None


def _download_svn(svn_url: str, location: str,
                  username: Optional[str] = None, password: Optional[str] = None) -> None:
    """Use svn to download a specific folder inside a git repo.

    This works only with remote Github repositories.

    Parameters
    ----------
    svn_url: str
        The github URL adapted to use the SVN protocol
    location: str
        The location to download the folder
    username: str
        The username
    password: str
        The password

    """
    cmd = ['svn', 'export']
    if username:
        cmd.extend(['--username', username])
    if password:
        cmd.extend(['--password', password])

    cmd.extend(['--force', svn_url, location])

    ret = subprocess.check_call(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    if ret != 0:
        raise ImportError(f"Could not download folder through svn {svn_url}")


def install_extensions(extensions: Dict[str, str],
                       user_flag: bool = False) -> None:
    """Install extensions.

    At this point, all extensions must be either local paths or
    valid pypi packages.

    Remote extensions hosted in Github must have been download first.

    Parameters
    ----------
    extensions: Dict[str, str]
        Dictionary of extensions
    user_flag: bool
        Use --user flag when running pip install

    """
    cmd = ['python3', '-m', 'pip', 'install', '-U']
    if user_flag:
        cmd.append('--user')
    for ext, resource in extensions.items():
        curr_cmd = cmd[:]

        try:
            if os.path.exists(resource):
                # Package is local
                if os.sep not in resource:
                    resource = f"./{resource}"
            else:
                # Package follows pypi notation: "torch>=0.4.1,<1.1"
                resource = f"{resource}"

            curr_cmd.append(resource)

            output: Union[bytes, str]
            output = subprocess.check_output(
                curr_cmd,
                stderr=subprocess.DEVNULL
            )

            output = output.decode("utf-8")

            for l in output.splitlines():
                logger.debug(l)
                r = re.search(r'Successfully uninstalled (?P<pkg_name>\D*)-(?P<version>.*)', l)
                if r and 'pkg_name' in r.groupdict():
                    logger.info(cl.RE(f"WARNING: While installing {ext}, " +
                                      f"existing {r.groupdict()['pkg_name']}-" +
                                      f"{r.groupdict()['version']} was uninstalled."))
        except subprocess.CalledProcessError:
            raise ImportError(f"Could not install package in {resource}")

        logger.info(cl.GR(f"Successfully installed {ext}"))


def is_installed_module(module_name: str) -> bool:
    """Whether the module is installed.

    Parameters
    ----------
    module_name: str
        The name of the module to check for

    Returns
    -------
    bool
        True if the module is installed locally, False otherwise.

    """
    return importlib.util.find_spec(module_name) is not None


def import_modules(modules: Iterable[str]) -> None:
    """Dinamically import modules

    Parameters
    ----------
    modules: Iterable[str]
        An iterable of strings containing the modules
        to import

    """
    for mod_name in modules:
        try:
            # Importing modules adds undesired handlers to
            # the root logger.
            # We will backup the handlers and updates them
            # after importing
            backup_handlers = logging.root.handlers[:]

            importlib.import_module(mod_name)

            # Remove all existing root handlers and
            # re-apply the backed up root handlers
            for x in logging.root.handlers[:]:
                logging.root.removeHandler(x)
            for x in backup_handlers:
                logging.root.addHandler(x)

            logger.info(cl.YE(f"Imported extensions {mod_name}"))
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Error importing {mod_name}: {e}. Please 'pip install' " +
                "the package manually or use '-i' flag (only applies when running " +
                "flambe as cmd line program)"
            )


def setup_default_modules():
    from flambe.compile.utils import make_component
    from flambe.optim import LRScheduler
    import torch
    import ray
    exclude = ['torch.nn.quantized', 'torch.nn.qat']
    make_component(torch.nn.Module, only_module='torch.nn', exclude=exclude)
    make_component(torch.optim.Optimizer, only_module='torch.optim')
    make_component(torch.optim.lr_scheduler._LRScheduler,
                   only_module='torch.optim.lr_scheduler', parent_component_class=LRScheduler)
    make_component(ray.tune.schedulers.TrialScheduler)
    make_component(ray.tune.suggest.SearchAlgorithm)
