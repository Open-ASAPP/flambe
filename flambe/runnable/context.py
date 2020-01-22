# from __future__ import annotations
import os
from typing import Optional, List, Type, Any, Dict, cast, Tuple
from types import TracebackType

from io import StringIO
import sys

import errno

import logging
import configparser

from flambe.compile import Registrable
from flambe.compile import yaml
from flambe.runnable import error
from flambe.runnable.runnable import Runnable
from flambe.compile.extensions import download_extensions
from flambe.compile.extensions import install_extensions, import_modules
from flambe.const import FLAMBE_GLOBAL_FOLDER
from flambe.logging import coloredlogs as cl

logger = logging.getLogger(__name__)


class SafeExecutionContext:
    """Context manager handling the experiment's creation and execution.

    Parameters
    ----------
    yaml_file: str
        The experiment filename

    """
    def __init__(self, yaml_file: str) -> None:
        self.to_remove: List[str] = []
        self.yaml_file = yaml_file
        self.content: str = ""

    def __enter__(self) -> "SafeExecutionContext":
        """A SafeExecutionContext should be used as a context manager
        to handle all possible errors in a clear way.

        Examples
        --------

        >>> with SafeExecutionContext(...) as ex:
        >>>     ...

        """
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 tb: Optional[TracebackType]):
        """Exit method for the context manager.

        This method will catch any exception, and return True. This
        means that all exceptions produced in a SafeExecutionContext
        (used with the context manager) will not continue to raise.

        """
        if exc_type is not None and exc_value is not None:
            # Rollback and undo cluster in case or exception
            logger.error(cl.RE(repr(exc_value)), exc_info=(exc_type, exc_value, tb))
            if isinstance(exc_value, error.RunnableFileError):
                sys.exit(errno.EINVAL)
            else:
                sys.exit(-1)

        return False

    def preprocess(self,
                   secrets: Optional[str] = None,
                   download_ext: bool = True,
                   install_ext: bool = False,
                   import_ext: bool = True,
                   check_tags: bool = True,
                   **kwargs) -> Tuple[Runnable, Dict[str, str]]:
        """Preprocess the runnable file.

        Looks for syntax errors, import errors, etc. Also injects
        the secrets into the runnables.

        If this method runs and ends without exceptions, then the
        experiment is ok to be run. If this method raises an Error and
        the SafeExecutionContext is used as context manager,
        then the __exit__ method will be executed.

        Parameters
        ----------
        secrets: Optional[str]
            Optional path to the secrets file
        install_ext: bool
            Whether to install the extensions or not.
            This process also downloads the remote extensions.
            Defaults to False
        install_ext: bool
            Whether to import the extensions or not.
            Defaults to True.
        check_tags: bool
            Whether to check that all tags are valid. Defaults to True.

        Returns
        -------
        Tuple[Runnable, Dict[str, str]]
            A tuple containing the compiled Runnable and a dict
            containing the extensions the Runnable uses.

        Raises
        ------
        Exception
            Depending on the error.

        """
        content, extensions = self.first_parse()

        config = configparser.ConfigParser()
        if secrets:
            config.read(secrets)

        if install_ext:
            t = os.path.join(FLAMBE_GLOBAL_FOLDER, "extensions")
            extensions = download_extensions(extensions, t)
            install_extensions(extensions, user_flag=False)

        if import_ext:
            import_modules(extensions.keys())

        # Check that all tags are valid
        if check_tags:
            self.check_tags(content)

        # Compile the runnable now that the extensions were imported.
        runnable = self.compile_runnable(content)

        if secrets:
            runnable.inject_secrets(secrets)

        if extensions:
            runnable.inject_extensions(extensions)

        runnable.parse()

        return runnable, extensions

    def first_parse(self) -> Tuple[str, Dict]:
        """Check if valid YAML file and also load config

        In this first parse the runnable does not get compiled because
        it could be a custom Runnable, so it needs the extensions
        to be imported first.

        """
        if not os.path.exists(self.yaml_file):
            raise FileNotFoundError(
                f"Configuration file '{self.yaml_file}' not found. Terminating."
            )

        with open(self.yaml_file, 'r') as f:
            content = f.read()

        try:
            yamls = list(yaml.load_all(content))
        except TypeError as e:
            raise error.ParsingRunnableError(f"Syntax error compiling the runnable: {str(e)}")

        if len(yamls) > 2:
            raise ValueError(f"{self.yaml_file} should contain an (optional) extensions sections" +
                             " and the main runnable object.")

        extensions: Dict[str, str] = {}
        if len(yamls) == 2:
            extensions = dict(yamls[0])

        # We want self.content to be a string with the raw content
        # We will precompile later once all extensions are registered.
        with StringIO() as stream:
            yaml.dump(yamls[-1], stream)
            content = stream.getvalue()

        return content, extensions

    def check_tags(self, content: str):
        """Check that all the tags are valid.

        Parameters
        ----------
        content : str
            The content of the YAML file

        Raises
        ------
        TagError

        """
        # Get all the registered tags, and flatten
        registered_tags = {t for _, tags in Registrable._yaml_tags.items() for t in tags}

        # Check against tags in this config
        parsing_events = yaml.parse(content)
        for event in parsing_events:
            if hasattr(event, 'tag') and event.tag is not None:
                if event.tag not in registered_tags:
                    raise error.TagError(f"Unknown tag: {event.tag}. Make sure the class, \
                                            or factory was correctly registered.")

    def compile_runnable(self, content: str) -> Runnable:
        """Compiles and returns the Runnable.

        IMPORTANT: This method should run after all
        extensions were registered.

        Parameters
        ----------
        content: str
            The runnable, as a YAML string

        Returns
        -------
        Runnable
            The compiled experiment.

        """
        ret: Any = yaml.load(content)
        if not isinstance(ret, Runnable):
            raise ValueError("Tried to run a non-Runnable")
        cast(Runnable, ret)
        ret.inject_content(content)
        return ret
