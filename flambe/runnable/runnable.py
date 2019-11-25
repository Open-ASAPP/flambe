from abc import abstractmethod
import configparser
from typing import Dict, Optional, Callable

from flambe.compile import MappedRegistrable
from flambe.runnable.utils import DEFAULT_USER_PROVIDER


class Runnable(MappedRegistrable):
    """Base class for all runnables.

    A runnable contains a single run method that needs to
    be implemented. It must contain all the logic for
    the runnable.

    Each runnable has also access to the secrets the user
    provides.

    Examples of Runnables: Experiment, Cluster

    Attributes
    ----------
    config: configparser.ConfigParser
        The secrets that the user provides. For example,
        'config["AWS"]["ACCESS_KEY"]'
    extensions: Dict[str, str]
        The extensions used for this runnable.
    content: Optional[str]
        This attribute will hold the YAML representation
        of the Runnable.
    user_provider: Callable[[], str]
        The logic for specifying the user triggering this
        Runnable. If not passed, by default it will pick the computer's
        user.

    """
    def __init__(self, user_provider: Callable[[], str] = None, **kwargs) -> None:
        self.config = configparser.ConfigParser()
        self.extensions: Dict[str, str] = {}
        self.content: Optional[str] = None

        self.user_provider = user_provider or DEFAULT_USER_PROVIDER

    def inject_content(self, content: str) -> None:
        """Inject the original YAML string that was used
        to generate this Runnable instance.

        Parameters
        ----------
        content: str
            The YAML, as a string

        """
        self.content = content

    def inject_secrets(self, secrets: str) -> None:
        """Inject the secrets once the Runnable
        was created.

        Parameters
        ----------
        secrets: str
            The filepath to the secrets

        """
        self.config.read(secrets)

    def inject_extensions(self, extensions: Dict[str, str]) -> None:
        """Inject extensions to the Runnable

        Parameters
        ----------
        extensions: Dict[str, str]
            The extensions

        """
        self.extensions = extensions

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Run the runnable.

        Each implementation will implement its
        own logic, with the parameters it needs.

        """
        raise NotImplementedError()

    def parse(self) -> None:
        """Parse the runnable to determine if it's able to run.
        Raises
        ------
        ParsingExperimentError
            In case a parsing error is found.
        """
        pass
