import tempfile
import os
import boto3
import dill

import subprocess
from urllib.parse import urlparse

import flambe
from flambe.runnable import Runnable, error
from flambe.compile import Component, Schema
from flambe.compile.const import DEFAULT_PROTOCOL
from flambe.logging import coloredlogs as cl

import logging

logger = logging.getLogger(__name__)


class Builder(Runnable):
    """Implement a Builder.

    A builder is a simple object that can be used to create
    any Component post-experiment, and export it to a local
    or remote location.

    Currently supports local, and S3 locations.

    Attributes
    ----------
    config: configparser.ConfigParser
        The secrets that the user provides. For example,
        'config["AWS"]["ACCESS_KEY"]'

    """
    def __init__(self,
                 component: Schema,
                 destination: str,
                 storage: str = 'local',
                 compress: bool = False,
                 pickle_only: bool = False,
                 pickle_module=dill,
                 pickle_protocol=DEFAULT_PROTOCOL) -> None:
        """Initialize the Builder.

        Parameters
        ----------
        component : Schema
            The object to build, and export
        destination : str
            The destination where the object should be saved.
            If an s3 bucket is specified, 's3' should also be
            specified as the storage argument. s3 destinations
            should have the following syntax:
            's3://<bucket-name>[/path/to/folder]'
        storage: str
            The storage location. One of: [local | s3]
        compress : bool
            Whether to compress the save file / directory via tar + gz
        pickle_only : bool
            Use given pickle_module instead of the hiearchical save
            format (the default is False).
        pickle_module : type
            Pickle module that has load and dump methods; dump should
            accept a pickle_protocol parameter (the default is dill).
        pickle_protocol : type
            Pickle protocol to use; see pickle for more details (the
            default is 2).

        """
        super().__init__()

        self.destination = destination
        self.component = component

        self.compiled_component: Component

        self.storage = storage
        self.serialization_args = {
            'compress': compress,
            'pickle_only': pickle_only,
            'pickle_module': pickle_module,
            'pickle_protocol': pickle_protocol
        }

    def run(self, force: bool = False, **kwargs) -> None:
        """Run the Builder."""

        # Add information about the extensions. This ensures
        # the compiled component has the extensions information
        self.component.add_extensions_metadata(self.extensions)

        self.compiled_component = self.component()  # Compile Schema

        if self.storage == 'local':
            self.save_local(force)
        elif self.storage == 's3':
            self.save_s3(force)
        else:
            msg = f"Unknown storage {self.storage}, should be one of: [local, s3]"
            raise ValueError(msg)

    def save_local(self, force) -> None:
        """Save an object locally.

        Parameters
        ----------
        force: bool
            Wheter to use a non-empty folder or not

        """
        if (
            os.path.exists(self.destination) and
            os.listdir(self.destination) and
            not force
        ):
            raise error.ParsingRunnableError(
                f"Destination {self.destination} folder is not empty. " +
                "Use --force to force the usage of this folder or " +
                "pick another destination."
            )

        flambe.save(self.compiled_component, self.destination, **self.serialization_args)

    def get_boto_session(self):
        """Get a boto Session

        """
        return boto3.Session()

    def save_s3(self, force) -> None:
        """Save an object to s3 using awscli

        Parameters
        ----------
        force: bool
            Wheter to use a non-empty bucket folder or not

        """
        url = urlparse(self.destination)

        if url.scheme != 's3' or url.netloc == '':
            raise error.ParsingRunnableError(
                "When uploading to s3, destination should be: " +
                "s3://<bucket-name>[/path/to/dir]"
            )

        bucket_name = url.netloc
        s3 = self.get_boto_session().resource('s3')
        bucket = s3.Bucket(bucket_name)

        for content in bucket.objects.all():
            path = url.path[1:]  # Remove first '/'
            if content.key.startswith(path) and not force:
                raise error.ParsingRunnableError(
                    f"Destination {self.destination} is not empty. " +
                    "Use --force to force the usage of this bucket folder or " +
                    "pick another destination."
                )

        with tempfile.TemporaryDirectory() as tmpdirname:
            flambe.save(self.compiled_component, tmpdirname, **self.serialization_args)
            try:
                subprocess.check_output(
                    f"aws s3 cp --recursive {tmpdirname} {self.destination}".split(),
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
            except subprocess.CalledProcessError as exc:
                logger.debug(exc.output)
                raise ValueError(f"Error uploading artifacts to s3. " +
                                 "Check logs for more information")
            else:
                logger.info(cl.BL(f"Done uploading to {self.destination}"))
