from contextlib import contextmanager
from urllib.parse import urlparse, ParseResult
import boto3
import botocore
import os
import subprocess
import tempfile
import requests

import logging

logger = logging.getLogger(__name__)


def s3_exists(url: ParseResult) -> bool:
    """Return is an S3 resource exists.

    Parameters
    ----------
    url: ParseResult
        The parsed URL.

    Returns
    -------
    bool
        True if it exists. False otherwise.

    """
    s3 = boto3.resource('s3')
    try:
        bucket = s3.Bucket(url.netloc)
        path = url.path[1:]  # Not consider starting '/'
        objs = list(bucket.objects.filter(Prefix=path))
        return len(objs) > 0
    except s3.meta.client.exceptions.NoSuchBucket:
        return False


def s3_remote_file(url: ParseResult) -> bool:
    """Check if an S3 hosted artifact is a file or a folder.

    Parameters
    ----------
    url: ParseResult
        The parsed URL.

    Returns
    -------
    bool
        True if it's a file, False if it's a folder.

    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(url.netloc)
    path = url.path[1:]  # Not consider starting '/'
    objs = list(bucket.objects.filter(Prefix=path))
    if len(objs) == 1 and objs[0].key == path:
        return True

    return False


def download_s3_file(url: str, destination: str) -> None:
    """Download an S3 file.

    Parameters
    ----------
    url: str
        The S3 URL. Should follow the format:
        's3://<bucket-name>[/path/to/file]'
    destination: str
        The output file where to copy the content

    """
    try:
        parsed_url = urlparse(url)
        s3 = boto3.client('s3')
        s3.download_file(parsed_url.netloc, parsed_url.path[1:], destination)
    except botocore.client.ClientError:
        raise ValueError(f"Error downlaoding artifact from s3.")


def http_exists(url: str) -> bool:
    """Check if an HTTP/HTTPS file exists.

    Parameters
    ----------
    url: str
        The HTTP/HTTPS URL.

    Returns
    -------
    bool
        True if the HTTP file exists

    """
    try:
        r = requests.head(url, allow_redirects=True)
        return r.status_code != 404
    except requests.ConnectionError:
        return False


def download_http_file(url: str, destination: str) -> None:
    """Download an HTTP/HTTPS file.

    Parameters
    ----------
    url: str
        The HTTP/HTTPS URL.
    destination: str
        The output file where to copy the content. Needs to support
        binary writing.

    """
    r = requests.get(url, allow_redirects=True)
    with open(destination, 'wb') as f:
        f.write(r.content)


def download_s3_folder(url: str, destination: str) -> None:
    """Download an S3 folder.

    Parameters
    ----------
    url: str
        The S3 URL. Should follow the format:
        's3://<bucket-name>[/path/to/folder]'
    destination: str
        The output folder where to copy the content

    """
    try:
        subprocess.check_output(
            f"aws s3 cp --recursive {url} {destination}".split(),
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as exc:
        logger.debug(exc.output)
        raise ValueError(f"Error downlaoding artifacts from s3. " +
                         "Check logs for more information")


@contextmanager
def download_manager(path: str):
    """Manager for downloading remote URLs

    Parameters
    ----------
    path: str
        The remote URL to download. Currently, only S3 and http/https
        URLs are supported.
        In case it's already a local path, it yields the same path.

    Examples
    --------

    >>> with download_manager("https://host.com/my/file.zip") as path:
    >>>     os.path.exists(path)
    >>> True

    Yields
    ------
    str
        The new local path

    """
    if os.path.exists(path):
        yield path

    else:
        url = urlparse(path)
        if url.scheme == 's3':
            if not s3_exists(url):
                raise ValueError(f"URL {path} not available")

            if s3_remote_file(url):
                with tempfile.NamedTemporaryFile() as tmpfile:
                    download_s3_file(path, tmpfile.name)
                    yield tmpfile.name
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    download_s3_folder(path, tmpdir)
                    yield tmpdir

        elif url.scheme == 'http' or url.scheme == 'https':
            if not http_exists(path):
                raise ValueError(f"URL {path} not available")

            with tempfile.NamedTemporaryFile('wb') as tmpfile:
                download_http_file(path, tmpfile.name)
                yield tmpfile.name

        else:
            raise ValueError("Currently only S3 and http/https URLs are supported")
