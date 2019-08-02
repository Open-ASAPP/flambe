from collections import namedtuple
import boto3

from flambe.cluster import const

from typing import Optional


RemoteCommand = namedtuple('RemoteCommand', ['success', 'msg'])


def _get_images():
    """Get the official AWS public AMIs created by Flambe
    that have tag 'Creator: flambe@asapp.com'

    ATTENTION: why not just search the tags? We need to make sure
    the AMIs we pick were created by the Flambe team. Because of tags
    values not being unique, anyone can create a public AMI with
    'Creator: flambe@asapp.com' as a tag. If we pick that AMI, then
    we could potentially be creating instances with unknown AMIs,
    causing potential security issues.
    By filtering by our acount id (which can be public), then we can
    make sure that all AMIs that are being scanned were created
    by Flambe team.

    """
    client = boto3.client('ec2')
    return client.describe_images(Owners=[const.AWS_FLAMBE_ACCOUNT])


def _get_matching_ami(_type: str, version: str, default: bool = True) -> Optional[str]:
    """Gives the matching AMI given the type.

    If default is on, then it will return a default AMI in case it does
    not find the correct one matching version.

    Parameters
    ----------
    _type: str
        It can be either 'factory' or 'orchestrator'.
        Note that the type is lowercase in the AMI tag.
    version: str
        For example, "0.2.1" or "2.0".
    default: bool
        If default is True, then if no matching AMI for the version is
        found it will look for a default flambe (which is version 0.0.0)
        and return that AMI id.

    Returns
    -------
    The ImageId if it's found. None if not.

    """
    ami = _get_ami(_type, version)

    if ami is None and default:
        return _find_default_ami(_type)

    return ami


def _get_ami(_type: str, version: str) -> Optional[str]:
    """Given a type and a version, get the correct Flambe AMI.

    Parameters
    ----------
    _type: str
        It can be either 'factory' or 'orchestrator'.
        Note that the type is lowercase in the AMI tag.
    version: str
        For example, "0.2.1" or "2.0".

    Returns
    -------
    The ImageId if it's found. None if not.

    """
    images = _get_images()
    for i in images['Images']:
        # Name is for example 'flambe-orchestrator 0.0.0'
        n, v = i['Name'].split()
        if n == f"flambe-{_type.lower()}-ami" and v == version:
            return i['ImageId']

    return None


def _find_default_ami(_type: str):
    """Returns an AMI with version 0.0.0, which is the default.
    This means that doesn't contain flambe itself but it has
    some heavy dependencies already installed (like pytorch).

    Parameters
    ----------
    _type: str
        Wether is "orchestrator" or "factory"

    Returns
    -------
    The ImageId

    Raises
    ------
    ClusterError
        If AMI is not found

    """
    return _get_ami(_type, '0.0.0')


def _get_matching_factory_ami(version: str) -> Optional[str]:
    """Get the matching ImageId for the factory.

    Returns
    -------
    The ImageId

    """
    return _get_matching_ami("factory", version=version)


def _get_matching_orchestrator_ami(version: str) -> Optional[str]:
    """Get the matching ImageId for the orchestrator.

    Returns
    -------
    The ImageId

    """
    return _get_matching_ami("orchestrator", version=version)
