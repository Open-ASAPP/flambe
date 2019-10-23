

"""
setup.py
"""

from setuptools import setup, find_packages
from typing import Dict
import os


NAME = "flambe"
AUTHOR = "ASAPP Inc."
EMAIL = "flambe@asapp.com"
DESCRIPTION = "Pytorch based library for robust prototyping, standardized  \
               benchmarking, and effortless experiment management"


def readme():
    with open('README-pypi.rst', encoding='utf-8') as f:
        return f.read()


def required():
    with open('requirements.txt') as f:
        return f.read().splitlines()


# So that we don't import flambe.
VERSION: Dict[str, str] = {}
with open("flambe/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(

    name=NAME,
    version=os.environ.get("TAG_VERSION", VERSION['VERSION']),

    description=DESCRIPTION,
    long_description=readme(),
    long_description_content_type="text/x-rst; charset=UTF-8",

    # Author information
    author=AUTHOR,
    author_email=EMAIL,

    # What is packaged here.
    packages=find_packages(exclude=("tests", "tests.*", "extensions")),
    scripts=[
        'bin/flambe',
        'bin/flambe-site'
    ],

    install_requires=required(),
    include_package_data=True,

    python_requires='>=3.6.1',
    zip_safe=True

)
