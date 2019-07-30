==========
Contribute
==========


Guide
-----


We aim to foster a healthy, inclusive, organized and efficient community around
the Flambé project. We believe prioritizing responsiveness, thoroughness and helping
others along with following best practices in GitHub issue tracking,
and our own state of the art model tracking will help us achieve the shared goal
of making Flambé as helpful as possible to the research community. Below we outline
what’s involved in different aspects of contribution.

Filing an Issue (Bug or Enhancement)
************************************

Please create an issue on Github_.

.. _Github: http://github.com/Open-ASAPP/flambe

Please follow the templates shown when you click on “Create Issue.”

Fixing a Bug (PR, etc.)
***********************

Open an issue for the bug if one doesn’t exist already. Then start working on
your own branch off of master with the following naming convention: issue-###_short-desc_github-handle,
using the issue number for the relevant bug. Once you have finished your work on your
own branch (and ideally added new tests to catch the bug!), you can open a PR to
dev. The PR will be reviewed and merged, and ultimately included in a future
release (see next section).

If your bug fix does not follow our style and contribution guidelines, you will
have to make the necessary changes before we can accept the PR. Further, you can
expect at least 1-2 reviews from the flambé team where we will check for code quality.

Adding New Features
*******************

Open an issue for the new feature or model you would like to add. Then please
follow the appropriate steps defined in :ref:`contrib-extending_label`

Running Tests
*************

To run tests (pytest, mypy, etc.) locally, you will need to install the testing requirements:

.. code:: bash

    pip install tox
    # To run, just execute tox in the terminal
    tox

Remember to add your own tests for any new code you write!


Style
------

Flambé is lightweight, pragmatic, and easy to understand; To achieve these
qualities we follow Python’s PEP8 style guide, Google’s python style guide,
and use NumPy docstring format. Because Machine Learning pipelines are prone
to subtle bugs, we use Python’s type annotations to enforce safe type contracts
across the codebase. In general, we recommend following best practices,
particularly as outlined here, up until the point that stylistic concerns
dramatically slow down development.

When contributing to Flambé, please make sure to read the style guideline for code and numpy docstrings:

- http://google.github.io/styleguide/pyguide.html
- https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

We run Flake8 and mypy in our test suite, which will flag a lot of errors unless you respect the guidelines.

“A Foolish Consistency is the Hobgoblin of Little Minds” - Emerson


.. _contrib-extending_label:

Extending Flambé with New Classes
---------------------------------

Please check back later for more information on contributing to the Flambé repo
