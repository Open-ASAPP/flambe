# Style

Flambé is lightweight, pragmatic, and easy to understand; To achieve these qualities we follow Python’s PEP8 style guide, Google’s python style guide, and use NumPy docstring format. Because Machine Learning pipelines are prone to subtle bugs, we use Python’s type annotations to enforce safe type contracts across the codebase. In general, we recommend following best practices, particularly as outlined here, up until the point that stylistic concerns dramatically slow down development.

When contributing to Flambé, please make sure to read the style guideline for code and numpy docstrings:

- http://google.github.io/styleguide/pyguide.html
- https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

We run Flake8 and mypy in our test suite, which will flag a lot of errors unless you respect the guidelines.

> “A Foolish Consistency is the Hobgoblin of Little Minds” - Emerson
