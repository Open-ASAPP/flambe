# flake8: noqa: E402

import logging as main_logging

# Work based on https://github.com/tensorflow/tensorflow/issues/26691
# This check is done to avoid tensorflow import (done by pytorch 1.1)
# break logging.
try:
    # Tensorflow uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    main_logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

main_logging.disable(main_logging.WARNING)

from flambe.compile import Component, Schema, save, load
from flambe.compile import save_state_to_file, load_state_from_file
from flambe.logging import log
from flambe import compile, dataset, experiment, field, learn, nlp, vision, export, model
from flambe import cluster, metric, nn, runner, sampler, runnable, tokenizer, optim
from flambe.version import VERSION as __version__
from flambe.logo import ASCII_LOGO


__all__ = ['Component', 'Schema', 'log', 'tokenizer',
           'compile', 'dataset', 'experiment', 'field', 'learn', 'export',
           'cluster', 'metric', 'nn', 'model', 'optim', 'runner', 'runnable', 'sampler',
            'nlp', 'vision', '__version__', 'ASCII_LOGO',  'save', 'load',
           'load_state_from_file', 'save_state_to_file']


main_logging.disable(main_logging.NOTSET)
