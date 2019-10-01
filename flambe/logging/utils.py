from typing import Dict, Union, Optional, Callable, Any
import time
import logging
import inspect
from types import SimpleNamespace

import torch
import numpy
import random

from flambe.logging import ScalarT, ScalarsT, TextT, ImageT, HistogramT, PRCurveT
import colorama
from colorama import Fore, Style
colorama.init()


ValueT = Union[float, Dict[str, float], str]


d: Dict[str, Callable] = {
    "GR": lambda x: Fore.GREEN + x + Style.RESET_ALL,
    "RE": lambda x: Fore.RED + x + Style.RESET_ALL,
    "YE": lambda x: Fore.YELLOW + x + Style.RESET_ALL,
    "BL": lambda x: Fore.CYAN + x + Style.RESET_ALL,
    "MA": lambda x: Fore.MAGENTA + x + Style.RESET_ALL,
    "RA": lambda x: random.choice([Fore.GREEN, Fore.RED, Fore.YELLOW,
                                   Fore.CYAN, Fore.MAGENTA]) + x + Style.RESET_ALL
}
coloredlogs = SimpleNamespace(**d)


def _get_context_logger() -> logging.Logger:
    """Return the appropriate logger related to the module that
    logs.

    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    logger = logging.getLogger(module.__name__)  # type: ignore
    return logger


def get_trial_dir() -> str:
    """Get the output path used by the currently active trial.

    Returns
    -------
    str
        The output path

    """
    return logging.root._log_dir  # type: ignore


def log(tag: str,
        data: ValueT,
        global_step: int,
        walltime: Optional[float] = None) -> None:
    """Log data to tensorboard and console (convenience function)

    Inspects type of data and uses the appropriate wrapper for
    tensorboard to consume the data. Supports floats (scalar),
    dictionary mapping tags to gloats (scalars), and strings (text).

    Parameters
    ----------
    tag : str
        Name of data, used as the tensorboard tag
    data : ValueT
        The scalar or text to log
    global_step : int
        Iteration number associated with data
    walltime : Optional[float]
        Walltime for data (the default is None).

    Examples
    -------
    Normally you would have to do the following to log a scalar
    >>> import logging; from flambe.logging import ScalarT
    >>> logger = logging.getLogger(__name__)
    >>> logger.info(ScalarT(tag, data, step, walltime))
    But this method allows you to write a more concise statement with
    a common interface
    >>> from flambe.logging import log
    >>> log(tag, data, step)

    """
    fn: Callable[..., Any]
    if isinstance(data, (float, int)) or isinstance(data, torch.Tensor):
        if isinstance(data, torch.Tensor):
            data = data.item()
        fn = log_scalar
    elif isinstance(data, dict):
        fn = log_scalars
    elif isinstance(data, str):
        fn = log_text
    else:
        _get_context_logger().info(f"{tag} #{global_step}: {data} (not logged to tensorboard)")
        return
    # Ignore type for complicated branching, fn could have a number of
    # different signatures
    fn(tag, data, global_step)  # type: ignore


def log_scalar(tag: str,
               data: float,
               global_step: int,
               walltime: Optional[float] = None,
               logger: Optional[logging.Logger] = None) -> None:
    """Log tensorboard compatible scalar value with common interface

    Parameters
    ----------
    tag : str
        Tensorboard tag associated with scalar data
    data : float
        Scalar float value
    global_step : int
        The global step or iteration number
    walltime : Optional[float]
        Current walltime, for example from `time.time()`
    logger: Optional[logging.Logger]
        logger to use for logging the scalar

    """
    logger = logger or _get_context_logger()
    logger.info(ScalarT(tag=tag, scalar_value=data, global_step=global_step,
                        walltime=walltime or time.time()))


def log_scalars(tag: str,
                data: Dict[str, float],
                global_step: int,
                walltime: Optional[float] = None,
                logger: Optional[logging.Logger] = None) -> None:
    """Log tensorboard compatible scalar values with common interface

    Parameters
    ----------
    tag : str
        Main tensorboard tag associated with all data
    data : Dict[str, float]
        Scalar float value
    global_step : int
        The global step or iteration number
    walltime : Optional[float]
        Current walltime, for example from `time.time()`
    logger: Optional[logging.Logger]
        logger to use for logging the scalar

    """
    logger = logger or _get_context_logger()
    logger.info(ScalarsT(main_tag=tag,
                         tag_scalar_dict=data,
                         global_step=global_step,
                         walltime=walltime or time.time()))


def log_text(tag: str,
             data: str,
             global_step: int,
             walltime: Optional[float] = None,
             logger: Optional[logging.Logger] = None) -> None:
    """Log tensorboard compatible text value with common interface

    Parameters
    ----------
    tag : str
        Tensorboard tag associated with data
    data : str
        Scalar float value
    global_step : int
        The global step or iteration number
    walltime : Optional[float]
        Current walltime, for example from `time.time()`
    logger: Optional[logging.Logger]
        logger to use for logging the scalar

    """
    logger = logger or _get_context_logger()
    logger.info(TextT(tag=tag, text_string=data, global_step=global_step,
                      walltime=walltime or time.time()))


def log_image(tag: str,
              data: str,
              global_step: int,
              walltime: Optional[float] = None,
              logger: Optional[logging.Logger] = None) -> None:
    """Log tensorboard compatible image value with common interface

    Parameters
    ----------
    tag : str
        Tensorboard tag associated with data
    data : str
        Scalar float value
    global_step : int
        The global step or iteration number
    walltime : Optional[float]
        Current walltime, for example from `time.time()`
    logger: Optional[logging.Logger]
        logger to use for logging the scalar

    """
    logger = logger or _get_context_logger()
    logger.info(ImageT(tag=tag, img_tensor=data, global_step=global_step,
                       walltime=walltime or time.time()))


def log_pr_curve(tag: str,
                 labels: Union[torch.Tensor, numpy.array],
                 predictions: Union[torch.Tensor, numpy.array],
                 global_step: int,
                 num_thresholds: int = 127,
                 walltime: Optional[float] = None,
                 logger: Optional[logging.Logger] = None) -> None:
    """Log tensorboard compatible image value with common interface

    Parameters
    ----------
    tag: str
        Data identifier
    labels: Union[torch.Tensor, numpy.array]
        Containing 0, 1 values
    predictions: Union[torch.Tensor, numpy.array]
        Containing 0<=x<=1 values. Needs to match labels size
    num_thresholds: int = 127
        The number of thresholds to evaluate. Max value allowed 127.
    weights: Optional[float] = None
        No description provided.
    global_step: int
        Iteration associated with this value
    walltime: float
        Wall clock time associated with this value
    logger: Optional[logging.Logger]
        logger to use for logging the scalar

    """
    logger = logger or _get_context_logger()
    logger.info(PRCurveT(tag=tag, labels=labels, predictions=predictions,
                         num_thresholds=num_thresholds, global_step=global_step,
                         walltime=walltime or time.time()))


def log_histogram(tag: str,
                  data: str,
                  global_step: int,
                  bins: str = 'auto',
                  walltime: Optional[float] = None,
                  logger: Optional[logging.Logger] = None) -> None:
    """Log tensorboard compatible image value with common interface

    Parameters
    ----------
    tag : str
        Tensorboard tag associated with data
    data : str
        Scalar float value
    global_step : int
        The global step or iteration number
    walltime : Optional[float]
        Current walltime, for example from `time.time()`
    logger: Optional[logging.Logger]
        logger to use for logging the scalar

    """
    logger = logger or _get_context_logger()
    logger.info(HistogramT(tag=tag, values=data, global_step=global_step, bins=bins,
                           walltime=walltime or time.time()))
