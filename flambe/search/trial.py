import enum
from typing import Dict, Any, List, Optional


class Status(enum.Enum):
    """The set of valid status for a trial.

    - `CREATED`: the trial is ready for execution
    - `RUNNING`: the trial is currently executing
    - `PAUSED`: the trial was paused by the search algorithm
    - `RESUME`: the trial is ready to be resumed
    - `HAS_RESULT`: the trial has a new result
    - `TERMINATED`: the trial ran successfully
    - `ERROR`: the trial has an error

    """
    CREATED = 0
    RUNNING = 1
    PAUSED = 2
    RESUME = 3
    HAS_RESULT = 4
    TERMINATED = 5
    ERROR = 6


class Trial(object):
    """An abstract representation of a trial.

    A Trial object has three main attributes that can be read
    and modified, namely: `status`, `metrics` and `parameters`.

    """
    def __init__(self, params: Dict[str, Any]):
        """Initialize a Trial.

        Parameters
        ----------
        params : Dict[str, Any]
            Set of hyperparameters to use for this trial.

        """
        self.params = params
        self.status = Status.CREATED
        self.metrics: List[float] = []

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get the set of hyperparameters for this trial."""
        return self.params

    @property
    def best_metric(self) -> Optional[float]:
        """Get the best metric recorded so far."""
        return max(self.metrics) if self.metrics else None

    def get_metric(self, step: int) -> float:
        """Get the current trial results."""
        if self.metrics is None:
            raise ValueError("No metrics available.")
        elif step - 1 > len(self.metrics):
            raise ValueError(f"This trial only has {len(self.metrics)}, \
                             but step #{step} was requested")
        return self.metrics[step]

    def get_status(self) -> Status:
        """Get the current trial status."""
        return self.status

    def set_metric(self, metric: float, step: int = None):
        """Set the current trial results. #1 indexed."""
        if step and step < len(self.metrics):
            raise ValueError("Attempting to erase previous metric")
        elif step and step > len(self.metrics):
            raise ValueError(f"Step ({step}) is bigger than the expected \
                               step value ({len(self.metrics) + 1}).")
        self.metrics.append(metric)

    def set_status(self, status: Status):
        """Set the status of this trial."""
        self.status = status

    def is_running(self) -> bool:
        """Whether the trial is currently running."""
        return self.status == Status.RUNNING

    def has_result(self) -> bool:
        """Whether the trial has a new result to report."""
        return self.status == Status.HAS_RESULT

    def is_paused(self) -> bool:
        """Whether the trial is currently paused"""
        return self.status == Status.PAUSED

    def is_resuming(self) -> bool:
        """Whether the trial should resume execution."""
        return self.status == Status.RESUME

    def is_terminated(self) -> bool:
        """Whether the trial was terminated."""
        return self.status == Status.TERMINATED

    def is_error(self) -> bool:
        """Whether the trial ended with an error."""
        return self.status == Status.ERROR

    def is_created(self) -> bool:
        """Whether the trial was just created."""
        return self.status == Status.CREATED

    def set_running(self):
        """Sets the trial status to running."""
        self.status = Status.RUNNING

    def set_has_result(self):
        """Sets the trial status to has result."""
        self.status = Status.HAS_RESULT

    def set_paused(self):
        """Sets the trial status to pause."""
        self.status = Status.PAUSED

    def set_resume(self):
        """Sets the trial status to resume."""
        self.status = Status.RESUME

    def set_terminated(self):
        """Sets the trial status to terminated."""
        self.status = Status.TERMINATED

    def set_error(self):
        """Sets the trial status to error."""
        self.status = Status.ERROR

    def set_created(self):
        """Sets the trial status to created."""
        self.status = Status.CREATED

    def generate_name(self, prefix: str = '') -> str:
        """Generate a name for this parameter variant.

        Parameters
        ----------
        params: Any
            Parameter object to parse. Can be a sequence, mapping, or
            any other Python object.
        prefix: str, optional
            A prefix to add to the generated name.

        Returns
        -------
        str
            A string name to represent the variant when dumping results.

        """
        def helper(params):
            if isinstance(params, (list, tuple, set)):
                name = ",".join([helper(param) for param in params])
                name = f"[{name}]"
            elif isinstance(params, dict):
                name = '|'
                for param, value in params.items():
                    if isinstance(value, dict):
                        name += helper({f'{param}.{k}': v for k, v in value.items()})[1:]
                    else:
                        name += f'{param}={helper(value)}|'
            else:
                name = str(params)

        return prefix + helper(self.parameters)
