from typing import Optional, Dict, Union

import torch

from flambe.compile import Component
from flambe.dataset import Dataset
from flambe.learn.utils import select_device
from flambe.nn import Module  # type: ignore[attr-defined]
from flambe.metric import Metric
from flambe.sampler import Sampler, BaseSampler
from flambe.logging import log


class Evaluator(Component):
    """Implement an Evaluator block.

    An `Evaluator` takes as input data, and a model and executes
    the evaluation. This is a single step `Component` object.

    """

    def __init__(self,
                 dataset: Dataset,
                 model: Module,
                 metric_fn: Metric,
                 eval_sampler: Optional[Sampler] = None,
                 eval_data: str = 'test',
                 device: Optional[str] = None) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        dataset : Dataset
            The dataset to run evaluation on
        model : Module
            The model to train
        metric_fn: Metric
            The metric to use for evaluation
        eval_sampler : Optional[Sampler]
            The sampler to use over validation examples. By default
            it will use `BaseSampler` with batch size 16 and without
            shuffling.
        eval_data: str
            The data split to evaluate on: one of train, val or test
        device: str, optional
            The device to use in the computation.

        """
        self.eval_sampler = eval_sampler or BaseSampler(batch_size=16, shuffle=False)
        self.model = model
        self.metric_fn = metric_fn
        self.dataset = dataset

        self.device = select_device(device)

        data = getattr(dataset, eval_data)
        self._eval_iterator = self.eval_sampler.sample(data)

        # By default, no prefix applied to tb logs
        self.tb_log_prefix = None

        self.eval_metric: Union[float, None] = None
        self.register_attrs('eval_metric')

    def run(self, block_name: str = None) -> bool:
        """Run the evaluation.

        Returns
        ------
        bool
            Whether the component should continue running.

        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            metric_state: Dict = {}

            for batch in self._eval_iterator:
                pred, target = self.model(*[t.to(self.device) for t in batch])
                self.metric_fn.aggregate(metric_state, pred, target)

            self.eval_metric = self.metric_fn.finalize(metric_state)

            tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""

            log(f'{tb_prefix}Eval/{self.metric_fn}',  # type: ignore
                self.eval_metric, global_step=0)  # type: ignore

        return False

    def metric(self) -> Optional[float]:
        """Override this method to enable scheduling.

        Returns
        -------
        float
            The metric to compare computable varients

        """
        return self.eval_metric
