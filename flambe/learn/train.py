import math
from typing import Dict, List, Optional, Any, Tuple, Iterator, Iterable, Union

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from flambe.dataset import Dataset
from flambe.compile import Schema, State, Component, Link
from flambe.learn.utils import select_device
from flambe.nn import Module  # type: ignore[attr-defined]
from flambe.sampler import Sampler
from flambe.metric import Metric
from flambe.logging import log


class Trainer(Component):
    """Implement a Trainer block.

    A `Trainer` takes as input data, model and optimizer,
    and executes training incrementally in `run`.

    Note that it is important that a trainer run be long enough
    to not increase overhead, so at least a few seconds, and ideally
    multiple minutes.

    """

    def __init__(self,
                 dataset: Dataset,
                 train_sampler: Sampler,
                 val_sampler: Sampler,
                 model: Module,
                 loss_fn: Metric,
                 metric_fn: Metric,
                 optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 iter_scheduler: Optional[_LRScheduler] = None,
                 device: Optional[str] = None,
                 max_steps: int = 10,
                 epoch_per_step: float = 1.0,
                 iter_per_step: Optional[int] = None,
                 batches_per_iter: int = 1,
                 lower_is_better: bool = False,
                 max_grad_norm: Optional[float] = None,
                 max_grad_abs_val: Optional[float] = None,
                 extra_validation_metrics: Optional[Iterable[Metric]] = None,
                 extra_training_metrics: Optional[Iterable[Metric]] = None,
                 extra_training_metrics_log_interval: Optional[int] = None) \
            -> None:
        """Initialize an instance of Trainer

        Parameters
        ----------
        dataset : Dataset
            The dataset to use in training the model
        train_sampler : Sampler
            The sampler to use over training examples during training
        val_sampler : Sampler
            The sampler to use over validation examples
        model : Module
            The model to train
        loss_fn: Metric
            The loss function to use in training the model
        metric_fn: Metric
            The metric function to use in evaluation
        optimizer : torch.optim.Optimizer
            The optimizer to use
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            An optional learning rate scheduler to run after each step
        iter_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            An optional learning rate scheduler to run after each batch
            (i.e iteration)
        device: str, optional
            The device to use in the computation.
        max_steps : int, optional
            The maximum number of training steps to run
        epoch_per_step : float, optional
            Fraction of an epoch to perform in a single training step
            (i.e before a checkpoint.) Defaults to 1.
            Overridden by `iter_per_step`, if given.
        iter_per_step : int, optional
            Number of iterations to perform in a single training step.
            Overrides `epoch_per_step` if given.
        batches_per_iter : int, optional
            Number of batches to pass through the model before
            calling optimizer.step. Requires the sampler to have
            drop_last set to True. (default set to 1 so optimizer.step
            is called after every batch)
        lower_is_better : bool, optional
            If true, the lowest val metric is considered best,
            otherwise the highest. Defaults to False.
        max_grad_norm : float, optional
            Maximum Euclidean norm of gradient after clipping.
        max_grad_abs_val: float, optional
            Maximum absolute value of all gradient vector components
            after clipping.
        extra_validation_metrics: Optional[Iterable[Metric]]
            A dict with extra metrics to show in each step
            but which don't guide the training procedures
            (i.e model selection through early stopping)
            The key of the metric will be used for displaying
            the values in tensorboard. Only logged during eval
        extra_training_metrics: Optional[Iterable[Metric]]
            A dict with extra metrics to show in each step
            but which don't guide the training procedures
            (i.e model selection through early stopping)
            The key of the metric will be used for displaying
            the values in tensorboard. Only logged during train
        extra_training_metrics_log_interval: Optional[int]
            The interval during training to log the
            extra_training_metrics.
            Set to None to disable and only log at the end of an epoch
            For eval, the interval is _always_ the entire epoch.
        """
        self.dataset = dataset
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iter_scheduler = iter_scheduler
        self.lower_is_better = lower_is_better
        self.max_grad_norm = max_grad_norm
        self.max_grad_abs_val = max_grad_abs_val
        self.validation_metrics = extra_validation_metrics if \
            extra_validation_metrics is not None else []
        self.training_metrics = extra_training_metrics if \
            extra_training_metrics is not None else []
        self.extra_training_metrics_log_interval = -1 \
            if extra_training_metrics_log_interval is None \
            else extra_training_metrics_log_interval

        # By default, no prefix applied to tb logs
        self.tb_log_prefix = None

        # Select right device
        self.device = select_device(device)

        if (not getattr(self.train_sampler, 'drop_last', False) and batches_per_iter != 1):
            raise ValueError(f'batches_per_iter cannot be set to {batches_per_iter} '
                             'if the sampler does not have `drop_last` set to True')

        self.batches_per_iter = batches_per_iter
        n_batches = self.train_sampler.length(dataset.train)

        if iter_per_step is None:
            # Compute epoch per step
            if self.batches_per_iter > n_batches:
                raise Exception(f'Please set batches_per_iter ({self.batches_per_iter}) '
                                f'to be ≤ the length of your train_sampler '
                                f'({n_batches})')
            iter_per_epoch = n_batches // self.batches_per_iter
            iter_per_step = math.ceil(epoch_per_step * iter_per_epoch)
        else:
            # Iter per step takes precedent over epoch_per_step
            epoch_per_step = iter_per_step / n_batches

        self.iter_per_step = iter_per_step
        self.max_steps = max_steps

        self._step = 0
        self._best_metric: Union[float, None] = None
        self._last_train_log_step = 0
        self._best_model: Dict[str, torch.Tensor] = dict()
        self.register_attrs('_step', '_best_metric', '_best_model')

        self.n_epochs = math.ceil(epoch_per_step * max_steps)

        self._create_train_iterator()

    def _create_train_iterator(self):
        self._train_iterator = self.train_sampler.sample(self.dataset.train, self.n_epochs)

    def _batch_to_device(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Move the current batch on the correct device.

        Can be overriden if a batch doesn't follow the expected
        structure. For example if the batch is a dictionary.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, ...]
            The batch to train on.

        """
        batch = tuple(t.to(self.device) for t in batch)
        return batch

    def _compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the loss given a single batch
        DEPRECATED, only exists for legacy compatibility with custom
        trainers

        Parameters
        ----------
        batch: Tuple[torch.Tensor, ...]
            The batch to train on.

        """
        print('Warning: flambe.learn.train.Trainer._compute_loss is deprecated. '
              'Please use flambe.learn.train.Trainer._compute_batch in the future.')
        batch = self._batch_to_device(batch)
        pred, target = self.model(*batch)
        loss = self.loss_fn(pred, target)
        return loss

    def _compute_batch(self, batch: Tuple[torch.Tensor, ...],
                       metrics: List[Tuple] = []) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Computes a batch.

        Does a model forward pass over a batch, and returns prediction,
        target and loss.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, ...]
            The batch to train on.

        """
        batch = self._batch_to_device(batch)
        pred, target = self.model(*batch)
        for metric, state in metrics:
            metric.aggregate(state, pred, target)
        loss = self.loss_fn(pred, target)
        return pred, target, loss

    @staticmethod
    def _log_metrics(log_prefix: str,
                     metrics_with_states: List[Tuple],
                     global_step: int) -> None:
        """Logs all provided metrics

        Iterates through the provided list of metrics with states,
        finalizes the metric, and logs it.

        Parameters
        ----------
        log_prefix: str
            A string, such as a tensorboard prefix
        metrics_with_states: List[Tuple[Metric, Dict]]
            a list of metric-state tuples
        global_step: int
            the global step for loggin
        """
        for metric, state in metrics_with_states:
            log(f'{log_prefix}/{metric}', metric.finalize(state), global_step)

    def _train_step(self) -> None:
        """Run a training step over the training data."""
        self.model.train()
        metrics_with_states: List[Tuple] = [(metric, {}) for metric in self.training_metrics]
        self._last_train_log_step = 0

        log_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""
        log_prefix += 'Training'

        with torch.enable_grad():
            for i in range(self.iter_per_step):
                # Zero the gradients and clear the accumulated loss
                self.optimizer.zero_grad()
                accumulated_loss = 0.0
                for _ in range(self.batches_per_iter):
                    # Get next batch
                    try:
                        batch = next(self._train_iterator)
                    except StopIteration:
                        self._create_train_iterator()
                        batch = next(self._train_iterator)
                    batch = self._batch_to_device(batch)

                    # Compute loss
                    _, _, loss = self._compute_batch(batch, metrics_with_states)
                    accumulated_loss += loss.item() / self.batches_per_iter
                    loss.backward()

                # Log loss
                global_step = (self.iter_per_step * self._step) + i

                # Clip gradients if necessary
                if self.max_grad_norm:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.max_grad_abs_val:
                    clip_grad_value_(self.model.parameters(), self.max_grad_abs_val)

                log(f'{log_prefix}/Loss', accumulated_loss, global_step)
                log(f'{log_prefix}/Gradient_Norm', self.model.gradient_norm,
                    global_step)
                log(f'{log_prefix}/Parameter_Norm', self.model.parameter_norm,
                    global_step)

                # Optimize
                self.optimizer.step()

                # Update iter scheduler
                if self.iter_scheduler is not None:
                    learning_rate = self.iter_scheduler.get_lr()[0]  # type: ignore
                    log(f'{log_prefix}/LR', learning_rate, global_step)
                    self.iter_scheduler.step()  # type: ignore

                # Zero the gradients when exiting a train step
                self.optimizer.zero_grad()
                # logging train metrics
                if self.extra_training_metrics_log_interval > self._last_train_log_step:
                    self._log_metrics(log_prefix, metrics_with_states, global_step)
                    self._last_train_log_step = i
            if self._last_train_log_step != i:
                # log again at end of step, if not logged at the end of
                # step before
                self._log_metrics(log_prefix, metrics_with_states, global_step)

    def _aggregate_preds(self, data_iterator: Iterator) \
            -> Tuple[torch.Tensor, torch.Tensor, float]:
        """ DEPRECATED
        Aggregate the predicitons,
        targets and mean loss for the dataset.

        Parameters
        ----------
        data_iterator: Iterator
            Batches of data.

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, float]
            The predictions, targets and mean loss.

        DEPRECATED; only existed to aggregate for the metric functions.
        The metric functions do this in-place now.

        """
        preds, targets, loss = [], [], []
        for batch in data_iterator:
            pred, target, batch_loss = self._compute_batch(batch)
            loss.append(batch_loss.item())
            preds.append(pred.cpu())
            targets.append(target.cpu())
        loss = sum(loss) / len(loss)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        return preds, targets, loss

    def _eval_step(self) -> None:
        """Run an evaluation step over the validation data."""
        self.model.eval()
        metric_fn_state: Dict[Metric, Dict] = {}
        metrics_with_states: List[Tuple] = \
            [(metric, {}) for metric in self.validation_metrics]

        # Initialize a 1-epoch iteration through the validation set
        val_iterator = self.val_sampler.sample(self.dataset.val)

        with torch.no_grad():
            loss = []
            for batch in val_iterator:
                _, _, batch_loss = self._compute_batch(
                    batch, [(self.metric_fn, metric_fn_state), *metrics_with_states])
                loss.append(batch_loss.item())
            val_loss = np.NaN if loss == [] else sum(loss) / len(loss)
            val_metric = self.metric_fn.finalize(metric_fn_state)

        # Update best model
        sign = (-1)**(self.lower_is_better)
        if self._best_metric is None or (sign * val_metric > sign * self._best_metric):
            self._best_metric = val_metric
            best_model_state = self.model.state_dict()
            for k, t in best_model_state.items():
                best_model_state[k] = t.cpu().detach()
            self._best_model = best_model_state

        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                # torch's _LRScheduler.step DOES have a default value
                # so passing in no args is fine; it will automatically
                # compute the current epoch
                self.scheduler.step()  # type: ignore

        tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""

        # Log metrics
        log(f'{tb_prefix}Validation/Loss', val_loss, self._step)
        log(f'{tb_prefix}Validation/{self.metric_fn}', val_metric, self._step)
        log(f'{tb_prefix}Best/{self.metric_fn}',
            self._best_metric, self._step)  # type: ignore
        for (metric, state) in metrics_with_states:
            log(f'{tb_prefix}Validation/{metric}',
                metric.finalize(state), self._step)  # type: ignore

    def run(self) -> bool:
        """Evaluate and then train until the next checkpoint

        Returns
        ------
        bool
            Whether the component should continue running.

        """
        self._eval_step()
        if self._step < self.max_steps:
            self._train_step()

        # Simple stopping rule, if we exceed the max number of steps
        self._step += 1
        continue_ = self._step < self.max_steps
        if not continue_:
            self._eval_step()
            self.model.cpu()
            self.model.load_state_dict(self._best_model, strict=False)

        return continue_

    def metric(self) -> Optional[float]:
        """Override this method to enable scheduling.

        Returns
        -------
        float
            The metric to compare computable variants.

        """
        return self._best_metric

    def _state(self,
               state_dict: State,
               prefix: str,
               local_metadata: Dict[str, Any]) -> State:
        state_dict[prefix + 'optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict[prefix + 'scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state(self,
                    state_dict: State,
                    prefix: str,
                    local_metadata: Dict[str, Any],
                    strict: bool,
                    missing_keys: List[Any],
                    unexpected_keys: List[Any],
                    error_msgs: List[Any]) -> None:
        self.optimizer.load_state_dict(state_dict[prefix + 'optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict[prefix + 'scheduler'])
        # Useful when loading the model after training
        done = self._step >= self.max_steps
        if done:
            self.model.load_state_dict(self._best_model, strict=False)

    @classmethod
    def precompile(cls, **kwargs):
        """Override initialization.

        Ensure that the model is compiled and pushed to the right
        device before its parameters are passed to the optimizer.

        """
        # Select right device
        device = kwargs.get('device', None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        def move_to_device(obj: Any):
            if isinstance(obj, torch.nn.Module):
                obj.to(device)

        # Compile all objects and push Modules to the device
        for k, obj in kwargs.items():
            if isinstance(obj, (Schema, Link)):
                obj.post_init_hooks.append(move_to_device)
