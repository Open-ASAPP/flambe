from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from flambe.dataset import Dataset
from flambe.metric import Metric
from flambe.sampler import Sampler
from flambe.learn import Trainer
from flambe.nn import Module  # type: ignore[attr-defined]


class DistillationTrainer(Trainer):
    """Implement a Distillation Trainer.

    Perform knowledge distillation between a teacher and a student
    model. Note that the model outputs are expected to be raw logits.
    Make sure that you are not applying a softmax after the decoder.
    You can replace the traditional Decoder with a MLPEncoder.

    """

    def __init__(self,
                 dataset: Dataset,
                 train_sampler: Sampler,
                 val_sampler: Sampler,
                 teacher_model: Module,
                 student_model: Module,
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
                 extra_validation_metrics: Optional[List[Metric]] = None,
                 teacher_columns: Optional[Tuple[int, ...]] = None,
                 student_columns: Optional[Tuple[int, ...]] = None,
                 alpha_kl: float = 0.5,
                 temperature: int = 1,
                 unlabel_dataset: Optional[Dataset] = None,
                 unlabel_sampler: Optional[Sampler] = None) -> None:
        """Initialize the Trainer.

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the first N columns of data for the
            student model, and the last N columns for the target.
        train_sampler : Sampler
            The sampler to use over training examples
        val_sampler : Sampler
            The sampler to use over validation examples
        model : Module
            The model to train
        optimizer : torch.optim.Optimizer
            The optimizer to use
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            An optional learning rate scheduler
        iter_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            An optional learning rate scheduler to run after each batch
            (i.e iteration)
        device: str, optional
            The device to use in the computation. Only used by compile.
        max_steps : int, optional
            The maximum number of training steps to run
        epoch_per_step : float, optional
            Fraction of an epoch to perform in a single training step
            (i.e before a checkpoint.) Defaults to 1.
            Overriden by `iter_per_step`, if given.
        iter_per_step : int, optional
            Number of iterations to perform in a single training step.
            Overrides `epoch_per_step` if given.
        batches_per_iter : int, optional
            Number of batches to pass through the model before
            calling optimizer.step. Requires the sampler to have
            drop_last set to True. (default set to 1 so optimizer.step
            is called after every batch)
        lower_is_better : bool, optional
            If true, the lowest dev metric is considered best,
            otherwise the highest. Defaults to False.
        max_grad_norm : float, optional
            Maximum Euclidean norm of gradient after clipping.
        max_grad_abs_val: float, optional
            Maximum absolute value of all gradient vector components
            after clipping.
        extra_validation_metrics: Optional[List[Metric]]
            A list with extra metrics to show in each step
            but which don't guide the training procedures
            (i.e model selection through early stopping)
        alpha_kl: float, optional
            Weight applied to the distillation loss.
        temperature: int, optional
            The temperature applied to the logits
        unlabel_dataset: Dataset, optional
            Optional dataset of unlabel data
        unlabel_sampler: Sampler, optional
            Optional sampler over unlabel examples

        """
        super().__init__(dataset,
                         train_sampler,  # type: ignore
                         val_sampler,
                         student_model,
                         loss_fn,
                         metric_fn,
                         optimizer,
                         scheduler,
                         iter_scheduler,
                         device,
                         max_steps,
                         epoch_per_step,
                         iter_per_step,
                         batches_per_iter,
                         lower_is_better,
                         max_grad_norm,
                         max_grad_abs_val,
                         extra_validation_metrics)

        self.student_model = self.model
        self.teacher_model = teacher_model

        self.teacher_columns = teacher_columns
        self.student_columns = student_columns

        self.alpha_kl = alpha_kl
        self.temp = temperature

        self.unlabel_dataset = None
        self.unlabel_sampler = None
        if unlabel_sampler is not None and unlabel_dataset is not None:
            self.unlabel_sampler = unlabel_sampler
            self._unlabel_iterator = unlabel_sampler.sample(unlabel_dataset.train, -1)

    def _compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the loss for a single batch

        Important: the student and teacher output predictions must
        be the raw logits, so ensure that your decoder object is step
        with `take_log=False`.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, ...]
            The batch to train on

        Returns
        -------
        torch.Tensor
            The computed loss

        """
        student_columns = self.student_columns or range(len(batch))
        teacher_columns = self.teacher_columns or range(len(batch))

        student_batch = [batch[i] for i in student_columns]
        teacher_batch = [batch[i].detach() for i in teacher_columns]

        student_logits, student_target = self.student_model(*student_batch)

        with torch.no_grad():
            teacher_logits, _ = self.teacher_model(*teacher_batch)

        loss = torch.tensor(0.).to(self.device)
        student_pred = F.log_softmax(student_logits, dim=-1)

        if self.alpha_kl < 1.0:
            loss += (1 - self.alpha_kl) * self.loss_fn(student_pred, student_target)

        # Add unlabelled batch
        if self.unlabel_sampler is not None:
            # Get next batch
            unlabelled, = next(self._unlabel_iterator)

            student_unlabel_logits = self.student_model(unlabelled)
            teacher_unlabel_logits = self.teacher_model(unlabelled.detach())
            student_logits = torch.cat((student_logits, student_unlabel_logits))
            teacher_logits = torch.cat((teacher_logits, teacher_unlabel_logits))

        student_pred = F.log_softmax(student_logits / self.temp, dim=1)
        teacher_pred = F.softmax(teacher_logits / self.temp, dim=1)

        kl_loss = F.kl_div(student_pred, teacher_pred, size_average=False) / teacher_pred.shape[0]
        loss += (self.alpha_kl * self.temp**2) * kl_loss

        return loss

    def _aggregate_preds(self, data_iterator) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Aggregate the predicitons and targets for the dataset.

        Parameters
        ----------
        data_iterator: Iterator
            Batches of data

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
            The predictions, and targets

        """
        preds, targets = [], []

        for batch in data_iterator:
            student_columns = self.student_columns or range(len(batch))
            student_batch = [batch[i] for i in student_columns]

            pred, target = self.model(*[t.to(self.device) for t in student_batch])
            pred = F.log_softmax(pred, dim=-1)

            preds.append(pred.cpu())
            targets.append(target.cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        return preds, targets, 0.
