from flambe.optim.scheduler import LambdaLR


class NoamScheduler(LambdaLR):
    """Linear warmup and then quadratic decay.

    Linearly increases the learning rate from 0 to 1 over
    `warmup` steps.
    Quadratically decreases the learning rate after.

    This scheduler is generally used after every training batch.

    """

    def __init__(self,
                 optimizer,
                 warmup: int,
                 d_model: int):
        """Initialize the NoamScheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Wrapped optimizer.
        warmup : int
            The number of linear warmup phases
        d_model : int, optional
            The index of last step. Default: -1

        """
        self.warmup = warmup
        self.d_model = d_model
        super().__init__(optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)  # type: ignore

    def lr_lambda(self, step: int) -> float:
        """Compue the learning rate factor.

        Parameters
        ----------
        step : int
            The current step. Could be training over
            validation steps.

        Returns
        -------
        float
            The output factor

        """
        if step == 0 and self.warmup == 0:
            return 1. / (self.d_model ** 0.5)
        else:
            if step > self.warmup:
                return 1. / (self.d_model ** 0.5) / (step ** 0.5)
            else:
                return step / (self.d_model ** 0.5) / (self.warmup ** 1.5)
