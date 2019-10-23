from flambe.optim.scheduler import LambdaLR


class WarmupLinearScheduler(LambdaLR):
    """Linear warmup and then linear decay.

    Linearly increases learning rate from 0 to 1 over
    `warmup` training steps.
    Linearly decreases learning rate from 1. to 0. over
    remaining `n_steps - warmup` steps.

    This scheduler is generally used after every training batch.

    """

    def __init__(self,
                 optimizer,
                 warmup: int,
                 n_steps: int):
        """Initialize the WarmupLinearScheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Wrapped optimizer.
        warmup : int
            The number of linear warmup phases
        n_steps : int, optional
            The index of last step. Default: -1

        """
        self.warmup = warmup
        self.n_steps = n_steps
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
        if step < self.warmup:
            return float(step) / float(max(1, self.warmup))
        return max(0.0, float(self.n_steps - step) / float(max(1.0, self.n_steps - self.warmup)))
