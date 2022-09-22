from torch.optim import Optimizer
import pytorch_lightning as pl


class LearningRateScheduler(object):
    r"""
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def set_lr(self, optimizer, factor):
        for idx, g in enumerate(optimizer.param_groups):
            g['lr'] = self.init_lr[idx] * factor

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class PolyLRScheduler(LearningRateScheduler, pl.LightningModule):
    r"""
    Transformer Learning Rate Scheduler proposed in "Attention Is All You Need"

    Args:
        optimizer (Optimizer): Optimizer.
        final_lr (float): Final learning rate.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
    """

    def __init__(
            self,
            optimizer: Optimizer,
            power: float,
            num_epochs: int,
            final_lr: float,
            warmup_steps: int,
            by_epoch=True
    ) -> None:
        self.stage = 0
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"

        super(PolyLRScheduler, self).__init__(optimizer, [pg["lr"] for pg in optimizer.param_groups])
        self.final_lr = final_lr  # final lr after decay
        self.warmup_steps = warmup_steps
        self.update_steps = 1
        self.power = power
        self.num_epochs = num_epochs
        self.start_epoch = 0

        self.step(0)  # start from 1/warmup_steps

    def _decide_stage(self):
        if self.update_steps <= self.warmup_steps:
            return 0
        else:
            return 1

    def step(self, epoch):
        self.stage = self._decide_stage()
        if self.stage == 0:
            warmup_rate = self.update_steps / self.warmup_steps
            self.set_lr(self.optimizer, warmup_rate)
            self.update_steps += 1
            if self.update_steps == self.warmup_steps:
                self.start_epoch = epoch
        elif self.stage == 1:  # start to decay with epoch
            decay_rate = (1 - (epoch - self.start_epoch) / (self.num_epochs - self.start_epoch)) ** self.power
            self.set_lr(self.optimizer, decay_rate)
        else:
            raise ValueError("Undefined stage")

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
