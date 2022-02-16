import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        # self.writer = tf.summary.create_file_writer(log_dir)
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, scalar, step):
        """Add scalar summary."""
        # print('sumary')
        # tf.summary.scalar(tag, value, step=step)
        # self.writer.flush()
        self.writer.add_scalars(
            'loss', scalar,
            global_step=step
        )