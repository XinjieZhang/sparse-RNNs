# coding utf-8
import numpy as np


def seq_iterator(raw_data, input_width, label_width, batch_size, noise=False):
    """
    Iterate on the raw return sequence data.
    Args:
    - raw_data: array
    - batch_size: int, the batch size.
    - num_steps: int, the number of unrolls.
    Yields:
    - Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
    - ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.float32)
    if len(raw_data.shape) == 1:
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        data = np.zeros([batch_size, batch_len], dtype=np.float32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

        epoch_size = (batch_len - label_width) // input_width

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*input_width:(i+1)*input_width]
            if noise:
                x = x + 0.001 * np.std(x) * np.random.rand(x.shape[0], x.shape[1])
            y = data[:, (i+1)*input_width:(i+1)*input_width+label_width]
            yield (x, y)

    else:
        data_len = raw_data.shape[0]
        data_dim = raw_data.shape[1]
        batch_len = data_len // batch_size
        data = np.zeros([batch_size, batch_len, data_dim], dtype=np.float32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1), :]

        epoch_size = (batch_len - label_width) // input_width

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*input_width: (i+1)*input_width, :]
            y = data[:, (i+1) * input_width: (i+1)*input_width+label_width, :]
            yield (x, y)
