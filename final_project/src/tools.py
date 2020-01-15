import numpy as np
def prepare_batch(inputs):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    max_sequence_length = max(sequence_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length],
                                  dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    return inputs_batch_major, sequence_lengths

dev_num = 0
def dev_input_generator(x,y,batch_size):
    global dev_num
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        x_batch.append(x[dev_num])
        y_batch.append(y[dev_num])
        dev_num += 1
        if dev_num >= len(x):
            dev_num = 0
    return x_batch, y_batch


test_num = 0
def test_input_generator(x,y,batch_size):
    global test_num
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        x_batch.append(x[test_num])
        y_batch.append(y[test_num])
        test_num += 1
        if test_num >= len(x):
            test_num = 0
    return x_batch, y_batch

def batch_generator(x, y):
    while True:
        i = np.random.randint(0, len(x))
        yield [x[i], y[i]]

train_num = 0
def input_generator(x, y, batch_size):
    global train_num
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        x_batch.append(x[train_num])
        y_batch.append(y[train_num])
        train_num += 1
        if train_num >= len(x):
            train_num = 0
    return x_batch, y_batch





