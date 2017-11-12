import numpy as np

def mean_std_groups(x, y, group_size):
    num_groups = int(len(x) / group_size)

    x, x_tail = x[:group_size * num_groups], x[group_size * num_groups:]
    x = x.reshape((num_groups, group_size))

    y, y_tail = y[:group_size * num_groups], y[group_size * num_groups:]
    y = y.reshape((num_groups, group_size))

    x_means = x.mean(axis=1)
    x_stds = x.std(axis=1)

    if len(x_tail) > 0:
        x_means = np.concatenate([x_means, x_tail.mean(axis=0, keepdims=True)])
        x_stds = np.concatenate([x_stds, x_tail.std(axis=0, keepdims=True)])

    y_means = y.mean(axis=1)
    y_stds = y.std(axis=1)

    if len(y_tail) > 0:
        y_means = np.concatenate([y_means, y_tail.mean(axis=0, keepdims=True)])
        y_stds = np.concatenate([y_stds, y_tail.std(axis=0, keepdims=True)])

    return x_means, x_stds, y_means, y_stds
