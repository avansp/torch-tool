from torch.utils.data import random_split

__all__ = ["dataset_splitter"]


def dataset_splitter(dataset, split=None):

    n = len(dataset)

    if split is None:
        split = [0.7, 0.2, 0.1]
    assert len(split) == 3, "Split must contain ratios for [train, validate, test]"

    # we're going to split the dataset into training & test
    n_split = [int(item * n) for item in split]
    if sum(n_split) < n:
        n_split[-1] += n - sum(n_split)
    assert sum(n_split) == n, "Sum of split must equal to 1.0"

    return random_split(dataset, n_split)