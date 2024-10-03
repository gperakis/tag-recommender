def generate_labels(bins: list[int]) -> list[str]:
    """
    Generates labels for bin ranges.

    For example:
    bins = [0, 10, 20, 30, 40]
    generate_labels(bins) → ['0-9', '10-19', '20-29', '30+']

    Parameters
    ----------
    bins : list[int]
        A list of bin edges.

    Returns
    -------
    list[str]
        A list of labels for each bin.
    """
    labels = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            labels.append(f"{bins[i]}+")
        else:
            labels.append(f"{bins[i]}-{bins[i + 1] - 1}")
    return labels
