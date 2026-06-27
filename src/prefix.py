import numpy as np

def prefix_sum(signal):
    prefix = np.zeros(len(signal))
    prefix[0] = signal[0]

    for i in range(1, len(signal)):
        prefix[i] = prefix[i-1] + signal[i]

    return prefix


def range_sum(prefix, left, right):

    if left > right:
        return 0.0

    if left == 0:
        return prefix[right]

    return prefix[right] - prefix[left-1]


def interval_indices(phase, start, end):
    """
    Returns a list of intervals.
    Each interval is a tuple (left,right).
    """

    intervals = []

    if start >= 0 and end <= 1:

        left = np.searchsorted(
            phase,
            start,
            side="left"
        )

        right = (
            np.searchsorted(
                phase,
                end,
                side="right"
            ) - 1
        )

        intervals.append((left, right))


    elif start < 0:

        left1 = np.searchsorted(
            phase,
            1 + start,
            side="left"
        )

        right1 = len(phase) - 1

        left2 = 0

        right2 = (
            np.searchsorted(
                phase,
                end,
                side="right"
            ) - 1
        )

        intervals.append((left1, right1))
        intervals.append((left2, right2))


    else:

        left1 = np.searchsorted(
            phase,
            start,
            side="left"
        )

        right1 = len(phase) - 1

        left2 = 0

        right2 = (
            np.searchsorted(
                phase,
                end - 1,
                side="right"
            ) - 1
        )

        intervals.append((left1, right1))
        intervals.append((left2, right2))

    return intervals