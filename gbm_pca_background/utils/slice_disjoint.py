import numpy as np


def slice_disjoint(arr):
    """
    slice and array of indices into disjoint sets

    :param arr:
    """

    slices = []
    start_slice = arr[0]
    counter = 0

    for i in range(len(arr) - 1):
        if arr[i + 1] > (arr[i] + 1):
            end_slice = arr[i]
            slices.append([start_slice, end_slice])
            start_slice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if end_slice != arr[-1]:
        slices.append([start_slice, arr[-1]])
    return slices

