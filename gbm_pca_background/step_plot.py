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


def step_plot(xbins, y, ax, fill=False, fill_min=0, **kwargs):
    """
    A doctored up step plot
    """

    if fill:

        x = []
        newy = []

        for (start, stop), v in zip(xbins, y):
            x.append(start)
            newy.append(v)
            x.append(stop)
            newy.append(v)

        ax.fill_between(x, newy, fill_min, **kwargs)

    else:

        # This supports a mask, so the line will not be drawn for missing bins

        new_x = []
        new_y = []

        for (x1, x2), y in zip(xbins, y):

            if len(new_x) == 0:

                # First iteration

                new_x.append(x1)
                new_x.append(x2)
                new_y.append(y)

            else:

                if x1 == new_x[-1]:

                    # This bin is contiguous to the previous one

                    new_x.append(x2)
                    new_y.append(y)

                else:

                    # This bin is not contiguous to the previous one
                    # Add a "missing bin"
                    new_x.append(x1)
                    new_y.append(np.nan)
                    new_x.append(x2)
                    new_y.append(y)

        new_y.append(new_y[-1])

        new_y = np.ma.masked_where(~np.isfinite(new_y), new_y)

        ax.step(new_x, new_y, where='post', **kwargs)
