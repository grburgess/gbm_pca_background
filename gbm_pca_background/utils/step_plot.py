import numpy as np


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
