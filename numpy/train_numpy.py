import numpy as np
import utils_numpy


def hack(x):
    """
    A dummy numpy function.

    Parameters
    ----------
    x : numpy.ndarray
        Input images. Expect these to have the shape (N, C, H, W), where N is the
        number of batches (processed in parallel), C is the number of channels (= 3),
        and (H, W) the dimensions of the image (height and width).

    Returns
    -------
    numpy.ndarray
        The representation, which must be (N, C').
    """
    reprs = []
    # Append a few pixels here and there
    reprs.append(x[:, :, 32, 32].mean(-1))
    reprs.append(x[:, :, 16, 32].mean(-1))
    reprs.append(x[:, :, 32, 16].mean(-1))
    reprs.append(x[:, :, -16, 32].mean(-1))
    reprs.append(x[:, :, 32, -16].mean(-1))
    # Append some global statistics
    reprs.append(x.mean((1, 2, 3)))
    reprs.append(x.var((1, 2, 3)))
    # Make representation and return
    reprs = np.stack(reprs, axis=-1)
    return reprs


if __name__ == '__main__':
    utils_numpy.export_function(hack)
