import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import deltametrics as dm


def compute_net_to_gross(CubeInstance, sand_frac='strata_sand_frac',
                         threshold_sand_frac=0.5):
    """Net-to-gross map from Cube.

    This is a basic routine to compute net-to-gross over a cube x-y domain.
    The routine determines the vertical proportion of "net" at each grid
    location, where net is anything with a higher sand fraction that
    `threshold_sand_frac`.

    Parameters
    ----------
    CubeInstance : :obj:`~deltametrics.cube.BaseCube` subclass instance
        Cube to compute net-to-gross.

    sand_frac : :obj:`str`, optional
        Which variable to use for sand fraction in the CubeInstance. Default
        value is 'strata_sand_frac'.

    threshold_sand_frac : :obj:`float`, optional
        Threshold value to determine "net" from sand fraction.

    """

    if not CubeInstance._knows_stratigraphy:
        raise AttributeError('CubeInstance must have computed stratigraphy.')

    warnings.warn('This routine is not tested, and should only be used for '
                  'the simplest cases of aggradational stratigraphy with '
                  'no subsidence. Only works for DataCube.')

    _var = CubeInstance[sand_frac]
    _gross_log = CubeInstance._psvd_idx
    _deposit_thickness = np.nanmax(CubeInstance._psvd_vxl_eta, axis=0)

    _psvd_data = CubeInstance[sand_frac][CubeInstance._psvd_idx]
    print("psvd data shape:", _psvd_data.shape)

    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(_deposit_thickness)

    # can probably vectorize this in the future
    _net_thickness = np.full(_deposit_thickness.shape, np.nan)
    for i in [10]:  #np.arange(_var.shape[1]):
        for j in [120]:  #np.arange(_var.shape[2]):
            # print(CubeInstance._psvd_vxl_eta[:,i,j])
            __bed_thickness = CubeInstance._psvd_vxl_eta[1:, i, j] - CubeInstance._psvd_vxl_eta[:-1, i, j]
            print(__bed_thickness)
            _i = CubeInstance._psvd_vxl_idx[:, i, j]
            print(_i)
            # _j = np.tile(np.arange(_i.shape[1]), (_i.shape[0], 1))
            _psvd_idx = CubeInstance._psvd_idx[:, i, j]
            print(_psvd_idx)
            _psvd_flld = CubeInstance._psvd_flld[:, i, j]
            print(_psvd_flld)

            __net_log = (_var[1:, i, j] > threshold_sand_frac)
            # print(__net_log.shape)
            _ret = __bed_thickness[__net_log[CubeInstance._psvd_idx[1:, i, j]]]
            # print(_ret.shape)
            _net_thickness[i, j] = np.nansum(_ret)
            # print(_net_thickness[i, j])

    ax[1].imshow(_net_thickness)
    # print("bed_thickness.shape", bed_thickness.shape)
    # _net = np.nansum(bed_thickness, axis=0)
    _gross = _deposit_thickness  # CubeInstance._psvd_flld[-1, ...]
    _gross[_gross < 1e-2] = np.nan

    ntg = _net_thickness / _gross
    ax[2].imshow(ntg)
    plt.show()

    return ntg


def compute_trajectory():
    """Show 1d profile at point.
    """
    pass


def compute_compensation(line1, line2):
    """Compute compensation statistic betwen two lines.

    Explain the stat.

    Parameters
    ----------
    line1 : :obj:`ndarray`
        First surface to use (two-dimensional matrix with x-z coordinates of
        line).

    line2 : :obj:`ndarray`
        Second surface to use (two-dimensional matrix with x-z coordinates of
        line).

    Returns
    -------
    CV : :obj:`float`
        Compensation statistic.

    """
    pass
