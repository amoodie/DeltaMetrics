import abc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.collections as collections

# from scipy import sparse

from sandplover.section import BaseSection

from sandplover.plot import VariableSet


class BaseCore(BaseSection):
    """BaseCore.

    Passthrough class for type checking differentiation.
    """

    def __init__(self, core_type, *args, **kwargs):
        super().__init__(core_type, *args, **kwargs)

    def show(self, *args, **kwargs):
        # expand dims then pass to standard?
        pass


class VerticalCore(BaseCore):
    """A vertical core, 1d section.

    A vertical core through the Cube. Implemented as a subclass (special) of
    the BaseSection, via the BaseCore.
    """

    def __init__(self, *args, px, py, **kwargs):
        """Instantiate

        Parameters
        ----------
        px
        py
        length, optional
        """
        self._input_px = px
        self._input_py = py
        self._input_length = kwargs.pop("length", None)
        super().__init__("vertical_core", *args, **kwargs)

    def _compute_section_coords(self):
        """Compute the core index locations."""
        # determine the length needed
        if self._input_length is None:
            self._length = self._underlying.shape[0]
        else:
            self._length = self._input_length

        self._x = np.array([self._input_px])
        self._y = np.array([self._input_py])

        # consistent with sections for slicing, temp till refactor input varnames
        self._dim1_idx = np.array(
            [int(self._input_px)]
        )  # must have dims for proper slicing of underlying
        self._dim2_idx = np.array([int(self._input_py)])

        # self._z = np.arange(self.length, dtype=int)


class HorizontalCore(BaseCore):
    def __init__(self):
        raise NotImplementedError


class KickoffCore(BaseCore):
    def __init__(self):
        raise NotImplementedError


class Bed(object):

    def __init__(self, z_bottom, thickness, sand_frac):
        self.z_bottom = z_bottom
        self.thickness = thickness
        self.sand_frac = sand_frac

    def __repr__(self):
        return ", ".join((str(self.z_bottom), str(self.thickness), str(self.sand_frac)))


class Column(object):

    def __init__(
        self,
        core,
        z=None,
        sand_frac="sandfrac",
        sand_frac_thresh=0.05,
        thickness_thresh=0.00,
        priority="mean",
        **kwargs,
    ):

        if isinstance(core, VerticalCore):
            self._core = core
            self._base = self._core._z[0]
            self._z = self._core._z
            self._sand_frac_key = sand_frac
            self._sand_frac = self._core[self._sand_frac_key]
        elif isinstance(core, np.ndarray):
            if z is None:
                raise NotImplementedError(
                    "No default z implemented yet, must construct manually."
                )
            else:
                self._z = z
                self._base = self._z[0]
                self._sand_frac = xr.DataArray(
                    core, dims=["z", "s"], coords={"z": self._z, "s": [1]}
                )
        else:
            raise TypeError("Bad core type.")

        if priority == "mean":
            # THIS SHOULD BE A WEIGHTED MEAN!!
            self.merge_func = lambda x: np.mean(x)
        elif priority == "max":
            self.merge_func = lambda x: np.max(x)
        elif priority == "min":
            self.merge_func = lambda x: np.min(x)
        else:
            raise ValueError

        self.sand_frac_thresh = sand_frac_thresh
        self.thickness_thresh = thickness_thresh

        # clean up the core
        self._sand_frac.data[self._sand_frac.data < 0] = 0

        # change in grain size over each elevation
        dgs = np.diff(np.atleast_1d(self._sand_frac), prepend=0, axis=0)
        dgs[0] = np.inf  # force lower bound to be change point
        sed_surf_idx = np.argmax(self._sand_frac.data)
        dgs[sed_surf_idx] = np.inf  # force upper bound to be change point
        dgs_pts = np.where(np.abs(dgs) >= self.sand_frac_thresh)[0]  # idx of change pts

        # loop through and make every bed
        self.bed_list = []
        for i in range(len(dgs_pts) - 1):
            bed_start = float(self._z[dgs_pts[i]])
            bed_end = float(self._z[dgs_pts[i + 1]])
            gs_val = np.nanmean(self._sand_frac[dgs_pts[i] : dgs_pts[i + 1]])

            _b = Bed(bed_start, bed_end - bed_start, gs_val)
            self.bed_list.append(_b)
            bed_start = bed_end
            i += 1

        # now, purge out beds that are smaller than thresh, until none remain
        # NOTE: this is horribly inefficient right now. Needs to be optimized.
        too_small = [bb.thickness < self.thickness_thresh for bb in self.bed_list]
        nbeds_init = len(self.bed_list)
        lp = 0
        while np.any(too_small):
            # find the smallest and finest grain size
            thck = [bb.thickness for bb in self.bed_list]  # thickness list
            grsz = [bb.sand_frac for bb in self.bed_list]  # grain size list
            # sort by thickness, then by grain size
            ind = np.lexsort((grsz, thck))

            # make a new bed comprising the two outside beds
            #    - special handling for edges
            bed_purge = ind[0]
            if bed_purge == 0:
                new_bed = Bed(
                    self.bed_list[0].z_bottom,
                    self.bed_list[0].thickness + self.bed_list[1].thickness,
                    self.merge_func(
                        np.array(
                            [self.bed_list[0].sand_frac, self.bed_list[1].sand_frac]
                        )
                    ),
                )
                # reconstruct the bed list
                new_list = [new_bed, *self.bed_list[2:]]
            elif bed_purge == len(self.bed_list) - 1:
                new_bed = Bed(
                    self.bed_list[-2].z_bottom,
                    self.bed_list[-2].thickness + self.bed_list[-1].thickness,
                    self.merge_func(
                        np.array(
                            [self.bed_list[-2].sand_frac, self.bed_list[-1].sand_frac]
                        )
                    ),
                )
                # reconstruct the bed list
                new_list = [*self.bed_list[:-2], new_bed]
            else:
                new_bed_thickness = np.sum(
                    [b.thickness for b in self.bed_list[bed_purge - 1 : bed_purge + 2]]
                )
                new_bed_gsz = self.merge_func(
                    np.array(
                        [
                            self.bed_list[bed_purge - 1].sand_frac,
                            self.bed_list[bed_purge + 1].sand_frac,
                        ]
                    )
                )
                new_bed = Bed(
                    self.bed_list[bed_purge - 1].z_bottom,
                    new_bed_thickness,
                    new_bed_gsz,
                )
                # reconstruct the bed list
                keep_list = np.ones(len(self.bed_list), dtype=np.bool)
                keep_list[bed_purge - 1 : bed_purge + 2] = False
                new_list = []
                added = False
                for j in range(len(keep_list)):
                    if keep_list[j]:
                        new_list.append(self.bed_list[j])
                    else:
                        if not added:
                            new_list.append(new_bed)
                            added = True

            self.bed_list = new_list
            too_small = [bb.thickness < self.thickness_thresh for bb in self.bed_list]
            lp += 1
            if lp > (nbeds_init * 2):
                raise RuntimeError(
                    "Something went wrong, this is a safety from infinite while loop."
                )

        self._nbeds = len(self.bed_list)

    def build_patch_collection(self):
        """Patch collection for display."""
        gs_map = np.array([0, 1])
        fills = []
        gs_pts = []
        for iz in np.arange(self._nbeds):
            _b = self.bed_list[iz]
            _scale = (_b.sand_frac - 0) / (1 - 0)
            gs_pt = ((gs_map[1] - gs_map[0]) * _scale) + gs_map[0]
            gs_pts.append(gs_pt)

            fills.append(Rectangle((-0.25, _b.z_bottom), gs_pt + 0.25, _b.thickness))
            fills_PC = collections.PatchCollection(fills, edgecolors="k")
            fills_PC.set_array(np.array(gs_pts))
        return fills_PC

    def get_beds(self, var):
        """get a variable from each bed."""
        return np.array([getattr(bb, var) for bb in self.bed_list])

    def get_bedarray(self):
        """return the thickness and sand frac as an array."""
        return np.array(
            [
                [getattr(bb, "thickness"), getattr(bb, "sand_frac")]
                for bb in self.bed_list
            ]
        )

    def show(self, cmap=None, bed_label=None, ax=None):

        if ax is None:
            fig, ax = plt.subplots(dpi=DPI)

        pc = self.build_patch_collection()

        if cmap is None:
            cmap = VariableSet()["strata_sand_frac"].cmap
        pc.set_cmap(cmap)
        pc.set_clim([0, 1])

        if not (bed_label is None):
            [
                ax[2].text(
                    bb.sand_frac,
                    (bb.z_bottom + (bb.thickness / 2)),
                    np.round(getattr(bb, bed_label), 2),
                    va="center",
                )
                for bb in self.bed_list
            ]

        ax.add_collection(pc)
        ax.set_xlim((-0.25, 1.25))
        ax.set_ylim((self._base, self._z[-1]))

    def discretize(self, z=None):
        """Create a discrete array from the column.

        Data is defined at every value in z. If Z is None,
        we use `self._z`.
        """
        if z is None:
            z = self._z
        disc = np.zeros_like(z)
        dz = z[1] - z[0]
        nz = disc.shape[0]

        for b, bed in enumerate(self.bed_list):
            idx_base = int(bed.z_bottom / dz)
            idx_height = int(bed.thickness / dz)
            disc[idx_base : idx_base + idx_height] = bed.sand_frac
        return disc

    def make_bed_inventory(self, augment=False, augment_scale=0.0025):
        d0 = self.discretize()
        if augment:
            dj = d0 + np.random.normal(0, augment_scale, size=(len(d0),))
            span = 1
            dj = np.convolve(dj, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
            dj = np.clip(dj, 0, 1)
        else:
            dj = d0

        dz = self._z[1] - self._z[0]
        nbed = dj.shape[0]
        bed_thicks = np.zeros((nbed, 2))
        for b in np.arange(nbed - 1):
            bedval = dj[b]
            before_where = np.where(dj[:b] >= bedval)[0]
            after_where = np.where(dj[(b + 1) :] >= bedval)[0]
            if before_where.size > 0:
                before_thick = (b - before_where[-1]) * dz
            else:
                before_thick = np.nan
            if after_where.size > 0:
                after_thick = (after_where[0] + 1) * dz
            else:
                after_thick = np.nan
            bed_thicks[b, :] = np.array([before_thick, after_thick])
        return bed_thicks
