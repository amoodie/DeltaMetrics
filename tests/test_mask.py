"""Tests for the mask.py script."""
import pytest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import unittest.mock as mock

from deltametrics import cube
from deltametrics import mask
from deltametrics.plan import OpeningAnglePlanform
from deltametrics.sample_data import _get_rcm8_path, _get_golf_path


rcm8_path = _get_rcm8_path()
rcm8cube = cube.DataCube(rcm8_path)

golf_path = _get_golf_path()
golfcube = cube.DataCube(golf_path)


@mock.patch.multiple(mask.BaseMask,
                     __abstractmethods__=set())
class TestBaseMask:
    """
    To test the BaseMask, we patch the base job with a filled abstract method
    `.run()`.

    .. note:: This patch is handled at the class level above!!
    """

    fake_input = np.ones((100, 200))

    @mock.patch('deltametrics.mask.BaseMask._set_shape_mask')
    def test_name_setter(self, patched):
        basemask = mask.BaseMask('somename', self.fake_input)
        assert basemask.mask_type == 'somename'
        patched.assert_called()  # this would change the shape
        assert basemask.shape is None  # so shape is not set
        assert basemask._mask is None  # so mask is not set

    def test_simple_example(self):
        basemask = mask.BaseMask('field', self.fake_input)

        # make a bunch of assertions
        assert np.all(basemask._mask == False)
        assert np.all(basemask.integer_mask == 0)
        assert basemask._mask is basemask.mask
        assert basemask.shape == self.fake_input.shape

    def test_show(self):
        """
        Here, we just test whether it works, and whether it takes a
        specific axis.
        """
        basemask = mask.BaseMask('field', self.fake_input)

        # test show with nothing
        basemask.show()
        plt.close()

        # test show with axes, bad values
        fig, ax = plt.subplots()
        basemask.show(ax=ax)
        plt.close()

    def test_no_data(self):
        """Test when no data input raises error."""
        with pytest.raises(ValueError, match=r'Expected 1 input, got 0.'):
            _ = mask.BaseMask('field')

    def test_invalid_data(self):
        """Test invalid data input."""
        with pytest.raises(TypeError, match=r'Input to mask instantiation *.'):
            _ = mask.BaseMask('field', 'a string!!')

    def test_return_empty(self):
        """Test when no data input, but allow empty, returns empty."""
        empty_basemask = mask.BaseMask('field', allow_empty=True)
        assert empty_basemask.mask_type == 'field'
        assert empty_basemask.shape is None
        assert empty_basemask._mask is None
        assert empty_basemask._mask is empty_basemask.mask

    def test_is_mask_deprecationwarning(self):
        """Test that TypeError is raised if is_mask is invalid."""
        with pytest.warns(DeprecationWarning):
            _ = mask.BaseMask('field', self.fake_input,
                              is_mask='invalid')
        with pytest.warns(DeprecationWarning):
            _ = mask.BaseMask('field', self.fake_input,
                              is_mask=True)

    def test_3dinput_deprecationerror(self):
        """Test that TypeError is raised if is_mask is invalid."""
        with pytest.raises(ValueError, match=r'Creating a `Mask` *.'):
            _ = mask.BaseMask('field', np.random.uniform(size=(10, 100, 200)))


class TestShorelineMask:
    """Tests associated with the mask.ShorelineMask class."""

    # define an input mask for the mask instantiation pathway
    _ElevationMask = mask.ElevationMask(
            rcm8cube['eta'][-1, :, :],
            elevation_threshold=0)

    def test_default_vals_array(self):
        """Test that instantiation works for an array."""
        # define the mask
        shoremask = mask.ShorelineMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert shoremask._input_flag == 'array'
        assert shoremask.mask_type == 'shoreline'
        assert shoremask.angle_threshold > 0
        assert shoremask._mask.dtype == np.bool

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not implemented pathway.')
    def test_default_vals_cube(self):
        """Test that instantiation works for an array."""
        # define the mask
        shoremask = mask.ShorelineMask(rcm8cube, t=-1)
        # make assertions
        assert shoremask._input_flag == 'cube'
        assert shoremask.mask_type == 'shoreline'
        assert shoremask.angle_threshold > 0
        assert shoremask._mask.dtype == np.bool

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not implemented pathway.')
    def test_default_vals_cubewithmeta(self):
        """Test that instantiation works for an array."""
        # define the mask
        shoremask = mask.ShorelineMask(golfcube, t=-1)
        # make assertions
        assert shoremask._input_flag == 'cube'
        assert shoremask.mask_type == 'shoreline'
        assert shoremask.angle_threshold > 0
        assert shoremask._mask.dtype == np.bool

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not implemented pathway.')
    def test_default_vals_mask(self):
        """Test that instantiation works for an array."""
        # define the mask
        shoremask = mask.ShorelineMask(self._ElevationMask)
        # make assertions
        assert shoremask._input_flag == 'mask'
        assert shoremask.mask_type == 'shoreline'
        assert shoremask.angle_threshold > 0
        assert shoremask._mask.dtype == np.bool

    def test_angle_threshold(self):
        """Test that instantiation works for an array."""
        # define the mask
        shoremask_default = mask.ShorelineMask(
            rcm8cube['eta'][-1, :, :])
        shoremask = mask.ShorelineMask(
            rcm8cube['eta'][-1, :, :],
            angle_threshold=45)
        # make assertions
        assert shoremask.angle_threshold == 45
        assert not np.all(shoremask_default == shoremask)

    def test_submergedLand(self):
        """Check what happens when there is no land above water."""
        # define the mask
        shoremask = mask.ShorelineMask(rcm8cube['eta'][0, :, :])
        # assert - expect all values to be False
        assert np.all(shoremask._mask == 0)


class TestLandMask:
    """Tests associated with the mask.LandMask class."""

    # define an input mask for the mask instantiation pathway
    _ElevationMask = mask.ElevationMask(
            golfcube['eta'][-1, :, :],
            elevation_threshold=0)

    _OAP_0 = OpeningAnglePlanform(
        golfcube['eta'][-1, :, :],
        elevation_threshold=0)
    _OAP_05 = OpeningAnglePlanform(
        golfcube['eta'][-1, :, :],
        elevation_threshold=0.5)

    def test_default_vals_array(self):
        """Test that instantiation works for an array."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert landmask._input_flag == 'array'
        assert landmask.mask_type == 'land'
        assert landmask.angle_threshold > 0
        assert landmask._mask.dtype == np.bool

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not implemented pathway.')
    def test_default_vals_cube(self):
        """Test that instantiation works for an array."""
        # define the mask
        landmask = mask.LandMask(rcm8cube, t=-1)
        # make assertions
        assert landmask._input_flag == 'cube'
        assert landmask.mask_type == 'land'
        assert landmask.angle_threshold > 0
        assert landmask._mask.dtype == np.bool

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not implemented pathway.')
    def test_default_vals_cubewithmeta(self):
        """Test that instantiation works for an array."""
        # define the mask
        landmask = mask.LandMask(golfcube, t=-1)
        # make assertions
        assert landmask._input_flag == 'cube'
        assert landmask.mask_type == 'land'
        assert landmask.angle_threshold > 0
        assert landmask._mask.dtype == np.bool

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not implemented pathway.')
    def test_default_vals_mask(self):
        """Test that instantiation works for an array."""
        # define the mask
        landmask = mask.LandMask(self._ElevationMask)
        # make assertions
        assert landmask._input_flag == 'mask'
        assert landmask.mask_type == 'land'
        assert landmask.angle_threshold > 0
        assert landmask._mask.dtype == np.bool

    def test_angle_threshold(self):
        """Test that instantiation works for an array."""
        # define the mask
        landmask_default = mask.LandMask(
            rcm8cube['eta'][-1, :, :])
        landmask = mask.LandMask(
            rcm8cube['eta'][-1, :, :],
            angle_threshold=45)
        # make assertions
        assert landmask.angle_threshold == 45
        assert not np.all(landmask_default == landmask)

    def test_submergedLand(self):
        """Check what happens when there is no land above water."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][0, :, :])
        # assert - expect all values to be False
        assert np.all(landmask._mask == 0)

    def test_static_from_OAP(self):
        landmask = mask.LandMask(golfcube['eta'][-1, :, :])
        mfOAP = mask.LandMask.from_OAP(self._OAP_0)

        landmask_05 = mask.LandMask(
            golfcube['eta'][-1, :, :],
            elevation_threshold=0.5)
        mfOAP_05 = mask.LandMask.from_OAP(self._OAP_05)

        assert np.all(landmask._mask == mfOAP._mask)
        assert np.all(landmask_05._mask == mfOAP_05._mask)

    def test_static_from_mask_ElevationMask(self):
        landmask = mask.LandMask(golfcube['eta'][-1, :, :])
        mfem = mask.LandMask.from_mask(self._ElevationMask)

        landmask_05 = mask.LandMask(
            golfcube['eta'][-1, :, :],
            elevation_threshold=0.5)

        assert np.all(landmask._mask == mfem._mask)
        assert np.sum(landmask_05.integer_mask) < np.sum(landmask.integer_mask)

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not implemented pathway.')
    def test_static_from_array(self):
        """Test that instantiation works for an array."""
        # define the mask
        landmask = mask.LandMask.from_array(np.ones((100, 200)))
        # make assertions
        assert landmask._input_flag == 'land'


class TestWetMask:
    """Tests associated with the mask.WetMask class."""

    def test_invalid_data(self):
        """Test invalid data input."""
        with pytest.raises(TypeError):
            wetmask = mask.WetMask('invalid')

    def test_default_vals(self):
        """Test that default values are assigned."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert wetmask.topo_threshold == -0.5
        assert wetmask.angle_threshold == 75
        assert wetmask.numviews == 3
        assert wetmask.mask_type == 'wet'

    def test_maskError(self):
        """Test that TypeError is raised if is_mask is invalid."""
        with pytest.raises(TypeError):
            wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                                   is_mask='invalid')

    def test_maskTrue(self):
        """Test that is_mask is True works."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                               is_mask=True)
        # do assertion
        assert np.all(wetmask.mask[-1, :, :] == rcm8cube['eta'][-1, :, :])

    def test_assign_vals(self):
        """Test that specified values are assigned."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                               topo_threshold=-1.0,
                               angle_threshold=100,
                               numviews=5)
        # make assertions
        assert wetmask.topo_threshold == -1.0
        assert wetmask.angle_threshold == 100
        assert wetmask.numviews == 5

    def test_land(self):
        """Check for important variables and the final mask."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert np.array_equal(wetmask.mask,
                              wetmask.mask.astype(bool)) is True
        assert hasattr(wetmask, 'oceanmap') is True
        assert hasattr(wetmask, 'mask') is True
        assert hasattr(wetmask, 'landmask') is True

    def test_givenland(self):
        """Test that a LandMask can be passed into it."""
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                               landmask=landmask)
        # make assertions
        assert hasattr(wetmask, 'landmask') is True
        assert hasattr(wetmask, 'oceanmap') is True
        assert hasattr(wetmask, 'mask') is True
        assert np.array_equal(wetmask.mask,
                              wetmask.mask.astype(bool)) is True

    def test_givenfakeland(self):
        """Test that a bad land mask doesn't break function."""
        landmask = 'not a mask'
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                               landmask=landmask)
        # make assertions
        assert hasattr(wetmask, 'landmask') is True
        assert hasattr(wetmask, 'oceanmap') is True
        assert hasattr(wetmask, 'mask') is True
        assert np.array_equal(wetmask.mask,
                              wetmask.mask.astype(bool)) is True

    def test_submerged(self):
        """Check what happens when there is no land above water."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][0, :, :])
        # assert - expect all values to be 0s
        assert np.all(wetmask.mask == 0)

    def test_submerged_givenland(self):
        """Check what happens when there is no land above water."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][0, :, :])
        wetmask = mask.WetMask(rcm8cube['eta'][0, :, :],
                               landmask=landmask)
        # assert - expect all values to be 0s
        assert np.all(wetmask.mask == 0)

    def test_3d(self):
        """Test with multiple time slices."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][30:33, :, :])
        # assert the shape
        assert np.shape(wetmask.mask) == (3, 120, 240)


class TestChannelMask:
    """Tests associated with the mask.ChannelMask class."""

    def test_invalid_data(self):
        """Test invalid data input."""
        with pytest.raises(TypeError):
            channelmask = mask.ChannelMask('invalid-velocity', 'invalid-topo')

    def test_default_vals(self):
        """Test that default values are assigned."""
        # define the mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :])
        # make assertions
        assert channelmask.velocity_threshold == 0.3
        assert channelmask.topo_threshold == -0.5
        assert channelmask.angle_threshold == 75
        assert channelmask.numviews == 3
        assert channelmask.mask_type == 'channel'

    def test_maskError(self):
        """Test that TypeError is raised if is_mask is invalid."""
        with pytest.raises(TypeError):
            channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                           rcm8cube['eta'][-1, :, :],
                                           is_mask='invalid')

    def test_maskTrue(self):
        """Test that is_mask is True works."""
        # define the mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :],
                                       is_mask=True)
        # do assertion
        assert np.all(channelmask.mask[-1, :, :] == rcm8cube['eta'][-1, :, :])

    def test_assign_vals(self):
        """Test that specified values are assigned."""
        # define the mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :],
                                       velocity_threshold=0.5,
                                       topo_threshold=-1.0,
                                       angle_threshold=100,
                                       numviews=5)
        # make assertions
        assert channelmask.velocity_threshold == 0.5
        assert channelmask.topo_threshold == -1.0
        assert channelmask.angle_threshold == 100
        assert channelmask.numviews == 5

    def test_imp_vars(self):
        """Check for important variables and the final mask."""
        # define the mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :])
        # make assertions
        assert np.array_equal(channelmask.mask,
                              channelmask.mask.astype(bool)) is True
        assert hasattr(channelmask, 'mask') is True
        assert hasattr(channelmask, 'landmask') is True
        assert hasattr(channelmask, 'flowmap') is True
        assert hasattr(channelmask, 'velocity') is True

    def test_givenland(self):
        """Test that a LandMask can be passed into it."""
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :],
                                       landmask=landmask)
        # make assertions
        assert hasattr(channelmask, 'landmask') is True
        assert hasattr(channelmask, 'oceanmap') is True
        assert hasattr(channelmask, 'mask') is True
        assert hasattr(channelmask, 'velocity') is True
        assert hasattr(channelmask, 'flowmap') is True
        assert np.array_equal(channelmask.mask,
                              channelmask.mask.astype(bool)) is True

    def test_givenfakeland(self):
        """Test that an improperly defined land mask still works."""
        landmask = 'not a mask'
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :],
                                       landmask=landmask)
        # make assertions
        assert hasattr(channelmask, 'landmask') is True
        assert hasattr(channelmask, 'oceanmap') is True
        assert hasattr(channelmask, 'mask') is True
        assert hasattr(channelmask, 'velocity') is True
        assert hasattr(channelmask, 'flowmap') is True
        assert np.array_equal(channelmask.mask,
                              channelmask.mask.astype(bool)) is True

    def test_givenwet(self):
        """Test that a WetMask can be passed into it."""
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :])
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :],
                                       wetmask=wetmask)
        # make assertions
        assert hasattr(channelmask, 'landmask') is True
        assert hasattr(channelmask, 'mask') is True
        assert hasattr(channelmask, 'velocity') is True
        assert hasattr(channelmask, 'flowmap') is True
        assert np.array_equal(channelmask.mask,
                              channelmask.mask.astype(bool)) is True

    def test_givenfakewet(self):
        """Test that an improperly defined wet mask still works."""
        wetmask = 'not a mask'
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :],
                                       wetmask=wetmask)
        # make assertions
        assert hasattr(channelmask, 'landmask') is True
        assert hasattr(channelmask, 'mask') is True
        assert hasattr(channelmask, 'velocity') is True
        assert hasattr(channelmask, 'flowmap') is True
        assert np.array_equal(channelmask.mask,
                              channelmask.mask.astype(bool)) is True

    def test_submerged(self):
        """Check what happens when there is no land above water."""
        # define zeros velocity array
        velocity = np.zeros_like(rcm8cube['velocity'][0, :, :].__array__())
        # define the mask
        channelmask = mask.ChannelMask(velocity,
                                       rcm8cube['eta'][0, :, :])
        # assert - expect all values to be 0s
        assert np.all(channelmask.mask == 0)

    def test_submerged_givenland(self):
        """Check what happens when there is no land above water."""
        # define zeros velocity array
        velocity = np.zeros_like(rcm8cube['velocity'][0, :, :].__array__())
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][0, :, :])
        channelmask = mask.ChannelMask(velocity,
                                       rcm8cube['eta'][0, :, :],
                                       landmask=landmask)
        # assert - expect all values to be 0s
        assert np.all(channelmask.mask == 0)

    def test_submerged_givenwet(self):
        """Check what happens when there is no land above water."""
        # define zeros velocity array
        velocity = np.zeros_like(rcm8cube['velocity'][0, :, :].__array__())
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][0, :, :])
        channelmask = mask.ChannelMask(velocity,
                                       rcm8cube['eta'][0, :, :],
                                       wetmask=wetmask)
        # assert - expect all values to be 0s
        assert np.all(channelmask.mask == 0)

    def test_invalidvelocity(self):
        """Raise TypeError if invalid velocity type is provided."""
        with pytest.raises(TypeError):
            channelmask = mask.ChannelMask('bad_velocity',
                                           rcm8cube['eta'][-1, :, :])

    def test_3d(self):
        """Test with multiple time slices."""
        # define the mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][30:33, :, :],
                                       rcm8cube['eta'][30:33, :, :])
        # assert the shape
        assert np.shape(channelmask.mask) == (3, 120, 240)


class TestEdgeMask:
    """Tests associated with the mask.EdgeMask class."""

    def test_invalid_data(self):
        """Test invalid data input."""
        with pytest.raises(TypeError):
            edgemask = mask.EdgeMask('invalid')

    def test_default_vals(self):
        """Test that default values are assigned."""
        # define the mask
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert edgemask.topo_threshold == -0.5
        assert edgemask.angle_threshold == 75
        assert edgemask.numviews == 3
        assert edgemask.mask_type == 'edge'

    def test_maskError(self):
        """Test that TypeError is raised if is_mask is invalid."""
        with pytest.raises(TypeError):
            edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :],
                                     is_mask='invalid')

    def test_maskTrue(self):
        """Test that is_mask is True works."""
        # define the mask
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :],
                                 is_mask=True)
        # do assertion
        assert np.all(edgemask.mask[-1, :, :] == rcm8cube['eta'][-1, :, :])

    def test_assign_vals(self):
        """Test that specified values are assigned."""
        # define the mask
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :],
                                 topo_threshold=-1.0,
                                 angle_threshold=100,
                                 numviews=5)
        # make assertions
        assert edgemask.topo_threshold == -1.0
        assert edgemask.angle_threshold == 100
        assert edgemask.numviews == 5

    def test_imp_vars(self):
        """Check for important variables and the final mask."""
        # define the mask
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert np.array_equal(edgemask.mask,
                              edgemask.mask.astype(bool)) is True
        assert hasattr(edgemask, 'mask') is True
        assert hasattr(edgemask, 'landmask') is True
        assert hasattr(edgemask, 'wetmask') is True

    def test_givenland(self):
        """Test that a LandMask can be passed into it."""
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :],
                                 landmask=landmask)
        # make assertions
        assert hasattr(edgemask, 'landmask') is True
        assert hasattr(edgemask, 'wetmask') is True
        assert hasattr(edgemask, 'mask') is True
        assert np.array_equal(edgemask.mask,
                              edgemask.mask.astype(bool)) is True

    def test_givenwet(self):
        """Test that a WetMask can be passed into it."""
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :])
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :],
                                 wetmask=wetmask)
        # make assertions
        assert hasattr(edgemask, 'landmask') is True
        assert hasattr(edgemask, 'mask') is True
        assert hasattr(edgemask, 'wetmask') is True
        assert np.array_equal(edgemask.mask,
                              edgemask.mask.astype(bool)) is True

    def test_givenwetandland(self):
        """Test that a WetMask and LandMask can be passed into it."""
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :])
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :],
                                 landmask=landmask,
                                 wetmask=wetmask)
        # make assertions
        assert hasattr(edgemask, 'landmask') is True
        assert hasattr(edgemask, 'mask') is True
        assert hasattr(edgemask, 'wetmask') is True
        assert np.array_equal(edgemask.mask,
                              edgemask.mask.astype(bool)) is True

    def test_givenbadwetandland(self):
        """Test that a bad pair of wet and land masks can be passed in."""
        landmask = 'bad land mask'
        wetmask = 'bad wet mask'
        edgemask = mask.EdgeMask(rcm8cube['eta'][-1, :, :],
                                 landmask=landmask,
                                 wetmask=wetmask)
        # make assertions
        assert hasattr(edgemask, 'landmask') is True
        assert hasattr(edgemask, 'mask') is True
        assert hasattr(edgemask, 'wetmask') is True
        assert np.array_equal(edgemask.mask,
                              edgemask.mask.astype(bool)) is True

    def test_submerged(self):
        """Check what happens when there is no land above water."""
        # define the mask
        edgemask = mask.EdgeMask(rcm8cube['eta'][0, :, :])
        # assert - expect all values to be 0s
        assert np.all(edgemask.mask == 0)

    def test_submerged_givenland(self):
        """Check what happens when there is no land above water."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][0, :, :])
        edgemask = mask.EdgeMask(rcm8cube['eta'][0, :, :],
                                 landmask=landmask)
        # assert - expect all values to be 0s
        assert np.all(edgemask.mask == 0)

    def test_submerged_givenwet(self):
        """Check what happens when there is no land above water."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][0, :, :])
        edgemask = mask.EdgeMask(rcm8cube['eta'][0, :, :],
                                 wetmask=wetmask)
        # assert - expect all values to be 0s
        assert np.all(edgemask.mask == 0)

    def test_submerged_givenboth(self):
        """Check what happens when there is no land above water."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][0, :, :])
        wetmask = mask.WetMask(rcm8cube['eta'][0, :, :])
        edgemask = mask.EdgeMask(rcm8cube['eta'][0, :, :],
                                 wetmask=wetmask,
                                 landmask=landmask)
        # assert - expect all values to be 0s
        assert np.all(edgemask.mask == 0)

    def test_3d(self):
        """Test with multiple time slices."""
        # define the mask
        edgemask = mask.EdgeMask(rcm8cube['eta'][30:33, :, :])
        # assert the shape
        assert np.shape(edgemask.mask) == (3, 120, 240)


class TestCenterlineMask:
    """Tests associated with the mask.CenterlineMask class."""

    def test_default_vals(self):
        """Test that default values are assigned."""
        # define a numpy array to serve as channelmask
        channelmask = np.zeros((5, 5))
        channelmask[1:2, 1:2] = 1.
        # define the mask
        centerlinemask = mask.CenterlineMask(channelmask)
        # make assertions
        assert centerlinemask.method == 'skeletonize'

    def test_maskError(self):
        """Test that TypeError is raised if is_mask is invalid."""
        # define a numpy array to serve as channelmask
        channelmask = np.zeros((5, 5))
        channelmask[1:2, 1:2] = 1.
        with pytest.raises(TypeError):
            centerlinemask = mask.CenterlineMask(channelmask,
                                                 is_mask='invalid')

    def test_channelmaskError(self):
        """Test that TypeError is raised if channelmask is invalid."""
        # define a numpy array to serve as channelmask
        channelmask = 'invalid'
        with pytest.raises(TypeError):
            centerlinemask = mask.CenterlineMask(channelmask,
                                                 is_mask=False)

    def test_maskTrue(self):
        """Test that is_mask is True works."""
        # define a numpy array to serve as channelmask
        channelmask = np.zeros((5, 5))
        channelmask[1:2, 1:2] = 1.
        # define the mask
        centerlinemask = mask.CenterlineMask(channelmask,
                                             is_mask=True)
        # do assertion
        assert np.all(centerlinemask.mask[-1, :, :] == channelmask)

    def test_passChannelMask(self):
        """Test that a ChannelMask object can be passed in."""
        # define channel mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :])
        # define the mask
        centerlinemask = mask.CenterlineMask(channelmask)
        # make assertions - check that mask is binary
        assert np.array_equal(centerlinemask.mask,
                              centerlinemask.mask.astype(bool)) is True

    def test_submerged(self):
        """Check what happens when there is no land above water."""
        # define channelmask
        channelmask = np.zeros((5, 5))
        # define the mask
        centerlinemask = mask.CenterlineMask(channelmask)
        # assert - expect all values to be 0s
        assert np.all(centerlinemask.mask == 0)

    def test_3d(self):
        """Test with multiple time slices."""
        # define the mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][30:33, :, :],
                                       rcm8cube['eta'][30:33, :, :])
        centerlinemask = mask.CenterlineMask(channelmask)
        # assert the shape
        assert np.shape(centerlinemask.mask) == (3, 120, 240)

    @pytest.mark.xfail()
    def test_rivamapDefaults(self):
        """Test rivamap extraction of centerlines."""
        # define channel mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :])
        # define the mask
        centerlinemask = mask.CenterlineMask(channelmask,
                                             method='rivamap',
                                             is_mask=False)
        # do assertion
        assert centerlinemask.minScale == 1.5
        assert centerlinemask.nrScales == 12
        assert centerlinemask.nms_threshold == 0.1
        assert hasattr(centerlinemask, 'psi') is True
        assert hasattr(centerlinemask, 'nms') is True
        assert hasattr(centerlinemask, 'mask') is True

    @pytest.mark.xfail()
    def test_rivamapCustom(self):
        """Test rivamap extraction of centerlines with custom values."""
        # define channel mask
        channelmask = mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                       rcm8cube['eta'][-1, :, :])
        # define the mask
        centerlinemask = mask.CenterlineMask(channelmask,
                                             method='rivamap',
                                             is_mask=False,
                                             minScale=2.5,
                                             nrScales=10,
                                             nms_threshold=0.05)
        # do assertion
        assert centerlinemask.minScale == 2.5
        assert centerlinemask.nrScales == 10
        assert centerlinemask.nms_threshold == 0.05
        assert hasattr(centerlinemask, 'psi') is True
        assert hasattr(centerlinemask, 'nms') is True
        assert hasattr(centerlinemask, 'mask') is True


class TestGeometricMask:
    """Tests associated with the mask.GeometricMask class."""

    def test_initialize_gm(self):
        """Test initialization."""
        arr = np.zeros((1, 10, 10))
        gmsk = mask.GeometricMask(arr)
        assert gmsk.mask_type == 'geometric'
        assert np.shape(gmsk._mask) == np.shape(arr)
        assert np.all(gmsk._mask == 1)
        assert gmsk._xc == 0
        assert gmsk._yc == 5
        assert gmsk.xc == gmsk._xc
        assert gmsk.yc == gmsk._yc

    def test_initialize_wrong_ismask(self):
        """Test with bad input mask."""
        arr = np.zeros((1, 10, 10))
        arr[0, 1, 1] = 1
        arr[0, 0, 0] = 5  # make input non-binary
        with pytest.raises(TypeError):
            mask.GeometricMask(arr, is_mask='blah')

    def test_initialize_with_mask(self):
        """Test init with mask."""
        msk = np.zeros((1, 10, 10))
        msk[0, 0, :] = 1
        gmsk = mask.GeometricMask(msk, is_mask=True)
        assert np.all(gmsk._mask == msk)

    def test_circular_default(self):
        """Test circular mask with defaults, small case."""
        arr = np.zeros((1, 5, 5))
        gmsk = mask.GeometricMask(arr)
        gmsk.circular(1)
        assert gmsk._mask[0, 0, 2] == 0

    def test_circular_2radii(self):
        """Test circular mask with 2 radii, small case."""
        arr = np.zeros((1, 7, 7))
        gmsk = mask.GeometricMask(arr)
        gmsk.circular(1, 2)
        assert gmsk._mask[0, 0, 3] == 0
        assert np.all(gmsk._mask[0, :, -1] == 0)
        assert np.all(gmsk._mask[0, :, 0] == 0)
        assert np.all(gmsk._mask[0, -1, :] == 0)

    def test_circular_custom_origin(self):
        """Test circular mask with defined origin."""
        arr = np.zeros((1, 7, 7))
        gmsk = mask.GeometricMask(arr)
        gmsk.circular(1, 2, origin=(3, 3))
        assert gmsk._mask[0, 3, 3] == 0
        assert np.all(gmsk._mask == np.array([[[0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 1., 0., 0., 0.],
                                               [0., 0., 1., 1., 1., 0., 0.],
                                               [0., 1., 1., 0., 1., 1., 0.],
                                               [0., 0., 1., 1., 1., 0., 0.],
                                               [0., 0., 0., 1., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0.]]])
                      )
        assert gmsk.xc == 3
        assert gmsk.yc == 3

    def test_strike_one(self):
        """Test strike masking with one value."""
        arr = np.zeros((1, 7, 7))
        gmsk = mask.GeometricMask(arr)
        gmsk.strike(2)
        assert np.all(gmsk._mask[0, :2, :] == 0)
        assert np.all(gmsk._mask[0, 2:, :] == 1)

    def test_strike_two(self):
        """Test strike masking with two values."""
        arr = np.zeros((1, 7, 7))
        gmsk = mask.GeometricMask(arr)
        gmsk.strike(2, 4)
        assert np.all(gmsk._mask[0, :2, :] == 0)
        assert np.all(gmsk._mask[0, 2:4, :] == 1)
        assert np.all(gmsk._mask[0, 4:, :] == 0)

    def test_dip_one(self):
        """Test dip masking with one value."""
        arr = np.zeros((1, 7, 7))
        gmsk = mask.GeometricMask(arr)
        gmsk.dip(5)
        assert np.all(gmsk._mask[0, :, 1:-1] == 1)
        assert np.all(gmsk._mask[0, :, 0] == 0)
        assert np.all(gmsk._mask[0, :, -1] == 0)

    def test_dip_two(self):
        """Test dip masking with two values."""
        arr = np.zeros((1, 7, 7))
        gmsk = mask.GeometricMask(arr)
        gmsk.dip(2, 4)
        assert np.all(gmsk._mask[0, :, 0:2] == 0)
        assert np.all(gmsk._mask[0, :, 2:4] == 1)
        assert np.all(gmsk._mask[0, :, 4:] == 0)

    def test_angular_half(self):
        """Test angular mask over half of domain"""
        arr = np.zeros((1, 100, 200))
        gmsk = mask.GeometricMask(arr)
        theta1 = 0
        theta2 = np.pi/2
        gmsk.angular(theta1, theta2)
        # assert 1s half
        assert np.all(gmsk._mask[-1, :, :101] == 1)
        assert np.all(gmsk._mask[-1, :, 101:] == 0)

    def test_angular_bad_dims(self):
        """raise error."""
        arr = np.zeros((5, 5))
        gmsk = mask.GeometricMask(arr)
        with pytest.raises(ValueError):
            gmsk.angular(0, np.pi/2)


def test_plotter(tmp_path):
    """Test the show() function."""
    import matplotlib.pyplot as plt
    shore_mask = mask.ShorelineMask(rcm8cube['eta'][-1, :, :])
    shore_mask.show()
    plt.savefig(tmp_path / 'mask_fig.png')
    plt.close()
    # assert that figure was made
    assert os.path.isfile(tmp_path / 'mask_fig.png') is True


def test_plotter_error():
    """Test error for show() function."""
    msk = mask.BaseMask('bad', np.zeros((3, 3)))
    with pytest.raises(AttributeError):
        msk.show()


def test_init_OAM():
    """Initialize the OAM class."""
    oam = mask.OAM('test', np.zeros((3, 3)))
    # assertions
    assert oam.mask_type == 'test'
    assert np.all(oam.data == np.zeros((3, 3)))


def test_wrong_size():
    """Raise error if array with bad dimensions is used."""
    bad_arr = np.arange(0, 5)
    with pytest.raises(ValueError):
        mask.BaseMask('bad1D', bad_arr)
