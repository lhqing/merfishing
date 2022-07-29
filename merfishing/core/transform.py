"""
Dealing with merfish data coordinates, convert between pixel and micron coordinates.

The key information is recorded in the manifest file located in the region_dir/images/manifest.json.
The micron_to_pixel_transform.csv file is located in the region_dir/images/micron_to_mosaic_pixel_transform.csv.
The csv file can be calculated from the manifest file.
The equation is:

# mosaicMicronsPerPixel = 0.1083333 # by default
# s = 1 / mosaicMicronsPerPixel
#
# micron_to_pixel_mat = np.float32(
#     [[s*1,   0,     -s*micronExtents[0]],
#      [0,     s*1,   -s*micronExtents[1]],
#      [0,     0,     1                 ]]
# )
# pixel_to_micron_mat = cv2.invert(micron_to_pixel_mat)
"""

import cv2
import numpy as np
import pandas as pd


def _value_to_3d(values):
    n_dims = len(values.shape)
    if n_dims == 1:
        # value is in [x, y]
        values = np.array([[values]])
    elif n_dims == 2:
        values = np.array([values])
    elif n_dims == 3:
        pass
    else:
        raise ValueError(f"Input values dimensionality wrong, values.shape is {values.shape}.")
    return values, n_dims


def _return_input_dims(values, n_dims):
    if n_dims == 1:
        # value is in [x, y]
        values = values[0, 0, :2]
    elif n_dims == 2:
        values = values[0, :, :2]
    else:
        # n_dims == 3
        pass
    return values


class MerfishTransform:
    """Transform MERFISH image pixel and micron coordinates."""

    def __init__(self, micron_to_pixel_transform_path):
        self._micron_to_pixel_trans_mat = pd.read_csv(micron_to_pixel_transform_path, sep=" ", header=None).values

        _, pixel_to_micron_trans_mat = cv2.invert(self._micron_to_pixel_trans_mat)
        self._pixel_to_micron_trans_mat = pixel_to_micron_trans_mat

    def pixel_to_micron_transform(self, values):
        """Convert coordinates from pixel to micron."""
        values = np.array(values).astype(np.float32)
        values, input_dims = _value_to_3d(values)
        values_3d = cv2.transform(values, self._pixel_to_micron_trans_mat)
        values = _return_input_dims(values_3d, input_dims)
        return values

    def micron_to_pixel_transform(self, values, do_round=True):
        """Convert coordinates from micron to pixel."""
        values = np.array(values)
        values, input_dims = _value_to_3d(values)
        values_3d = cv2.transform(values, self._micron_to_pixel_trans_mat)
        values = _return_input_dims(values_3d, input_dims)
        if do_round:
            values = np.round(values).astype(int)
        return values
