"""Using the cellpose2 package to do nuclei and cell segmentation on DAPI and PolyT images."""

from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from cellpose import models, utils
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances_chunked

# Default microscope setting
MICRON_PER_PIXEL = 0.108
Z_SLICE_DISTANCE = 1.5
NUM_Z_SLICES = 7
VOXEL_VOLUME = MICRON_PER_PIXEL**2 * Z_SLICE_DISTANCE


def _connect_masks(records, buffer_size=15):
    def _filter_distance(_chunk):
        _x, _y = np.where(_chunk < buffer_size)
        not_self = _x != _y
        _x = _x[not_self]
        _y = _y[not_self]
        return _x, _y

    xs = []
    ys = []
    for chunk in pairwise_distances_chunked(records[["center_x", "center_y"]]):
        x, y = _filter_distance(chunk)
        xs.append(x)
        ys.append(y)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    graph = coo_matrix((np.ones_like(xs), (xs, ys)), shape=(records.shape[0], records.shape[0]), dtype="bool")
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    # label should start from 1, because 0 means no feature
    labels += 1
    return labels


def _cellpose_plot(image, masks, flows, z=0):
    import matplotlib.pyplot as plt
    from bigfish.stack import maximum_projection

    mask = masks[z]
    flow = flows[z]
    # maximum projection z axis of each channel
    _plot_image = np.array([maximum_projection(image[..., i]) for i in range(image.shape[-1])]).transpose([1, 2, 0])

    # noinspection PyArgumentList
    _plot_image = _plot_image / _plot_image.max()
    fig, axes = plt.subplots(figsize=(12, 3), ncols=4, dpi=200)

    ax = axes[0]
    ax.imshow(_plot_image)
    ax.set_title("Input Image")

    ax = axes[1]
    ax.imshow(_plot_image)
    outlines = utils.outlines_list(mask)
    # plot image with outlines overlaid in red
    for o in outlines:
        ax.plot(o[:, 0], o[:, 1], color="r", linewidth=0.5)
    ax.set_title(f"Cellpose Mask Outlines z={z}")

    ax = axes[2]
    ax.imshow(flow[2], cmap="RdBu_r")
    ax.set_title(f"Cell Probability z={z}")

    ax = axes[3]
    ax.imshow(flow[0])
    ax.set_title(f"Cell Flow z={z}")
    return


def _generate_features(mask, buffer_pixel_size) -> Tuple[pd.DataFrame, dict]:
    # single mask records, each mask is at single z plane
    mask_records = []
    for z in range(mask.shape[0]):
        mask_2d = mask[z]
        for mask_id in range(1, mask_2d.max() + 1):
            bool_mask = mask_2d == mask_id
            y, x = np.where(bool_mask)
            record = pd.Series(
                {
                    "z": z,
                    "center_x": x.mean(),
                    "center_y": y.mean(),
                    "min_x": x.min(),
                    "min_y": y.min(),
                    "max_x": x.max(),
                    "max_y": y.max(),
                    "volume": bool_mask.sum() * VOXEL_VOLUME,
                },
                name=f"{z}_{mask_id}",
            )
            mask_records.append(record)
    mask_records = pd.DataFrame(mask_records)

    # merge close mask to features
    mask_records["feature_id"] = _connect_masks(mask_records, buffer_size=buffer_pixel_size)
    feature_records = mask_records.groupby("feature_id").agg(
        {
            "center_x": "mean",
            "center_y": "mean",
            "min_x": "min",
            "min_y": "min",
            "max_x": "max",
            "max_y": "max",
            "volume": "sum",
            "z": lambda i: i.unique().size,
        }
    )

    mask_to_feature_map = mask_records["feature_id"].to_dict()
    return feature_records, mask_to_feature_map


def _generate_feature_mask(mask: np.ndarray, mask_feature_map: dict):
    feature_mask = np.zeros_like(mask)
    for mask_id, feature_id in mask_feature_map.items():
        z, mask_idx = mask_id.split("_")
        feature_mask[int(z)][mask[int(z)] == int(mask_idx)] = feature_id
    return feature_mask


def run_cellpose(
    image: np.ndarray,
    model_type,
    diameter: int,
    gpu=False,
    channels: list = None,
    channel_axis: int = 3,
    z_axis: int = 0,
    buffer_pixel_size: int = 15,
    plot: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Run cell pose for cell segmentation.

    Parameters
    ----------
    image : np.ndarray
        Input image stack for segmentation.
    model_type : str
        Type of segmentation (nuclei or cyto)
    diameter : int
        Average diameter for features
    gpu : bool
        A bool variable indicates whether to use GPU
    channels : list
        list of channels, either of length 2 or of length number of images by 2.
        First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
        For instance, to segment grayscale images, input [0,0]. To segment images with cells
        in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
        image with cells in green and nuclei in blue, input [[0,0], [2,3]].
    channel_axis : int
        The channel axis of the input images.
    z_axis : int
        The z axis of the input images.
    buffer_pixel_size : int
        pixel buffer size to merge masks across z planes together into feature
    plot : bool
        set to True to plot segmentation results
    verbose : bool
        set to True to print segmentation results

    Returns
    -------
    mask: single 3D array
        labelled image, where 0=no masks; 1,2,...=mask labels
    feature_meta: pd.DataFrame
        feature metadata including centroid, bbox, volume, # of z planes
    """
    model = models.Cellpose(gpu=gpu, model_type=model_type)
    if verbose:
        print(f"Running Cellpose {model_type} model")

    masks, flows, styles, diams = model.eval(
        [image.take(i, axis=z_axis) for i in range(image.shape[z_axis])],
        diameter=diameter,
        do_3D=False,
        channels=channels,
        channel_axis=channel_axis,
    )
    mask = np.array(masks)  # mask.shape = (z, y, x)
    if verbose:
        # noinspection PyArgumentList
        print(f"Cellpose generated {mask.max(axis=(1, 2)).sum()} masks")

    if plot:
        # make cellpose diagnostic plots
        _cellpose_plot(image, masks, flows, z=0)

    # call features
    feature_meta, mask_to_feature_map = _generate_features(mask, buffer_pixel_size)
    feature_mask = _generate_feature_mask(mask, mask_to_feature_map)
    if verbose:
        print(f"Cellpose generated {feature_meta.shape[0]} features")
    return feature_mask, feature_meta


def outlines_list_3d(mask_3d: np.ndarray, feature_ids=None, transform_func=None, as_polygon=False) -> dict:
    """
    Get outlines of masks as a list to loop over for plotting.

    Parameters
    ----------
    mask_3d : np.ndarray
        3D mask array.
    feature_ids : list
        list of feature ids to plot, if None, plot all features in mask.
    transform_func : callable
        function to transform outline coordinates (e.g., from local pixel to global microns).
    as_polygon : bool
        set to True to return outlines as polygons.

    Returns
    -------
    outlines : dict
        dictionary of outlines, keyed by feature id.
    """
    import cv2
    from shapely.geometry import Polygon

    if feature_ids is None:
        # get all features in mask
        feature_ids = np.unique(mask_3d)[1:]

    outlines = defaultdict(dict)
    for z in range(mask_3d.shape[0]):
        mask_2d: np.ndarray = mask_3d[z]
        for n in feature_ids:
            mn = mask_2d == n
            if mn.sum() > 0:
                contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                contours = contours[-2]
                cmax = np.argmax([c.shape[0] for c in contours])
                pix = contours[cmax].astype(int).squeeze()

                # skip small contours
                if len(pix) <= 4:
                    continue

                # coordinates transformation
                if transform_func is not None:
                    pix = transform_func(pix)

                # convert to shapely polygon
                if as_polygon:
                    pix = Polygon(pix)

                outlines[n][z] = pix
    return outlines
