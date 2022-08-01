"""
Using the bigfish package to process smFISH image, call spots.

See bigfish documentation for details.
https://big-fish.readthedocs.io/en/stable/
"""

from bigfish.detection import decompose_dense, detect_spots
from bigfish.plot import plot_detection, plot_elbow, plot_reference_spot
from bigfish.stack import maximum_projection


def call_spot(
    image,
    detect_dense=True,
    voxel_size=(300, 103, 103),  # in nanometer (one value per dimension zyx)
    spot_radius=(350, 150, 150),
    dense_alpha=0.8,
    dense_beta=1,
    dense_gamma=5,
    plot=False,
    verbose=True,
):
    """
    Call spots on a 2D (y, x) or 3D (z, y, z) image.

    Parameters
    ----------
    image
        2D (y, x) or 3D (z, y, z) image.
    detect_dense
        Whether to decompose dense spots.
    voxel_size
        Voxel size in nanometer (one value per dimension yx or zyx). See bigfish.detection.detect_spots for details.
    spot_radius
        Spot radius in nanometer (one value per dimension yx or zyx). See bigfish.detection.detect_spots for details.
    dense_alpha
        alpha impacts the number of spots per candidate region. See bigfish.detection.decompose_dense for details.
    dense_beta
        beta impacts the number of candidate regions to decompose. See bigfish.detection.decompose_dense for details.
    dense_gamma
        gamma impacts the filtering step to denoise the image. See bigfish.detection.decompose_dense for details.
    plot
        Whether to plot the results.
    verbose
        Whether to print the results.
    """
    if len(image.shape) == 2:
        if len(voxel_size) == 3:
            voxel_size = voxel_size[1:]
        if len(spot_radius) == 3:
            spot_radius = spot_radius[1:]

    spots, threshold = detect_spots(
        images=image,
        return_threshold=True,
        voxel_size=voxel_size,  # in nanometer (one value per dimension zyx)
        spot_radius=spot_radius,
    )  # in nanometer (one value per dimension zyx)
    if verbose:
        print("detected spots")
        print(f"\r shape: {spots.shape}")
        print(f"\r dtype: {spots.dtype}")
        print(f"\r threshold: {threshold}")

    if plot:
        if len(image.shape) == 3:
            _plot_image = maximum_projection(image)
        else:
            _plot_image = image
        plot_detection(_plot_image, spots, contrast=True)
        plot_elbow(images=image, voxel_size=voxel_size, spot_radius=spot_radius)

    if detect_dense:
        try:
            spots_post_decomposition, dense_regions, reference_spot = decompose_dense(
                image=image,
                spots=spots,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
                alpha=dense_alpha,
                beta=dense_beta,
                gamma=dense_gamma,
            )
            if verbose:
                print("detected spots before decomposition")
                print(f"\r shape: {spots.shape}")
                print(f"\r dtype: {spots.dtype}", "\n")
                print("detected spots after decomposition")
                print(f"\r shape: {spots_post_decomposition.shape}")
                print(f"\r dtype: {spots_post_decomposition.dtype}")
            if plot:
                if len(image.shape) == 3:
                    _plot_image = maximum_projection(image)
                else:
                    _plot_image = image
                plot_detection(_plot_image, spots_post_decomposition, contrast=True)
                plot_reference_spot(reference_spot, rescale=True)
            return spots_post_decomposition, dense_regions, reference_spot
        except RuntimeError as e:
            print("decompose_dense failed. Return normal spots only.")
            print(e)
            return spots, None, None
    else:
        return spots
