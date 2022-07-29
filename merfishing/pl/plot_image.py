import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class MerfishImageAxesPlotter:
    """Plotting class for MERFISH mosaic image overlay spots."""

    def __init__(self, ax, image, boundaries=None, cells=None, transform=None, offset=None):
        self.ax = ax
        self._data = image
        self._boundaries = boundaries
        self._cells = cells

        if (boundaries is not None) or (cells is not None):
            if transform is None:
                raise ValueError("transform is required if boundaries or scatters are provided")
            if offset is None:
                raise ValueError("offset is required if boundaries or scatters are provided")

        self._transform = transform
        self._offset = offset

    def plot_image(self, aspect="auto", hue_range=0.8, **kwargs):
        """Plot image."""
        if hue_range:
            vmin = self._data.min()
            vmax = self._data.max() * hue_range
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        if len(self._data.shape) == 3 and self._data.shape[2] == 3:
            # RGB image, vmin, vmax has no effect
            # make the image more saturated
            self._data[self._data > hue_range] = hue_range
            self._data = self._data / hue_range
        self.ax.imshow(self._data, aspect=aspect, **kwargs)
        return

    def plot_boundaries(self, cmap="viridis", first_only=False, linewidth=0.5, **kwargs):
        """Plot boundaries line plot."""
        if self._boundaries is None:
            print("No boundaries to plot")
            return

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap).copy()
        xmin, ymin = self._offset

        for bd in self._boundaries.values():
            z_coords = bd.z_coords
            cnorm = Normalize(z_coords.min(), z_coords.max())
            for z, line in bd.items():
                if cmap is not None:
                    color = cmap(cnorm(bd.z_coords[int(z)]))
                    kwargs["color"] = color
                line = self._transform.micron_to_pixel_transform(line)
                x = line[0, :, 0]
                y = line[0, :, 1]
                self.ax.plot(x - xmin, y - ymin, linewidth=linewidth, **kwargs)
                if first_only:
                    break
        return

    def plot_cell_centers(self, s=10, marker="*", linewidth=0, color="green", **kwargs):
        """Plot cell centers in scatter."""
        if self._cells is None:
            print("No cells to plot")
            return

        self.plot_scatters(self._cells, s=s, marker=marker, linewidth=linewidth, color=color, **kwargs)
        return

    def plot_scatters(self, coords, **kwargs):
        """Plot general scatters, such as cell center or gene transcripts."""
        cell_center_coords = self._transform.micron_to_pixel_transform(coords)
        xmin, ymin = self._offset
        x = cell_center_coords[:, 0] - xmin
        y = cell_center_coords[:, 1] - ymin
        self.ax.scatter(x, y, **kwargs)
        return
