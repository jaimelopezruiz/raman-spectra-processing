import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image

# === INPUT: stress values from your 3x5 matrix (in MPa) ===
stress_values = np.array([
    [1007.488, 1326.032,    7.408,  -11.112, 1377.888],
    [ 674.128,  663.016,    0.000,  663.016,  663.016],
    [ 655.608,  655.608,  374.104,     np.nan,    np.nan]
])

# === IMAGE: microscope image path ===
image_path = "Irr Region of Interest Marked.png"  # Replace with path to your microscope image

# === SPATIAL CONFIGURATION ===
# Real spacing between points (based on 5 µm scale bar)
x_spacing = 3  # μm between columns
y_spacing = 3  # μm between rows

# Grid positions
nrows, ncols = stress_values.shape
x = np.arange(ncols) * x_spacing
y = np.arange(nrows) * y_spacing
X, Y = np.meshgrid(x, y)

# Interpolation setup
points = np.column_stack((X.flatten(), Y.flatten()))
values = stress_values.flatten()

# Remove NaNs
mask = ~np.isnan(values)
points = points[mask]
values = values[mask]

# Fine interpolation grid
xi = np.linspace(x.min(), x.max(), 300)
yi = np.linspace(y.min(), y.max(), 300)
xi, yi = np.meshgrid(xi, yi)
zi = griddata(points, values, (xi, yi), method='cubic')

# === PLOT ===
fig, ax = plt.subplots(figsize=(8, 6))

# Load microscope image
img = Image.open(image_path)
img_extent = [
    x.min() - x_spacing / 2, x.max() + x_spacing / 2,
    y.max() + y_spacing / 2, y.min() - y_spacing / 2  # Y reversed to match top-left origin
]

# Show image
ax.imshow(img, extent=img_extent, aspect='auto')

# Contourf overlay
cmap = 'jet'
stress_map = ax.contourf(xi, yi, zi, levels=100, cmap=cmap, alpha=0.65)

# Contour lines
contours = ax.contour(xi, yi, zi, levels=np.arange(-2000, 2000, 250),
                      colors='k', linewidths=0.5)

# Scatter original points
ax.scatter(X, Y, c='black', s=15, zorder=10)

# Formatting
cbar = fig.colorbar(stress_map, ax=ax, label='Stress (MPa)')
ax.set_title('Stress Map from Raman (MPa)')
ax.set_xlabel('X (μm)')
ax.set_ylabel('Y (μm)')
ax.set_aspect('equal')
ax.set_xlim(img_extent[0], img_extent[1])
ax.set_ylim(img_extent[2], img_extent[3])

plt.tight_layout()
plt.show()
