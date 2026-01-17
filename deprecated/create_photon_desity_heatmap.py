# load each block into the interactive python shell after running simulation with "python -i simulation.py"
# this will create reflection density (x-z plane) or scatter density (x-y plane)

# ========== reflection density

is_reflection = simulation.photon_np_array["reflection"] == True
positions_filtered = simulation.photon_np_array["position"][is_reflection]
tree = KDTree(positions_filtered)
x_min, x_max = 0, 150
z_min, z_max = -75, 75
grid_res = 100

x = np.linspace(x_min, x_max, grid_res)
z = np.linspace(z_min, z_max, grid_res)
xx, zz = np.meshgrid(x, z)
yy = np.full_like(xx, simulation.seafloor_y)

sample_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
dists, _ = tree.query(sample_points, k=1)
heatmap = dists.reshape(grid_res, grid_res)

plt.imshow(heatmap, origin='lower', extent=(x_min, x_max, z_min, z_max), cmap='hot')
plt.colorbar(label='Distance to Nearest Photon')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Reflection Density Map (X-Z plane)')
plt.show()

# ========== scatter density

is_scatter = simulation.photon_np_array["reflection"] == False
positions_filtered_2 = simulation.photon_np_array["position"][is_scatter]
tree_scatter = KDTree(positions_filtered_2)
x_min_2, x_max_2 = 0, 150
y_min_2, y_max_2 = simulation.seafloor_y, simulation.water_surface_y
grid_res_2 = 100

x_2 = np.linspace(x_min_2, x_max_2, grid_res_2)
y_2 = np.linspace(y_min_2, y_max_2, grid_res_2)
xx_2, yy_2 = np.meshgrid(x_2, y_2)
zz_2 = np.full_like(xx_2, 0)

sample_points_2 = np.stack([xx_2, yy_2, zz_2], axis=-1).reshape(-1, 3)
dists_2, _ = tree_scatter.query(sample_points_2, k=1)
heatmap_2 = dists_2.reshape(grid_res_2, grid_res_2)

plt.imshow(heatmap, origin='lower', extent=(x_min_2, x_max_2, y_min_2, y_max_2), cmap='hot')
plt.colorbar(label='Distance to Nearest Photon')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Density Map (X-Y plane)')
plt.show()
