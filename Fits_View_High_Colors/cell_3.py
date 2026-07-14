# --- Cell 3: build the custom colormap ---
colors = ["#00FFFF", "#8F00FF", "#FFFFFF"]   # cyan → high ultraviolet → white
my_cmap = LinearSegmentedColormap.from_list("cyan_uv_white", colors, N=256)
