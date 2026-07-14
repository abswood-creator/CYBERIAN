# --- Cell 4: scale + plot ---
vmin, vmax = ZScaleInterval().get_limits(data)

plt.figure(figsize=(8, 8))
plt.imshow(data, cmap=my_cmap, vmin=vmin, vmax=vmax, origin='lower')
plt.colorbar(label='Counts')
plt.title('j8q901081_drz')
plt.axis('off')
plt.tight_layout()
plt.show()
