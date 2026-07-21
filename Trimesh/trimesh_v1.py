import trimesh
import numpy as np

vertices = np.array([
    [0, 0, 0],
    [0, 10, 0],
    [5, 15, 0],
    [5, 5, 0]
])

faces = np.array([
    [0, 1, 2],
    [0, 2, 3]
])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# mesh.show()
