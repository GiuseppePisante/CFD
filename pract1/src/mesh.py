import numpy as np

import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, x_start, x_end, y_start, y_end, x_points, y_points):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_points = x_points
        self.y_points = y_points
        self.x, self.y = None, None

    def mesh_generator(self):
        self.x = np.linspace(self.x_start, self.x_end, self.x_points)
        self.y = np.linspace(self.y_start, self.y_end, self.y_points)
        self.x, self.y = np.meshgrid(self.x, self.y)

    def show(self):
        if self.x is None or self.y is None:
            raise ValueError("Mesh not generated. Call mesh_generator() first.")
        plt.figure()
        plt.plot(self.x, self.y, marker='.', color='k', linestyle='none')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Mesh')
        plt.show()