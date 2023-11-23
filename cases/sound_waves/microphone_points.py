import numpy as np


class Microphone:
    """Slices microphone points where makes a sound measurements."""
    def __init__(self, matrix: np.ndarray = None):
        self.matrix = np.random.rand(120, 120) if matrix is None else matrix

    def array(self):
        """Generates np.array of sound pressure."""
        arrs = [
            [
                [
                    self.matrix[0, 120 // 2],
                    self.matrix[0, 3 * 120 // 4],
                    self.matrix[0, -1],
                    self.matrix[3 * 120 // 4, -1],
                    self.matrix[120 // 2, -1],
                    self.matrix[120 // 4, -1],
                    self.matrix[-1, -1],
                    self.matrix[-1, 3 * 120 // 4],
                    self.matrix[-1, 120 // 2],
                ]
            ],
            [self.matrix[0, 64:-1][::4], self.matrix[7:-7, -1][::3], self.matrix[-1, 64:-1][::4]],
            [self.matrix[0, 59:-1], self.matrix[:, -1], self.matrix[-1, 59:-1]],
            self.matrix,
        ]
        return arrs
