import numpy as np


class Cylindre:
    def __init__(self, rayon: float, hauteur: float, **kwargs):
        self.rayon = rayon
        self.hauteur = hauteur
        self.hdemi = kwargs.get("h", 1.0) / 2
        self.dimensions = (int(rayon / self.hdemi), int(hauteur / self.hdemi))

    @property
    def grille(self):
        return np.zeros(self.dimensions)
