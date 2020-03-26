import numpy as np


class Cylindre:
    def __init__(self, rayon: float, hauteur: float, **kwargs):
        self.rayon = rayon
        self.hauteur = hauteur
        self.h = kwargs.get("h", 1.0)
        self.dimensions = (int(rayon/self.h), int(hauteur/self.h))

    @property
    def grille(self):
        return np.ones(self.dimensions)
