import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from Relaxation import Relaxation


class RelaxationGaussSeidel(Relaxation):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(RelaxationGaussSeidel, self).__init__(grille, frontieres, **kwargs)
        self.w: float = kwargs.get("w", 0.0)

    def faire_iteration(self):
        self.iteration += 1
        self.grille_precedente = np.copy(self.grille)
        prochaine_grille = np.copy(self.grille)
        r = np.indices(self.grille.shape)[0][2:-2, 2:-2]
        prochaine_grille[2:-2, 2:-2] = (self.grille[:-4, 2:-2] + self.grille[4:, 2:-2]
                                        + self.grille[2:-2, :-4] + self.grille[2:-2, 4:]) / 4 \
                                       + (self.grille[3:-1, 2:-2] - self.grille[1:-3, 2:-2]) / (r * 4)

        self.grille = prochaine_grille
        self.appliquer_frontieres()

        self.calcul_erreur()
        self.verification_terminal()

        return self.grille, self.iteration