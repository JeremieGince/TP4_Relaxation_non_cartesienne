import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Relaxation:
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        self.iteration: int = 0
        self.grille: np.ndarray = grille
        self.grille_precedente = None
        self.frontieres: np.ma.MaskedArray = frontieres
        self.terminal: bool = False
        self.difference_courante: float = np.inf
        self.erreur: float = kwargs.get("erreur", 1e1)
        self.h = kwargs.get("h", 1)

        self.appliquer_frontieres()

    def appliquer_frontieres(self):
        self.grille = np.where(self.frontieres.mask, self.frontieres, self.grille)

    def faire_iteration(self):
        self.iteration += 1
        prochaine_grille = np.copy(self.grille)
        r = np.indices(self.grille.shape)[0][2:-2, 2:-2]
        prochaine_grille[2:-2, 2:-2] = (self.grille[:-4, 2:-2] + self.grille[4:, 2:-2]
                                        + self.grille[2:-2, :-4] + self.grille[2:-2, 4:])/4 \
                                        + (self.grille[3:-1, 2:-2] - self.grille[1:-3, 2:-2])/(r*4)

        self.grille_precedente = np.copy(self.grille)
        self.grille = prochaine_grille
        self.appliquer_frontieres()

        self.calcul_erreur()
        self.verification_terminal()
        return self.grille, self.iteration

    def calcul_erreur(self):
        if self.grille_precedente is not None:
            gv = self.grille[np.invert(self.frontieres.mask)].flatten()
            gnv = self.grille_precedente[np.invert(self.frontieres.mask)].flatten()
            chng = np.sqrt(np.sum((gv - gnv) ** 2))
            self.difference_courante = chng

    def verification_terminal(self):
        self.terminal = self.difference_courante <= self.erreur

    def __call__(self, nombre_iterations: int = None, **kwargs):
        if nombre_iterations is None:
            nombre_iterations = np.int(np.inf)

        for i in range(nombre_iterations):
            self.faire_iteration()

            if kwargs.get("verbose", False) and 100*((i+1)/nombre_iterations) % 10 == 0:
                print(f"itr: {i+1}/{nombre_iterations} -> diff: {self.difference_courante}, Terminal: {self.terminal}")
            if kwargs.get("affichage", False) and 100*((i+1)/nombre_iterations) % 10 == 0:
                self.afficher_carte_de_chaleur()

            if self.terminal:
                if kwargs.get("verbose", False) and 100 * ((i + 1) / nombre_iterations) % 10 != 0:
                    print(f"itr: {i + 1}/{nombre_iterations} -> diff: {self.difference_courante},"
                          f" Terminal: {self.terminal}")
                if kwargs.get("affichage", False) and 100 * ((i + 1) / nombre_iterations) % 10 != 0:
                    self.afficher_carte_de_chaleur()
                break

        return self.grille, self.iteration

    def afficher_carte_de_chaleur(self):
        sns.set()
        ax = sns.heatmap(self.grille)
        plt.xlabel("z [2cm/h]")
        plt.ylabel("r [2cm/h]")
        plt.show()

