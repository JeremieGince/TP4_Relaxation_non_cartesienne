import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy import signal
from numba import jit


def calcul_erreur(h: float):
    C: float = 1e-16
    return (1/8)*h**2 + 6*C/h


class Relaxation:
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        self.methode = "Relaxation"
        self.iteration: int = 0
        self.grille: np.ndarray = grille
        self.grille_precedente = None
        self.frontieres: np.ma.MaskedArray = frontieres
        self.terminal: bool = False
        self.difference_precedente: float = None
        self.difference_courante: float = np.inf
        self.h: float = kwargs.get("h", 0.1)
        self.erreur: float = kwargs.get("erreur", calcul_erreur(self.h))
        self.nom = kwargs.get("nom", "carte_de_chaleur")

        # Pas modifier
        self.kernel = lambda r: (1 / 4) * np.array(
            [[0, 0, 1, 0, 0],
             [0, 0, -1 / max(r, 1), 0, 0],
             [1, 0, 0, 0, 1],
             [0, 0, 1 / max(r, 1), 0, 0],
             [0, 0, 1, 0, 0]]
        )
        self.kernels = np.array([
            self.kernel(i) for i in range(self.grille.shape[0])
        ])

        self.appliquer_frontieres()

    def appliquer_frontieres(self):
        self.grille = np.where(self.frontieres.mask, self.frontieres, self.grille)

    @jit
    def calcul_sur_grille(self):
        # prochaine_grille = np.copy(self.grille)
        # r = np.indices(self.grille.shape)[0][1:-1, 1:-1]
        # prochaine_grille[2:-2, 2:-2] = (self.grille[:-2, 1:-1] + self.grille[2:, 1:-1]
        #                                 + self.grille[1:-1, :-2] + self.grille[1:-1, 2:]) / 4 \
        #                                 + (self.grille[2:, 1:-1] - self.grille[0:-2, 1:-1]) / (r * 8)
        #
        # self.grille = prochaine_grille
        self.calcul_sur_grille_conv()

    @jit
    def calcul_sur_grille_conv(self):
        prochaine_grille = np.copy(self.grille)

        for i in range(self.grille[1:-1, :].shape[0]):
            j = i + 2
            # Pas modifier (signal....
            prochaine_grille[1:-1, 1:-1][i, :] = signal.convolve2d(self.grille[j - 2:j + 3, :], self.kernels[j],
                                                                   mode="valid")
        self.grille = prochaine_grille

    def faire_iteration(self):
        self.iteration += 1
        self.grille_precedente = np.copy(self.grille)
        self.calcul_sur_grille()
        self.appliquer_frontieres()

        self.calcul_erreur()
        self.verification_terminal()
        return self.grille, self.iteration

    @jit
    def calcul_erreur(self):
        erreur = np.sum((self.grille[1:-1, 1:-1] - self.grille_precedente[1:-1, 1:-1] )**2)
        self.difference_courante = np.sqrt(erreur)
        print(self.difference_courante)

    def verification_terminal(self):
        self.terminal = self.difference_courante <= self.erreur
        # if self.difference_precedente is not None and self.difference_courante > self.difference_precedente:
        #     self.terminal = True
        #     self.grille = self.grille_precedente

    def __call__(self, nombre_iterations: int = None, **kwargs):
        temps_de_depart = time.time()
        if nombre_iterations is None or nombre_iterations < 0:
            nombre_iterations = np.iinfo(np.int32).max

        for i in range(nombre_iterations):
            self.faire_iteration()

            if kwargs.get("verbose", False) and 100*((i+1)/nombre_iterations) % 10 == 0:
                print(f"itr: {i+1}/{nombre_iterations} -> diff: {self.difference_courante}, Terminal: {self.terminal}")
            if kwargs.get("affichage", False) and 100*((i+1)/nombre_iterations) % 10 == 0:
                self.afficher_carte_de_chaleur()

            if self.terminal:
                if kwargs.get("verbose", False) and 100*((i+1)/nombre_iterations) % 10 != 0:
                    self.afficher_etat()
                if kwargs.get("affichage", False) and 100*((i+1)/nombre_iterations) % 10 != 0:
                    self.afficher_carte_de_chaleur()
                break

        temps_de_calcul = time.time()-temps_de_depart
        if kwargs.get("verbose", False):
            print(f"Temps de calcul: {temps_de_calcul} s")

        return self.grille, self.iteration, temps_de_calcul

    def afficher_etat(self):
        print(f"--- Grille {self.nom} --- \n "
              f"Méthode: {self.methode} \n"
              f"itr: {self.iteration} -> diff: {self.difference_courante}, Terminal: {self.terminal} \n"
              f"--- {'-'*(7+len(self.nom))} --- \n")

    def afficher_carte_de_chaleur(self):
        plt.clf()
        sns.set()
        ax = sns.heatmap(self.grille)
        plt.xlabel("z [2cm/h]")
        plt.ylabel("r [2cm/h]")
        plt.title(f"{self.nom} - {self.iteration} itérations")
        plt.savefig(f"Figures/{self.nom}-{self.iteration}itr.png", dpi=300)
        plt.show()


class RelaxationGaussSeidel(Relaxation):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(RelaxationGaussSeidel, self).__init__(grille, frontieres, **kwargs)
        self.methode = "Relaxation Gauss-Seidel"
        # Pas modifier attention de ne pas oublier le facteur 2 sur le r (ce qui fait 4 fois 2 donc 8)
        self.kernel = lambda r: (1 / 4) * np.array(
                                    [[0, 0, 1, 0, 0],
                                     [0, 0, -1/max(r, 1), 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 1/max(r, 1), 0, 0],
                                     [0, 0, 1, 0, 0]]
                                )
        self.kernels = np.array([
            self.kernel(i) for i in range(self.grille.shape[0])
        ])

    @staticmethod
    @jit
    def calcul_sur_grille_jit(grille: np.ndarray, kernels: np.ndarray):
        for i in range(grille[1:-1, :].shape[0]):
            j = i + 2
            # Pas modifier (signal....
            grille[1:-1, 1:-1][i, :] = signal.convolve2d(grille[j-2:j+3, :], kernels[j], mode="valid")
        return grille

    def calcul_sur_grille_conv(self):
        for i in range(self.grille[1:-1, :].shape[0]):
            j = i + 2
            # Pas modifier (signal....
            self.grille[1:-1, 1:-1][i, :] = signal.convolve2d(self.grille[j-2:j+3, :], self.kernels[j], mode="valid")

    def calcul_sur_grille_sclicing(self):
        for i in range(self.grille[2:-2, :].shape[0]):
            if i % 2 == 0:
                continue
            r = np.indices(self.grille.shape)[0][1:-1, 1:-1]
            self.grille[1:-1, 1:-1][i, :] = ((self.grille[:-2, 1:-1] + self.grille[2:, 1:-1]
                                             + self.grille[1:-1, :-2] + self.grille[1:-1, 2:]) / 4
                                             + (self.grille[2:, 1:-1] - self.grille[:-2, 1:-1]) / (r * 8))[i, :]

        for i in range(self.grille[1:-1, :].shape[0]):
            # Pas modifier le if est tu vraiment utile maintenant que c h et non h/2
            if i % 2 != 0:
                continue
            r = np.indices(self.grille.shape)[0][1:-1, 1:-1]
            self.grille[2:-2, 2:-2][i, :] = ((self.grille[:-4, 2:-2] + self.grille[4:, 2:-2]
                                             + self.grille[2:-2, :-4] + self.grille[2:-2, 4:]) / 4 \
                                             + (self.grille[3:-1, 2:-2] - self.grille[1:-3, 2:-2]) / (r * 4))[i, :]

    def calcul_sur_grille(self):
        # self.calcul_sur_grille_sclicing()
        # self.grille = self.calcul_sur_grille_jit(self.grille, self.kernels)
        self.calcul_sur_grille_conv()


class SurRelaxation(Relaxation):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(SurRelaxation, self).__init__(grille, frontieres, **kwargs)
        self.methode = "Sur relaxation"
        self.w: float = kwargs.get("w", 0.00)
        # Pas modifier les kernels, whatch out pour le 4*2 qui mulripli le r
        self.kernel = lambda r: ((1 + self.w) / 4) * np.array(
            [[0, 0, 1, 0, 0],
             [0, 0, -1 / max(r, 1), 0, 0],
             [1, 0, 0, 0, 1],
             [0, 0, 1 / max(r, 1), 0, 0],
             [0, 0, 1, 0, 0]]
        ) - self.w * np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        )
        self.kernels = np.array([
            self.kernel(i) for i in range(self.grille.shape[0])
        ])

    @jit
    def calcul_sur_grille(self):
        prochaine_grille = np.copy(self.grille)

        # r = np.indices(self.grille.shape)[0][1:-1, 1:-1]
        # prochaine_grille[1:-1, 1:-1] = (1-self.w)*self.grille[1:-1, 1:-1] + \
        #                                 self.w*(self.grille[:-2, 1:-1] + self.grille[2:, 1:-1]
        #                                 + self.grille[1:-1, :-2] + self.grille[1:-1, 2:]) / 4 \
        #                                 + self.w*(self.grille[2:, 1:-1] - self.grille[:-2, 1:-1]) / (r * 8)

        for i in range(self.grille[1:-1, :].shape[0]):
            j = i + 2
            # Pas modifier (signal....
            prochaine_grille[1:-1, 1:-1][i, :] = signal.convolve2d(prochaine_grille[j-2:j+3, :], self.kernels[j], mode="valid")

        self.grille = prochaine_grille

    def afficher_etat(self):
        print(f"--- Grille {self.nom} --- \n "
              f"Méthode: {self.methode} avec w = {self.w} \n"
              f"itr: {self.iteration} -> diff: {self.difference_courante}, Terminal: {self.terminal} \n"
              f"--- {'-'*(7+len(self.nom))} --- \n")


class SurRelaxationGaussSeidel(SurRelaxation, RelaxationGaussSeidel):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(SurRelaxationGaussSeidel, self).__init__(grille, frontieres, **kwargs)
        self.methode = "Sur relaxation + Gauss-Seidel"
        # Pas modifier kernel, ici aussi whacth out pour le 4*2 du r
        self.kernel = lambda r: ((1+self.w) / 4) * np.array(
                                    [[0, 0, 1, 0, 0],
                                     [0, 0, -1/max(r, 1), 0, 0],
                                     [1, 0, 0, 0, 1],
                                     [0, 0, 1/max(r, 1), 0, 0],
                                     [0, 0, 1, 0, 0]]
                                ) - self.w * np.array(
                                                [[0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]
                                            )
        self.kernels = np.array([
            self.kernel(i) for i in range(self.grille.shape[0])
        ])

    @jit
    def calcul_sur_grille(self):
        RelaxationGaussSeidel.calcul_sur_grille(self)

    def afficher_etat(self):
        SurRelaxation.afficher_etat(self)
