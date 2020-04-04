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
        self.difference_courante: float = np.inf
        self.h: float = kwargs.get("h", 0.1)
        self.erreur: float = kwargs.get("erreur", calcul_erreur(self.h))
        self.temps_de_calcul: float = 0.00
        self.nom = kwargs.get("nom", "carte_de_chaleur")

        self.differences: list = list()
        self.temps: list = list()

        self.kernel = lambda r: (1 / 4) * np.array(
             [[0, 1-(1 / max(2*r, 1)), 0],
              [1, 0, 1],
              [0, (1+1 / max(2*r, 1)), 0]]
        )
        self.kernels = np.array([
            self.kernel(i) for i in range(self.grille.shape[0])
        ])

        self.appliquer_frontieres()

    def appliquer_frontieres(self):
        self.grille = np.where(self.frontieres.mask, self.frontieres, self.grille)

    @jit
    def calcul_sur_grille(self):
        prochaine_grille = np.copy(self.grille)

        for i in range(self.grille[1:-1, :].shape[0]):
            j = i + 1
            prochaine_grille[1:-1, 1:-1][i, :] = signal.convolve2d(self.grille[j - 1:j + 2, :], self.kernels[j],
                                                                   mode="valid")
        self.grille = prochaine_grille

    def faire_iteration(self):
        temps_depart = time.time()
        self.iteration += 1
        self.grille_precedente = np.copy(self.grille)
        self.calcul_sur_grille()
        self.appliquer_frontieres()

        self.calcul_erreur()
        self.verification_terminal()

        self.temps_de_calcul += time.time() - temps_depart
        self.temps.append(self.temps_de_calcul)
        return self.grille, self.iteration

    @jit
    def calcul_erreur(self):
        erreur = np.sum((self.grille[1:-1, 1:-1] - self.grille_precedente[1:-1, 1:-1] )**2)
        self.difference_courante = np.sqrt(erreur)
        self.differences.append(self.difference_courante)

    def verification_terminal(self):
        self.terminal = self.difference_courante <= self.erreur

    def __call__(self, nombre_iterations: int = None, **kwargs):
        temp_depart = time.time()
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

        temps_de_calcul = time.time() - temp_depart
        if kwargs.get("verbose", False):
            print(f"Temps de calcul: {temps_de_calcul} s")

        return self.grille, self.iteration, temps_de_calcul

    def afficher_etat(self):
        print(f"--- Grille {self.nom} --- \n "
              f"Méthode: {self.methode} \n"
              f"itr: {self.iteration} -> diff: {self.difference_courante:.5e}, Terminal: {self.terminal} \n"
              f"Temps de calcul total: {self.temps_de_calcul:.3f} s \n"
              f"--- {'-'*(7+len(self.nom))} --- \n")

    def afficher_carte_de_chaleur(self):
        plt.clf()
        sns.set()
        ax = sns.heatmap(self.grille)
        plt.xlabel("z [cm/h]")
        plt.ylabel("r [cm/h]")
        plt.title(f"{self.nom} - {self.iteration} itérations")
        plt.savefig(f"Figures/{self.nom.replace(' ', '_')}-{self.iteration}itr.png", dpi=300)
        plt.show()

    def afficher_differences_en_fct_iteration(self):
        plt.clf()
        plt.plot(range(1, self.iteration+1), self.differences, lw=2)
        plt.xlabel("Itération [-]")
        plt.ylabel("Différence [-]")
        plt.title(f"Différence en fonction du nombre d'itération pour {self.methode}")
        plt.savefig(f"Figures/errFctItr-{self.methode.replace(' ', '_')}-{self.iteration}itr.png", dpi=300)
        plt.show()

    def afficher_temps_en_fct_iteration(self):
        plt.clf()
        plt.plot(range(1, self.iteration+1), self.temps, lw=2)
        plt.xlabel("Itération [-]")
        plt.ylabel("Temps de calcul [-]")
        plt.title(f"Temps de calcul en fonction du nombre d'itération pour {self.methode}")
        plt.savefig(f"Figures/tempsFctItr-{self.methode.replace(' ', '_')}-{self.iteration}itr.png", dpi=300)
        plt.show()


class RelaxationGaussSeidel(Relaxation):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(RelaxationGaussSeidel, self).__init__(grille, frontieres, **kwargs)
        self.methode = "Relaxation Gauss-Seidel"

    @jit
    def calcul_sur_grille(self):
        for i in range(self.grille[1:-1, :].shape[0]):
            if i % 2 == 0:
                continue
            j = i + 1
            self.grille[1:-1, 1:-1][i, :] = signal.convolve2d(self.grille[j - 1:j + 2, :], self.kernels[j],
                                                              mode="valid")

        for i in range(self.grille[1:-1, :].shape[0]):
            if i % 2 != 0:
                continue
            j = i + 1
            self.grille[1:-1, 1:-1][i, :] = signal.convolve2d(self.grille[j - 1:j + 2, :], self.kernels[j],
                                                              mode="valid")


class SurRelaxation(Relaxation):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(SurRelaxation, self).__init__(grille, frontieres, **kwargs)
        self.methode = "Sur relaxation"
        self.w: float = kwargs.get("w", 0.00)

        self.kernel = lambda r: ((1 + self.w) / 4) * np.array(
            [[0, 1 - (1 / max(2 * r, 1)), 0],
             [1, 0, 1],
             [0, (1 + 1 / max(2 * r, 1)), 0]]
        ) - self.w * np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]
        )
        self.kernels = np.array([
            self.kernel(i) for i in range(self.grille.shape[0])
        ])

    @jit
    def calcul_sur_grille(self):
        prochaine_grille = np.copy(self.grille)

        for i in range(self.grille[1:-1, :].shape[0]):
            j = i + 1
            prochaine_grille[1:-1, 1:-1][i, :] = signal.convolve2d(prochaine_grille[j - 1:j + 2, :], self.kernels[j],
                                                                   mode="valid")

        self.grille = prochaine_grille

    def afficher_etat(self):
        print(f"--- Grille {self.nom} --- \n "
              f"Méthode: {self.methode} avec w = {self.w} \n"
              f"itr: {self.iteration} -> diff: {self.difference_courante:.5e}, Terminal: {self.terminal} \n"
              f"Temps de calcul total: {self.temps_de_calcul:.3f} s \n"
              f"--- {'-'*(7+len(self.nom))} --- \n")


class SurRelaxationGaussSeidel(SurRelaxation, RelaxationGaussSeidel):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(SurRelaxationGaussSeidel, self).__init__(grille, frontieres, **kwargs)
        self.methode = "Sur relaxation + Gauss-Seidel"

    @jit
    def calcul_sur_grille(self):
        RelaxationGaussSeidel.calcul_sur_grille(self)

    def afficher_etat(self):
        SurRelaxation.afficher_etat(self)
