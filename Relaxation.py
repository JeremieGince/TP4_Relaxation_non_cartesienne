import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy import signal


class Relaxation:
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        self.iteration: int = 0
        self.grille: np.ndarray = grille
        self.grille_precedente = None
        self.frontieres: np.ma.MaskedArray = frontieres
        self.terminal: bool = False
        self.difference_courante: float = np.inf
        self.erreur: float = kwargs.get("erreur", 1e1)
        self.h_par_indice = kwargs.get("h_par_indice", 1)
        self.nom = kwargs.get("nom", "carte_de_chaleur")
        self.mask_erreur = None

        self.appliquer_frontieres()

    def appliquer_frontieres(self):
        self.grille = np.where(self.frontieres.mask, self.frontieres, self.grille)

    def faire_iteration(self):
        self.iteration += 1
        self.grille_precedente = np.copy(self.grille)

        prochaine_grille = np.copy(self.grille)
        r = np.indices(self.grille.shape)[0][2:-2, 2:-2]
        prochaine_grille[2:-2, 2:-2] = (self.grille[:-4, 2:-2] + self.grille[4:, 2:-2]
                                        + self.grille[2:-2, :-4] + self.grille[2:-2, 4:])/4 \
                                        + (self.grille[3:-1, 2:-2] - self.grille[1:-3, 2:-2])/(r*2)

        self.grille = prochaine_grille
        self.appliquer_frontieres()

        self.calcul_erreur()
        self.verification_terminal()
        return self.grille, self.iteration

    def calcul_erreur(self):
        if self.iteration == 1:
            mask_shape = self.grille[2:-2, 2:-2].shape
            mask = np.ones(mask_shape)
            for i in range(0,mask_shape[0]):
                if (i % 2) != 0:
                    mask[i, :] = np.zeros((mask_shape[1]))
            for j in range(0,mask_shape[1]):
                if (j % 2) != 0:
                    mask[:, j] = np.zeros((mask_shape[0]))
            self.mask_erreur = mask

        grille_erreur = self.grille.copy()
        grille_erreur_precedente = self.grille_precedente.copy()
        grille_erreur[2:-2, 2:-2] = (grille_erreur[2:-2, 2:-2] + grille_erreur[1:-3, 2:-2]
                                        + grille_erreur[2:-2, 1:-3])/4
        grille_erreur_precedente[2:-2, 2:-2] = (grille_erreur_precedente[2:-2, 2:-2] + grille_erreur_precedente[1:-3, 2:-2]
                                        + grille_erreur_precedente[2:-2, 1:-3])/4
        erreur = np.ma.sum((np.ma.masked_where(self.mask_erreur, grille_erreur[2:-2, 2:-2]) - np.ma.masked_where(self.mask_erreur, grille_erreur_precedente[2:-2, 2:-2]))**2)
        self.difference_courante = np.sqrt(erreur)
        # print(self.difference_courante)

    def verification_terminal(self):
        self.terminal = self.difference_courante <= self.erreur

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
              f"itr: {self.iteration} -> diff: {self.difference_courante}, Terminal: {self.terminal} \n"
              f"--- {'-'*(7+len(self.nom))} --- \n")

    def afficher_carte_de_chaleur(self):
        sns.set()
        ax = sns.heatmap(self.grille)
        plt.xlabel("z [2cm/hdemi]")
        plt.ylabel("r [2cm/hdemi]")
        plt.title(f"{self.nom} - {self.iteration} itérations")
        plt.savefig(f"Figures/{self.nom}-{self.iteration}itr.png", dpi=300)
        plt.show()


class RelaxationGaussSeidel(Relaxation):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(RelaxationGaussSeidel, self).__init__(grille, frontieres, **kwargs)
        self.kernel_h = (1/4)*np.array(
            [[0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0]]
        )
        self.kernel_hdemi = (1/4)*np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, -1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]]
        )

    def faire_iteration(self):
        self.iteration += 1
        self.grille_precedente = np.copy(self.grille)

        # self.grille[2:-2, 2:-2] = (self.grille[:-4, 2:-2] + self.grille[4:, 2:-2]
        #                           + self.grille[2:-2, :-4] + self.grille[2:-2, 4:]) / 4 \
        #                           + (self.grille[3:-1, 2:-2] - self.grille[1:-3, 2:-2]) / (r * 4)

        # self.grille[2:-2, 2:-2] = self.grille[:-4, 2:-2] / 4
        # self.grille[2:-2, 2:-2] += self.grille[4:, 2:-2] / 4
        # self.grille[2:-2, 2:-2] += self.grille[2:-2, :-4] / 4
        # self.grille[2:-2, 2:-2] += self.grille[2:-2, 4:] / 4
        # self.grille[2:-2, 2:-2] += self.grille[3:-1, 2:-2] / (r*4)
        # self.grille[2:-2, 2:-2] += -self.grille[1:-3, 2:-2] / (r*4)

        prochaine_grille = np.copy(self.grille)
        r = np.indices(self.grille.shape)[0][2:-2, 2:-2]
        prochaine_grille[2:-2, 2:-2] = signal.convolve2d(self.grille, self.kernel_h, mode="same")[2:-2, 2:-2]
        prochaine_grille[2:-2, 2:-2] += signal.convolve2d(self.grille, self.kernel_hdemi, mode="same")[2:-2, 2:-2]/r

        self.grille = prochaine_grille

        self.appliquer_frontieres()

        self.calcul_erreur()
        self.verification_terminal()

        return self.grille, self.iteration


class SurRelaxation(Relaxation):
    def __init__(self, grille: np.ndarray, frontieres: np.ma.MaskedArray, **kwargs):
        super(SurRelaxation, self).__init__(grille, frontieres, **kwargs)
        self.w: float = kwargs.get("w", 0.0000000000000000000000000000000000000)

    def faire_iteration(self):
        self.iteration += 1
        self.grille_precedente = np.copy(self.grille)
        prochaine_grille = np.copy(self.grille)

        r = np.indices(self.grille.shape)[0][2:-2, 2:-2]
        prochaine_grille[2:-2, 2:-2] = (1-self.w)*self.grille[2:-2, 2:-2] + \
                                  self.w*(self.grille[:-4, 2:-2] + self.grille[4:, 2:-2]
                                  + self.grille[2:-2, :-4] + self.grille[2:-2, 4:]) / 4 \
                                  + self.w*(self.grille[3:-1, 2:-2] - self.grille[1:-3, 2:-2]) / (r * 2)

        self.grille = prochaine_grille
        self.appliquer_frontieres()
        self.calcul_erreur()
        self.verification_terminal()

        return self.grille, self.iteration


class SurRelaxationGaussSeidel(SurRelaxation):
    def faire_iteration(self):
        self.iteration += 1
        self.grille_precedente = np.copy(self.grille)

        r = np.indices(self.grille.shape)[0][2:-2, 2:-2]

        self.grille[2:-2, 2:-2] = -self.w*self.grille[2:-2, 2:-2] \
                                  + (1+self.w)*(self.grille[:-4, 2:-2] + self.grille[4:, 2:-2]
                                  + self.grille[2:-2, :-4] + self.grille[2:-2, 4:]) / 4 \
                                  + (1+self.w)*(self.grille[3:-1, 2:-2] - self.grille[1:-3, 2:-2]) / (r * 2)

        self.appliquer_frontieres()

        self.calcul_erreur()
        self.verification_terminal()

        return self.grille, self.iteration
