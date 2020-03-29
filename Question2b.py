import numpy as np
from Cylindre import Cylindre
from Relaxation import Relaxation
import seaborn as sns
import matplotlib.pyplot as plt


def solution_analytique(grille_dimensions: (int, int), h: float):

    v = lambda r, z: 150*(np.log(r) - np.log(10))/(np.log(1) - np.log(10))
    grille = []
    for i in range(grille_dimensions[0]):
        ligne = []
        for j in range(grille_dimensions[1]):
            vi = v(i*h, j*h)
            ligne.append(vi if vi != np.inf else 150)
        grille.append(ligne)

    grille = np.array(grille)
    print(grille)

    sns.set()
    ax = sns.heatmap(grille)
    plt.xlabel("z [2cm/h]")
    plt.ylabel("r [2cm/h]")
    plt.title("Solution analytique")
    plt.savefig(f"Figures/heatmap_2b_analytique.png", dpi=300)
    # plt.show()


if __name__ == '__main__':
    h = 0.1
    cylindre_1 = Cylindre(rayon=10, hauteur=30, h=h)

    print(cylindre_1.grille.shape)

    mask_150V = np.zeros(cylindre_1.grille.shape)
    mask_150V[int(1/cylindre_1.h)-1:int(1/cylindre_1.h)+1, :] = True

    mask_0V = np.zeros(cylindre_1.grille.shape)
    mask_0V[-2, :] = True

    print(mask_150V+mask_0V)

    frontieres = np.ma.array(
        np.where(mask_150V, 150, 0*cylindre_1.grille) + np.where(mask_0V, 0, 0*cylindre_1.grille),
        mask=(mask_150V+mask_0V)
    )

    print(frontieres.data, frontieres.mask, sep="\n")

    solution_analytique(cylindre_1.dimensions, cylindre_1.h)

    relax = Relaxation(grille=cylindre_1.grille, frontieres=frontieres, h_par_indice=0.5, erreur=0.005,
                       nom="Carte de chaleur problème 2b")
    relax(100_000, verbose=True, affichage=False)
    relax.afficher_carte_de_chaleur()

