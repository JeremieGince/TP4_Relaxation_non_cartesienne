import numpy as np
from Cylindre import Cylindre
from Relaxation import RelaxationGaussSeidel, SurRelaxation


if __name__ == '__main__':
    h = 0.1
    cylindre_1 = Cylindre(rayon=10, hauteur=30, h=h)

    mask_150V = np.zeros(cylindre_1.grille.shape)
    mask_150V[int(1 / cylindre_1.h) - 1:int(1 / cylindre_1.h) + 1, :] = True

    mask_0V = np.zeros(cylindre_1.grille.shape)
    mask_0V[-2, :] = True

    frontieres = np.ma.array(
        np.where(mask_150V, 150, 0 * cylindre_1.grille) + np.where(mask_0V, 0, 0 * cylindre_1.grille),
        mask=(mask_150V + mask_0V)
    )
    """
    # Question c i

    relax_gauss = RelaxationGaussSeidel(grille=cylindre_1.grille, frontieres=frontieres, h=h, erreur=-np.inf,
                                        nom="Carte de chaleur problème 2ci")
    relax_gauss(12_029, verbose=False, affichage=False)
    relax_gauss.afficher_etat()
    relax_gauss.afficher_carte_de_chaleur()
    """
    # Question c ii
    sur_relax = SurRelaxation(grille=cylindre_1.grille, frontieres=frontieres, h_par_indice=0.5, erreur=0.1,
                              nom="Carte de chaleur problème 2cii", w=1.0009)
    sur_relax(12_030, verbose=True, affichage=False)
    sur_relax.afficher_carte_de_chaleur()

