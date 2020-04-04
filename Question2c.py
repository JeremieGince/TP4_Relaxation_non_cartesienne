import numpy as np
from Cylindre import Cylindre
from Relaxation import RelaxationGaussSeidel, SurRelaxation, SurRelaxationGaussSeidel


if __name__ == '__main__':
    h = 0.1
    cylindre_1 = Cylindre(rayon=10, hauteur=30, h=h)

    mask_150V = np.zeros(cylindre_1.grille.shape)
    mask_150V[int(1 / cylindre_1.h):int(1 / cylindre_1.h) + 1, :] = True

    mask_0V = np.zeros(cylindre_1.grille.shape)
    mask_0V[-1:, :] = True
    mask_0V[:1, :] = True
    mask_0V[:int(1 / cylindre_1.h), :1] = True
    mask_0V[int(1 / cylindre_1.h) + 1:, :1] = True
    mask_0V[:int(1 / cylindre_1.h), -1:] = True
    mask_0V[int(1 / cylindre_1.h) + 1:, -1:] = True


    frontieres = np.ma.array(
        np.where(mask_150V, 150, cylindre_1.grille) + np.where(mask_0V, 0, cylindre_1.grille),
        mask=(mask_150V+mask_0V)
    )
    # Question c i
    relax_gauss = RelaxationGaussSeidel(grille=cylindre_1.grille, frontieres=frontieres, h=h,
                                        nom="Carte de chaleur problème 2ci")
    relax_gauss(30_000, verbose=True, affichage=False)
    relax_gauss.afficher_carte_de_chaleur()

    # Question c ii
    sur_relax = SurRelaxation(grille=cylindre_1.grille, frontieres=frontieres, h=h,
                              nom="Carte de chaleur problème 2cii", w=0.0005)  # w=0.0005
    sur_relax(30_000, verbose=True, affichage=False)
    sur_relax.afficher_carte_de_chaleur()

    # Question c iii
    sur_relax_gauss = SurRelaxationGaussSeidel(grille=cylindre_1.grille, frontieres=frontieres, h_par_indice=0.5, h=h,
                                               nom="Carte de chaleur problème 2ciii", w=0.1)
    sur_relax_gauss(30_000, verbose=True, affichage=False)
    sur_relax_gauss.afficher_carte_de_chaleur()

    relax_gauss.afficher_etat()
    sur_relax.afficher_etat()
    sur_relax_gauss.afficher_etat()

