import numpy as np
import matplotlib.pyplot as plt
from Cylindre import Cylindre
from Relaxation import Relaxation, RelaxationGaussSeidel, SurRelaxation, SurRelaxationGaussSeidel


if __name__ == '__main__':
    h = 0.1

    # Question 3 b
    cylindre_2 = Cylindre(rayon=10, hauteur=36, h=h)

    mask_150V_c2 = np.zeros(cylindre_2.grille.shape)
    mask_150V_c2[int(1 / cylindre_2.h):int(1 / cylindre_2.h) + 1, int(3 / cylindre_2.h):int(33 / cylindre_2.h)] = True

    mask_0V_c2 = np.zeros(cylindre_2.grille.shape)
    mask_0V_c2[-1, :int(9 / cylindre_2.h)] = True
    mask_0V_c2[-1, int(30 / cylindre_2.h):] = True
    mask_0V_c2[int(5 / cylindre_2.h):int(5 / cylindre_2.h) + 1, int(9 / cylindre_2.h):int(-9 / cylindre_2.h)] = True
    mask_0V_c2[:, [0, -1]] = True

    frontieres_c2 = np.ma.array(
        np.where(mask_150V_c2, 150, cylindre_2.grille) + np.where(mask_0V_c2, 0, cylindre_2.grille),
        mask=(mask_150V_c2 + mask_0V_c2)
    )

    # print(mask_150V_c2)
    # print()
    # print(mask_0V_c2)
    # print()
    # print(frontieres_c2.mask)

    # Question 2 b
    relax_2 = Relaxation(grille=cylindre_2.grille, frontieres=frontieres_c2, h_par_indice=1, h=h,
                         nom="Carte de chaleur problème 3b")
    relax_2(-1, verbose=False, affichage=False)
    relax_2.afficher_carte_de_chaleur()
    relax_2.afficher_etat()

    # Question 3 c
    cylindre_3 = Cylindre(rayon=10, hauteur=33, h=h)

    mask_150V_c3 = np.zeros(cylindre_3.grille.shape)
    mask_150V_c3[int(1 / cylindre_3.h):int(1 / cylindre_3.h) + 1, int(2 / cylindre_3.h):int(32 / cylindre_3.h)] = True

    mask_0V_c3 = np.zeros(cylindre_3.grille.shape)
    mask_0V_c3[-1, :int(10 / cylindre_3.h)] = True
    mask_0V_c3[int(8 / cylindre_3.h), int(10 / cylindre_3.h):int(22 / cylindre_3.h)] = True
    mask_0V_c3[int(4 / cylindre_3.h), int(22 / cylindre_3.h):] = True
    mask_0V_c3[:, [0, -1]] = True

    frontieres_c3 = np.ma.array(
        np.where(mask_150V_c3, 150, cylindre_3.grille) + np.where(mask_0V_c3, 0, cylindre_3.grille),
        mask=(mask_150V_c3 + mask_0V_c3)
    )

    # print(mask_150V_c3)
    # print()
    # print(mask_0V_c3)
    # print()
    # print(frontieres_c3.mask)

    relax_3 = Relaxation(grille=cylindre_3.grille, frontieres=frontieres_c3, h_par_indice=1, h=h,
                         nom="Carte de chaleur problème 3c")
    relax_3(-1, verbose=False, affichage=False)
    relax_3.afficher_carte_de_chaleur()
    relax_3.afficher_etat()
