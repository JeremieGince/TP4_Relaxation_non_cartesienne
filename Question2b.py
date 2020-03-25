import numpy as np
from Cylindre import Cylindre
from Relaxation import Relaxation


if __name__ == '__main__':
    h = 1
    cylindre_1 = Cylindre(rayon=10, hauteur=30, h=h)

    print(cylindre_1.grille.shape)

    mask_150V = np.zeros(cylindre_1.grille.shape)
    mask_150V[0:2//h, :] = True

    mask_0V = np.zeros(cylindre_1.grille.shape)
    mask_0V[-1, :] = True

    print(mask_150V+mask_0V)

    frontieres = np.ma.array(
        np.where(mask_150V, 150, 0*cylindre_1.grille) + np.where(mask_0V, 0, 0*cylindre_1.grille),
        mask=(mask_150V+mask_0V)
    )

    print(frontieres.data, frontieres.mask, sep="\n")

    relax = Relaxation(grille=cylindre_1.grille, frontieres=frontieres, h=h)
    print(relax(1))
