import numpy as np
import matplotlib.pyplot as plt
from Cylindre import Cylindre
from Relaxation import Relaxation, RelaxationGaussSeidel, SurRelaxation, SurRelaxationGaussSeidel


def afficher_differences_en_fct_iteration(methodes):
    plt.clf()
    for methode in methodes:
        plt.plot(np.log(np.array(range(1, methode.iteration + 1))), np.array(methode.differences), lw=2, label=methode.methode)
    plt.xlabel("ln(Itération) [-]")
    plt.ylabel("Erreur [-]")
    plt.title(f"Erreur en fonction du nombre d'itérations")
    plt.grid()
    plt.legend()
    plt.savefig(f"Figures/errFctItr.png", dpi=300)
    plt.show()


def afficher_temps_en_fct_iteration(methodes):
    plt.clf()
    for methode in methodes:
        plt.plot(range(1, methode.iteration + 1), methode.temps, lw=2, label=methode.methode)
    plt.xlabel("Itération [-]")
    plt.ylabel("Temps de calcul [-]")
    plt.title(f"Temps de calcul en fonction du nombre d'itérations")
    plt.grid()
    plt.legend()
    plt.savefig(f"Figures/tempsFctItr.png", dpi=300)
    plt.show()


def afficher_differences_en_fct_temps(methodes):
    plt.clf()
    for methode in methodes:
        plt.plot(np.log(np.array(methode.temps)), np.array(methode.differences), lw=2, label=methode.methode)
    plt.xlabel("Temps de calcul [ln(s)]")
    plt.ylabel("Erreur [-]")
    plt.title(f"Erreur en fonction du nombre du temps de calcul")
    plt.grid()
    plt.legend()
    plt.savefig(f"Figures/errFctTemps.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    h = 0.1
    cylindre_1 = Cylindre(rayon=10, hauteur=30, h=h)

    mask_150V = np.zeros(cylindre_1.grille.shape)
    mask_150V[int(1 / cylindre_1.h):int(1 / cylindre_1.h) + 1, :] = True

    mask_0V = np.zeros(cylindre_1.grille.shape)
    mask_0V[-1:, :] = True
    mask_0V[:int(1 / cylindre_1.h), :1] = True
    mask_0V[int(1 / cylindre_1.h) + 1:, :1] = True
    mask_0V[:int(1 / cylindre_1.h), -1:] = True
    mask_0V[int(1 / cylindre_1.h) + 1:, -1:] = True

    frontieres = np.ma.array(
        np.where(mask_150V, 150, cylindre_1.grille) + np.where(mask_0V, 0, cylindre_1.grille),
        mask=(mask_150V+mask_0V)
    )

    # Question 2 b
    relax = Relaxation(grille=cylindre_1.grille, frontieres=frontieres, h_par_indice=1, h=h,
                       nom="Carte de chaleur problème 2b")
    relax(-1, verbose=False, affichage=False)
    relax.afficher_carte_de_chaleur()
    relax.afficher_etat()

    # Question 2 c i
    relax_gauss = RelaxationGaussSeidel(grille=cylindre_1.grille, frontieres=frontieres, h=h,
                                        nom="Carte de chaleur problème 2ci")
    relax_gauss(30_000, verbose=False, affichage=False)
    relax_gauss.afficher_carte_de_chaleur()
    relax_gauss.afficher_etat()

    # Question 2 c ii
    sur_relax = SurRelaxation(grille=cylindre_1.grille, frontieres=frontieres, h=h,
                              nom="Carte de chaleur problème 2cii", w=0.3)
    sur_relax(30_000, verbose=False, affichage=False)
    sur_relax.afficher_carte_de_chaleur()
    sur_relax.afficher_etat()

    # Question c 2 iii
    sur_relax_gauss = SurRelaxationGaussSeidel(grille=cylindre_1.grille, frontieres=frontieres, h_par_indice=0.5, h=h,
                                               nom="Carte de chaleur problème 2ciii", w=0.3)
    sur_relax_gauss(30_000, verbose=False, affichage=False)
    sur_relax_gauss.afficher_carte_de_chaleur()
    sur_relax_gauss.afficher_etat()

    methodes_relax = [relax, relax_gauss, sur_relax, sur_relax_gauss]

    afficher_differences_en_fct_iteration(methodes_relax)
    afficher_temps_en_fct_iteration(methodes_relax)
    afficher_differences_en_fct_temps(methodes_relax)



