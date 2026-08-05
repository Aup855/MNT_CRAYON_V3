# Solveur thermique 1D radial d'un crayon combustible nucléaire : v3 (pastille + jeu + gaine)

**Extension du solveur à un crayon combustible complet : trois régions concentriques [ pastille UO₂, jeu hélium, gaine Zircaloy ], reliées par une conductance de contact et une convection en surface externe.**

> Les v1 et v2 ne traitaient qu'un seul matériau (la pastille), avec d'abord une surface à température imposée (v1), puis une surface pilotée par un échange convectif (v2). Ces deux versions ignoraient donc le jeu combustible-gaine et la gaine elle-même.
> Alors que le jeu est, physiquement, la résistance thermique dominante du crayon. La v3 lève cette hypothèse : elle assemble deux maillages (pastille, gaine) reliés par une conductance de contact au niveau du jeu, non maillé.

---

## Sommaire

1. [Présentation du problème](#1-présentation-du-problème)
2. [Hypothèses simplificatrices](#2-hypothèses-simplificatrices)
3. [Méthodologie du code](#3-méthodologie-du-code)
4. [Résultats](#4-résultats)
5. [Points critiques](#5-points-critiques)
6. [Tests](#6-tests)
7. [Conclusion](#7-conclusion)

---

## 1. Présentation du problème

Un crayon combustible réel n'est pas une pastille nue : elle est entourée d'un mince jeu de gaz (hélium) puis d'une gaine métallique (Zircaloy) qui la sépare du fluide caloporteur. 
La chaleur générée par fission doit donc traverser **trois régions physiquement différentes** avant d'atteindre l'eau : conduction dans la pastille (avec source), un saut de température au contact du jeu (mauvais conducteur), puis conduction dans la gaine (sans source) avant la convection finale.

<p align="center">
  <img src="docs/img/schema_probleme_v3.png" width="850" alt="Coupe radiale du crayon combustible : pastille, jeu, gaine, avec convection en surface">
</p>

*Coupe radiale schématique : pastille UO₂ (source de fission, symétrie au centre), jeu hélium non maillé (conductance de contact `h_gap`), gaine Zircaloy (conduction pure, quasi isotherme), puis convection en surface externe (`h`, `T_fluide`).*

L'équation résolue dans chaque région solide reste l'équation de la chaleur radiale :

$$\rho c_p \frac{\partial T}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}\left(r\,k\,\frac{\partial T}{\partial r}\right) + \dot q'''$$

avec `k`, `ρc_p` et `q'''` désormais **propres à chaque matériau** (source uniquement dans la pastille), et deux liaisons non conductives à modéliser :
- **au jeu** : pas d'équation de conduction (le jeu n'est pas maillé), mais une **conductance de contact** `h_gap` qui relie directement le dernier nœud pastille au premier nœud gaine ;
- **en surface externe (`r = R_clad`)** : condition de **Robin**.

---

## 2. Hypothèses simplificatrices

La v3 lève l'hypothèse « monomatériau » de v1/v2 et en garde volontairement plusieurs autres, pour continuer à isoler les changements introduits à chaque étape :

| # | Hypothèse | Portée |
|---|-----------|--------|
| 1 | **Axisymétrie + invariance axiale** | Reprise à l'identique de v1/v2 : problème 1D radial pur, pas de profil de puissance axial. |
| 2 | **Jeu non maillé** | Le jeu (quelques dizaines de microns) est réduit à un unique coefficient d'échange `h_gap` entre deux nœuds, plutôt qu'à des cellules de volumes finis à part entière. |
| 3 | **Propriétés constantes par matériau** | `k_f`, `k_c`, `ρc_p,f`, `ρc_p,c`, `h_gap` indépendants de `T`. |
| 4 | **`h_gap` constant et uniforme** | h_gap reste une donnée d'entrée constante. |
| 5 | **Source volumique uniforme dans la pastille (cas de référence)** | Comme en v1/v2, avec profil `source_f(r)` arbitraire accepté par le solveur ; gaine sans aucune source. |
| 6 | **Pas d'évolution géométrique** | Géométrie figée dans le temps : pas de gonflement, de fissuration, de fermeture ou d'ouverture du jeu avec l'irradiation. |

---

## 3. Méthodologie du code

**Reprise intégrale du cœur v1/v2** : volumes finis vertex-centered, schéma de Crank-Nicolson, matrice tridiagonale assemblée une seule fois dans `__init__` puis résolue en `O(N)` via `scipy.linalg.solve_banded` à chaque pas de temps. Le traitement du nœud central (symétrie qui émerge de la géométrie) et du nœud de surface externe (Robin) sont conservés à l'identique.

**Ce qui change : un maillage global par concaténation.** Le maillage pastille (`N_f` nœuds, de `r=0` à `R_f`) et le maillage gaine (`N_c` nœuds, de `R_f+e_gap` à `R_clad`) sont construits séparément puis concaténés en un seul tableau de nœuds. Le premier nœud de gaine est positionné directement à `R_f + e_gap` : le jeu est géométriquement « sauté », sans qu'aucun nœud n'y soit placé. Chaque nœud porte son propre `ρc_p` (tableau indexé par matériau), et les deux nœuds encadrant le jeu (dernier nœud pastille, premier nœud gaine) portent chacun un volume de demi-coquille : la même formule déjà validée en v1/v2 pour le nœud de bord, appliquée ici deux fois.

**L'idée centrale : le jeu n'est qu'une conductance de plus.** Le tableau `aE` de conductances, jusque-là uniquement fait de termes de conduction `k·A/Δr`, reçoit un troisième type de terme : une conductance de contact unique, `h_gap · A_gap`, insérée à l'indice `N_f-1` (avec `A_gap = 2π(R_f + e_gap/2)`, l'aire prise au rayon moyen du jeu). Conséquence directe : la boucle d'assemblage des nœuds intérieurs, écrite pour v1/v2 et reprise sans aucune modification, s'applique telle quelle au nœud d'interface pastille/gaine. Ce nœud ne « sait » pas qu'il est à la frontière de deux matériaux. Il ne voit qu'une conductance à gauche et une conductance à droite, exactement comme n'importe quel autre nœud intérieur.

**API** : `CrayonV3(alpha_f, k_f, R_f, N_f, alpha_c, k_c, e_gap, e_clad, N_c, h_gap, h_conv, T_fluide, dt, T_init, source_f)`, puis `.step()` / `.solve(n_steps)`. Diagnostics dédiés à la structure multicouche : `T_surface_pastille()`, `T_interne_gaine()`, `T_surface_gaine()`, `flux_sortant_frontiere()`.

---

## 4. Résultats

Cas de validation, valeurs représentatives d'un crayon REP (`R_f = 4,1 mm`, `e_gap = 80 µm`, `e_clad = 0,57 mm`, `k_f = 3 W/(m·K)`, `k_c = 15 W/(m·K)`, `h_gap = 10⁴ W/(m²·K)`, `h_conv = 3,5×10⁴ W/(m²·K)`, `q''' = 3,8×10⁸ W/m³`, `T_fluide = 310 °C`). En régime permanent, la solution analytique exacte comporte quatre régimes raccordés (parabole dans la pastille, saut au jeu, logarithme dans la gaine, saut convectif en surface) :

<p align="center">
  <img src="docs/img/validation_v3.png" width="620" alt="Profil radial complet : parabole pastille, saut au jeu, logarithme gaine, saut convectif">
</p>

| Grandeur | Numérique | Analytique | Écart |
|---|---|---|---|
| Température au centre | 965,89 °C | 965,89 °C | 9,3 × 10⁻⁵ °C |
| Température surface pastille | 433,58 °C | 433,58 °C | 9,3 × 10⁻⁵ °C |
| Température interne gaine | 356,43 °C | 356,43 °C | 9,3 × 10⁻⁵ °C |
| Température surface gaine | 329,21 °C | 329,21 °C | 5,7 × 10⁻¹⁴ °C |

Deux résultats physiques à retenir directement du profil : le **saut au jeu domine largement** (77,15 °C) alors qu'il est presque sept fois plus mince que la gaine, dont le saut ne vaut que 27,22 °C. C'est la signature thermique typique d'un crayon REP réel, où le gaz d'hélium est la résistance thermique dominante malgré son épaisseur dérisoire.

---

## 5. Points critiques

Ce que cette v3 **ne** capture **pas**, en plus des limites déjà présentes en v1/v2 (propriétés constantes, pas de couplage 2D/3D, pas d'irradiation) :

- **`h_gap` fixé et donné, pas calculé.** En réalité, la conductance de contact du jeu dépend de la pression du gaz, de l'ouverture mécanique (elle-même pilotée par le gonflement du combustible et le fluage de la gaine) et du taux de combustion : ici c'est une constante d'entrée figée, alors qu'elle évolue fortement sur la vie du crayon.
- **`k_f` et `k_c` toujours constants.** Comme en v1/v2 : k_f est traité comme constant, alors que la conductivité de l'UO₂ diminue nettement quand la température augmente. Le modèle sous-estime donc l'écart de température entre le centre et la surface de la pastille
- **Jeu réduit à un coefficient unique, sans épaisseur ni inertie thermique propre.** Le gaz du jeu a une capacité thermique non nulle, ignorée ici puisqu'aucune cellule ne lui est associée : en transitoire rapide, ce n'est pas rigoureusement neutre.
- **Toujours 1D radial pur.** Aucun profil de puissance axial, aucun couplage neutronique : la géométrie reste un problème plan à `z` fixé.
---

## 6. Tests

Suite de tests (`pytest test_crayon_v3.py -v`), reprenant les tests hérités de v1/v2 (stabilité, isotherme sans source, centre maximum) et ajoutant les tests spécifiques à la structure multicouche :

| Test | Ce qu'il protège |
|---|---|
| `test_saut_jeu_plus_grand_que_saut_gaine` | Bon sens physique : le jeu doit dominer la résistance thermique, pas l'inverse. |
| `test_solution_analytique_complete` | Validation quantitative des 4 zones (parabole, 2 sauts, logarithme) simultanément, à 10⁻³ °C près. |
| `test_bilan_energie_regime_permanent` | Conservation de l'énergie à travers un système hétérogène à 2 matériaux et 1 interface de contact. |
| `test_gaine_quasi_isotherme` | Cohérence d'ordre de grandeur : la gaine (bon conducteur, fine) ne doit produire qu'un saut de quelques dizaines de degrés, pas des centaines. |
| `test_conductance_interface_dans_le_tableau_aE` | Vérifie que l'assemblage insère bien UNE SEULE conductance de contact au bon indice (`N_f-1`) du tableau `aE`, distincte des conductances de conduction voisines. |
| `test_centre_maximum_et_monotone` | Le centre reste le point le plus chaud, et la température décroît du centre vers la surface, y compris à travers les deux sauts. |
| `test_pas_de_singularite_au_centre` | Aucun NaN/Inf au nœud central, pour plusieurs résolutions de maillage pastille. |
| `test_stabilite_plusieurs_maillages` *(×3)* | Robustesse sur 3 couples de résolutions pastille/gaine. |
| `test_isotherme_sans_source` | Rétrocompatibilité : sans source et à `T_fluide` partout, rien ne doit bouger. |
| `test_stabilite_bornee` | Stabilité inconditionnelle de Crank-Nicolson même avec `dt` agressif. |

---

## 7. Conclusion

Cette v3 démontre que l'architecture en conductances génériques, posée dès la v2, absorbe un changement structurel bien plus profond qu'une simple condition limite : l'ajout d'une interface entre deux matériaux et d'un mode de liaison entièrement nouveau (le contact, plutôt que la conduction ou la convection) ne nécessite **aucune modification** de la boucle d'assemblage des nœuds intérieurs. Seuls le maillage global (concaténation) et le tableau `aE` (insertion d'un terme de contact) évoluent. La validation à 4 zones, conservée à mieux que 10⁻³ °C sur l'ensemble du profil et à la précision machine sur le bilan de puissance, confirme que malgré cette complexité ajoutée, la précision du solveur n'a pas été dégradée.
