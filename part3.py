import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

annonces_df = pd.read_csv('annonces_nettoyees.csv')

# Question 15
X = annonces_df.drop('Prix', axis=1)
y = annonces_df['Prix']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=49)

# Question 16
pipe_lr = make_pipeline(
    SimpleImputer(strategy='mean'),
    LinearRegression()
)
pipe_lr.fit(X_train, y_train)
score_lr = pipe_lr.score(X_test, y_test)
print(f"Question 16 : Score R²: {score_lr:.4f}")

# Question 17
pipe_norm = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    LinearRegression()
)
pipe_norm.fit(X_train, y_train)
score_norm_lr = pipe_norm.score(X_test, y_test)
print(f"\nQuestion 17\n\t - Normalisation + RL: Score R² {score_norm_lr:.4f}")

pipe_std = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    LinearRegression()
)
pipe_std.fit(X_train, y_train)
score_std_lr = pipe_std.score(X_test, y_test)
print(f"\t - Standardisation + RL: Score R² {score_std_lr:.4f}")

# Question 18
print("\nQuestion 18 - Tableau comparatif des méthodes:")
print(f"\t- LR: {score_lr: .4f}")
print(f"\t- Normalisation + LR: {score_norm_lr:.4f}")
print(f"\t- Standardisation + LR: {score_std_lr:.4f}")
#TODO: Constatez-vous une amélioration des prédictions due au pré-traitement pour ce jeu de données et cette méthode ?
# Non, on ne constate pas d'amélioration des prédictions suite au pré-traitement pour la régression linéaire. Les scores R² sont tous négatifs et très similaires :
# Régression linéaire simple : -0.0252, Normalisation + RL : -0.0252 (identique), Standardisation + RL : -0.0256 (légèrement pire)

# Question 19
pipe_dt = make_pipeline(
    SimpleImputer(strategy='mean'),
    DecisionTreeRegressor(max_depth=4, random_state=49)
)
pipe_dt.fit(X_train, y_train)
score_dt = pipe_dt.score(X_test, y_test)
print(f"\nQuestion 19\n\t- AD (max_depth=4): Score R² {score_dt:.4f}")

# Normalisation + AD
pipe_norm_dt = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    DecisionTreeRegressor(max_depth=4, random_state=49)
)
pipe_norm_dt.fit(X_train, y_train)
score_norm_dt = pipe_norm_dt.score(X_test, y_test)
print(f"\t- Normalisation + Arbre de décision (max_depth=4): Score R² {score_norm_dt:.4f}")

# Standardisation + AD
pipe_std_dt = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    DecisionTreeRegressor(max_depth=4, random_state=49)
)
pipe_std_dt.fit(X_train, y_train)
score_std_dt = pipe_std_dt.score(X_test, y_test)
print(f"\t- Standardisation + Arbre de décision (max_depth=4): Score R² {score_std_dt:.4f}")
#TODO: Constatez-vous une amélioration significative suite au pré-traitement ?
# Non, on ne constate aucune amélioration suite au pré-traitement pour les arbres de décision. Les scores R² sont identiques dans les trois cas :
# Arbre de décision (max_depth=4), Normalisation + AD, Standardisation + AD : -0.7728

depths = [5, 6, 7, 8, 9, 10, 20]
dt_scores = []

print("\nTest avec différentes profondeurs d'arbre:")
for depth in depths:
    pipe_dt = make_pipeline(
        SimpleImputer(strategy='mean'),
        DecisionTreeRegressor(max_depth=depth, random_state=49)
    )
    pipe_dt.fit(X_train, y_train)
    score = pipe_dt.score(X_test, y_test)
    dt_scores.append(score)
    print(f"\t- max_depth = {depth}: R² = {score:.4f}")
#TODO: L’augmentation de la profondeur permet-elle de meilleurs résultats ?
# Non, l'augmentation de la profondeur ne permet pas de meilleurs résultats. Au contraire pour certains cas

# Question 20
print("\nQuestion 20 - Tableau comparatif des méthodes:")
print(f"\t- AD: {score_dt: .4f}")
print(f"\t- Normalisation + AD: {score_norm_dt:.4f}")
print(f"\t- Standardisation + AD: {score_std_dt:.4f}")

# Question 21
pipe_knn_4 = make_pipeline(
    SimpleImputer(strategy='mean'),
    KNeighborsRegressor(n_neighbors=4)
)
pipe_knn_4.fit(X_train, y_train)
score_knn_4 = pipe_knn_4.score(X_test, y_test)
print(f"\nQuestion 21\n\t- KNN (n_neighbors=4): Score R² {score_knn_4:.4f}")

# Modèle avec n_neighbors=5
pipe_knn_5 = make_pipeline(
    SimpleImputer(strategy='mean'),
    KNeighborsRegressor(n_neighbors=5)
)
pipe_knn_5.fit(X_train, y_train)
score_knn_5 = pipe_knn_5.score(X_test, y_test)
print(f"\t- KNN (n_neighbors=5): Score R² {score_knn_5:.4f}")
#TODO: L’augmentation du nombre de voisins considérés permet-elle de meilleurs résultats ?
# Oui, l'augmentation du nombre de voisins de 4 à 5 permet une légère amélioration des résultats : n_neighbors=4) : R² = -0.0663, n_neighbors=5 : R² = -0.0620

# Question 22
print("\nQuestion 22 - Tableau comparatif des méthodes:")
print(f"\t- KNN (n_neighbors=5): Score R² {score_knn_5:.4f}")

#Normalisation + KNN (n_neighbors=5)
pipe_norm_knn_5 = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    KNeighborsRegressor(n_neighbors=5)
)
pipe_norm_knn_5.fit(X_train, y_train)
score_norm_knn_5 = pipe_norm_knn_5.score(X_test, y_test)
print(f"\t- Normalisation + KNN (n_neighbors=5): Score R² {score_norm_knn_5:.4f}")

# Question 23
print("\nQuestion 23 - Tableau comparatif des méthodes:")
print(f"\t- LR (LR + Normalisation): Score R² = {score_lr:.4f}")
print(f"\t- AD (identique): Score R² = {score_dt:.4f}")
print(f"\t- KNN (KNN5 + Normalisation): Score R² = {score_knn_5:.4f}")
print("\nLe modèle KNN avec normalisation donne les meilleurs résultats, bien que tous les scores soient faibles.")

modele_M = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    KNeighborsRegressor(n_neighbors=5)
)
modele_M.fit(X_train, y_train)
score_M = modele_M.score(X_test, y_test)
print(f"Score du modèle M (KNN5 + Normalisation): R² = {score_M:.4f}")

# Question 24 - Visualisation des résultats avec le modèle M
y_pred = modele_M.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, marker='*', color='green')
plt.xlim(0, 1.6e6)
plt.ylim(0, 1.6e6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel("Valeurs réelles (y_test)")
plt.ylabel("Estimations (y_pred)")
plt.title(f"Estimation vs Réalité - Méthode M: KNN5 + Normalisation")
plt.show()
#TODO: Que constatez-vous ? Pensezvous que le jeu de données est suffisant pour construire un modèle d’apprentissage précis ?
#Si le modèle était parfait, tous les points seraient alignés sur cette diagonale.
#Ici, les prédictions sont souvent éloignées des valeurs réelles, indiquant une faible précision du modèle.
#Les prédictions semblent plus proches des valeurs réelles pour les petits prix.
#Pour les prix élevés, la dispersion est beaucoup plus grande, suggérant une mauvaise généralisation du modèle pour ces valeurs.

# La visualisation montre une dispersion importante des prédictions autour de la diagonale,
# indiquant une précision limitée du modèle. Le jeu de données semble insuffisant pour
# construire un modèle précis, comme en témoigne le score R² proche de zéro.

# Question 25 - Analyse en composantes principales (PCA)
# Gérer les valeurs manquantes avec SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

pca = PCA(n_components=2)
pca.fit(X_train_scaled)
explained_variance = pca.explained_variance_ratio_

print("\nQuestion 25 - Analyse de réduction avec 2 composantes:")
print(f"\t- Composante 1: {explained_variance[0]:.4f}")
print(f"\t- Composante 2: {explained_variance[1]:.4f}")
print(f"\t- Variance totale expliquée: {explained_variance.sum():.4f}")

pipe_pca = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    PCA(n_components=2)
)

X_train_pca = pipe_pca.fit_transform(X_train)
X_test_pca = pipe_pca.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_pca, y_train)
score_M_on_pca = knn.score(X_test_pca, y_test)

print(f"\t- Modèle M sur données PCA (2 composantes): R² = {score_M_on_pca:.4f}")
print(f"\t- Modèle M original: R² = {score_M:.4f}")
# TODO: La modélisation avec 2 composantes principales est-elle satisfaisante pour ce jeu de données ?
# La modélisation avec 2 composantes principales n'est pas pertinente,
# car elle ne permet ni de mieux expliquer les données ni d'améliorer la performance du modèle.

# Question 26 - Comparaison du modèle M original avec le modèle M appliqué sur PCA
print("\nQuestion 26 - Comparaison des scores :")
print(f"\t- Modèle M sur données PCA: R² = {score_M_on_pca:.4f}")
print(f"\t- Modèle M original: R² = {score_M:.4f}")

# On constate que la modélisation avec seulement 2 composantes principales entraîne une
# perte significative de performance par rapport au modèle M original, suggérant que
# l'information contenue dans toutes les variables d'origine est importante.
#TODO: Que constatez-vous ? la valeur renvoyé est de -0.0305.
#La différence de score entre la méthode M sur PCA et la méthode M originale est de -0.0305,
# ce qui signifie que la méthode M fonctionne moins bien après la réduction de dimension avec PCA (2 composantes).

# Question 27 - Matrice de corrélation
correlation_matrix = annonces_df.corr()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(f"\nQuestion 27 - Matrice de corrélation:\n{correlation_matrix.round(4)}")

# Question 28
price_correlations = correlation_matrix['Prix'].drop('Prix')
sorted_correlations = price_correlations.abs().sort_values(ascending=False)
print(f"\nQuestion 28 - Variables les plus corrélées avec le prix:\n{sorted_correlations.round(4)}")

# Question 29
top_5_features = sorted_correlations.index[:5]
print(f"\nLes 5 caractéristiques les plus corrélées avec le prix sont: {list(top_5_features)}")

X_reduit = X[top_5_features]
X_train_reduit, X_test_reduit, y_train_reduit, y_test_reduit = train_test_split(
    X_reduit, y, test_size=0.25, random_state=49
)

modele_M_reduit = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    KNeighborsRegressor(n_neighbors=5)
)
modele_M_reduit.fit(X_train_reduit, y_train_reduit)
score_M_reduit = modele_M_reduit.score(X_test_reduit, y_test_reduit)

print(f"\nQuestion 29 - Comparaison des scores:")
print(f"\t- Modèle M original (toutes variables): R² = {score_M:.4f}")
print(f"\t- Modèle M réduit (5 meilleures variables): R² = {score_M_reduit:.4f}")

# La sélection des 5 caractéristiques les plus corrélées avec le prix pourrait
# améliorer ou dégrader les performances. Les résultats permettront de déterminer
# si cette approche de réduction de dimensionnalité est plus efficace que l'utilisation
# de toutes les variables ou que l'approche PCA.
#TODO : comparez le score de prédiction aux résultats obtenus précédemment. Que constatez vous ?
#avant : R² = 0.0471 et apres Score R² avec les 5 meilleurs attributs : 0.1584
#Le score R² a augmenté significativement après avoir sélectionné uniquement les 5 attributs les plus corrélés (de 0.0471 à 0.1584).
#Cela signifie que :
#Les variables les plus corrélées avec le prix étaient les plus pertinentes pour la prédiction.