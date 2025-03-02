import pandas as pd
import unicodedata
import re

# Question 8
def charger_donnees():
    annonces = pd.read_csv('annonces_idf.csv')
    print(f"{len(annonces)} annonces chargées.")
    print("Aperçu des données:")
    print(annonces.head())
    print("\nInformations sur les colonnes:")
    print(annonces.info())
    print("\nStatistiques descriptives:")
    print(annonces.describe())
    return annonces

def normaliser_texte(texte):
    texte = texte.lower()
    texte = ''.join(c for c in unicodedata.normalize('NFD', texte) if not unicodedata.combining(c))
    texte = re.sub(r'[^a-z0-9]', '', texte)
    return texte

# Question 9
def remplacer_dpe_manquant(annonces):
    annonces['DPE'] = annonces['DPE'].replace('-', 'Vierge')
    return annonces

# Question 10
def remplacer_valeurs_numeriques_manquantes(annonces):
    colonnes_numeriques = ['Surface', 'NbrPieces', 'NbrChambres', 'NbrSdb']
    for col in colonnes_numeriques:
        annonces[col] = pd.to_numeric(annonces[col], errors='coerce')
        annonces[col] = annonces[col].fillna(annonces[col].mean())
    return annonces

# Question 11
def creer_variables_indicatrices(annonces):
    dpe_dummies = pd.get_dummies(annonces['DPE'], prefix='DPE')
    type_dummies = pd.get_dummies(annonces['Type'], prefix='Type')
    annonces = pd.concat([annonces, dpe_dummies, type_dummies], axis=1)
    annonces = annonces.drop(['DPE', 'Type'], axis=1)
    return annonces

# Question 12
def charger_donnees_villes():
    try:
        return pd.read_csv('cities.csv', delimiter=',')
    except Exception as e:
        print(f"Erreur lors du chargement du fichier cities.csv: {e}")
        return None

# Question 13 et 14
def ajouter_coordonnees_geographiques(annonces, villes):
    col_ville = 'label' if 'label' in villes.columns else villes.columns[0]
    annonces['ville_norm'] = annonces['Ville'].apply(normaliser_texte)
    villes['ville_norm'] = villes[col_ville].apply(normaliser_texte)
    col_lat, col_lon = None, None

    for col in villes.columns:
        if 'lat' in col.lower():
            col_lat = col
        elif 'lon' in col.lower():
            col_lon = col
    if not col_lat or not col_lon:
        return annonces

    # print(f"Colonnes - Ville: {col_ville}, Latitude: {col_lat}, Longitude: {col_lon}")
    villes[col_lat] = pd.to_numeric(villes[col_lat], errors='coerce')
    villes[col_lon] = pd.to_numeric(villes[col_lon], errors='coerce')
    annonces_avec_coord = pd.merge(annonces, villes[['ville_norm', col_lat, col_lon]], on='ville_norm', how='left')
    annonces_final = annonces_avec_coord.drop(['Ville', 'ville_norm'], axis=1)
    return annonces_final

def executer_nettoyage_data():
    annonces = charger_donnees()                                            # Question 8
    annonces = remplacer_dpe_manquant(annonces)                             # Question 9
    annonces = remplacer_valeurs_numeriques_manquantes(annonces)            # Question 10
    annonces = creer_variables_indicatrices(annonces)                       # Question 11
    villes = charger_donnees_villes()                                       # Question 12
    annonces_final = ajouter_coordonnees_geographiques(annonces, villes)    # Question 13 et 14

    if 'Type_-' in annonces_final.columns:
        annonces_final = annonces_final.drop(columns=['Type_-'])
    annonces_final.to_csv("annonces_nettoyees.csv", index=False)
    print(f"Nombre de colonnes après nettoyage: {len(annonces_final.columns)}")
    return annonces_final

executer_nettoyage_data()
