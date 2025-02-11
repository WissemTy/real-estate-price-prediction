from bs4 import BeautifulSoup
import requests
import re

PRIX_MINIMUM = 10000

class NonValide (Exception):

    def __str__(self):
        return f"Critères invalides"

def getSoup(page):
    # Prend en entrée l’URL d’une annonce, et renvoie la soupe correspondant à cette page HTML.
    html = requests.get(page).text
    return BeautifulSoup(html, 'html.parser')

def prix(soup):
    content = soup.find('p', class_="product-price fs-3 fw-bold").text
    content = content.replace('€', '').replace(' ', '') # Pour enlever le '€' et les espaces
    if (int(content) >= PRIX_MINIMUM):
        return content
    raise NonValide(content)

def ville(soup):
    content = soup.find('h2', class_="mt-0").text
    last_comma_index = content.rfind(", ")
    return content[last_comma_index + 2:]
'''
def type(soup):
    # Uniquement maison ou appartement
    content = soup.find('li', class_="col-12 d-flex justify-content-between").text
    if content[4:] == "Maison" or content[4:] == "Appartement":
        return content[4:]
    raise NonValide(content) '''

def type(soup):
    content = soup.find_all('li')

    for li in content:
        if "Type" in li.text:
            content = li.text[4:]
            if content == "Maison" or content == "Appartement":
                return  content
            raise NonValide
    return "Pas de Type trouvé"

def surface(soup):
    content = soup.find_all('li')

    for li in content:
        if "Surface" in li.text:
            match = re.search(r'\d+', li.text)  # Recherche la première séquence de chiffres
            if match:
                return int(match.group())
    return "Pas de surface trouvé"


def nbrpieces(soup):
    content = soup.find_all('li')

    for li in content:
        if "Nb. de pièces" in li.text:
            return li.text[-1]

    return "Pas de pièce trouvé"


def nbrchambres(soup):
    content = soup.find_all('li')

    for li in content:
        if "Nb. de chambres" in li.text:
            return li.text[-1]

    return "Pas de chambre trouvé"


def nbrsdb(soup):
    content = soup.find_all('li')

    for li in content:
        if "Nb. de sales de bains" in li.text:
            return li.text[-1]

    return "Pas de salle de bains trouvé"


def dpe(soup):
    content = soup.find_all('li')

    for li in content:
        if "Consommation d'énergie (DPE)" in li.text:
            match = re.search(r'\b([A-G])\b', li.text)
            if match:
                return match.group(1)

    return None


def informations(soup):
    try:
        # Extraire les différentes informations
        ville_info = ville(soup)
        type_info = type(soup)
        surface_info = surface(soup)
        nbr_pieces_info = nbrpieces(soup)
        nbr_chambres_info = nbrchambres(soup)
        nbr_sdb_info = nbrsdb(soup)
        dpe_info = dpe(soup)
        prix_info = prix(soup)

        # Combiner toutes les informations dans une chaîne de caractères
        result = f"{ville_info},{type_info},{surface_info},{nbr_pieces_info},{nbr_chambres_info},{nbr_sdb_info},{dpe_info},{prix_info}"
        return result

    except NonValide as e:
        raise e

# https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente annonce idf
# soupTestPrix = getSoup("https://www.immo-entre-particuliers.com/annonce-val-de-marne-le-kremlin-bicetre/377378-echange-un-grand-t3-de-71m2")
# print(prix(soupTestPrix))
soup =getSoup('https://www.immo-entre-particuliers.com/annonce-maroc/409215-terrain-commercial-titre-deux-facades-114m') #terrains
#soup = getSoup("https://www.immo-entre-particuliers.com/annonce-val-de-marne-lhay-les-roses/407514-belle-maison-familiale-au-calme")
#soup =getSoup("https://www.immo-entre-particuliers.com/annonce-isere-roussillon/409282-appartement-3-pieces-85m") #appartement
print("Prix:", prix(soup))
print("Ville:", ville(soup))
#print("Type:", type(soup))
print("Surface:", surface(soup))
print("Nbr Pieces:", nbrpieces(soup))
print("Nbr Chambre:", nbrchambres(soup))
print("Nbr Salle de bain:", nbrsdb(soup))
print("DPE:", dpe(soup))

print("Information:", informations(soup))


