from bs4 import BeautifulSoup
import requests
import re
import csv
import time

NO_RESULT = "-"

FILE_NAME = "annonces_idf.csv"
DELAY_TIME = 0.1
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
    if int(content) >= PRIX_MINIMUM:
        return content
    raise NonValide(content)

def ville(soup):
    content = soup.find('h2', class_="mt-0").text
    last_comma_index = content.rfind(", ")
    return content[last_comma_index + 2:]

def type(soup):
    content = soup.find_all('li')

    for li in content:
        if "Type" in li.text:
            content = li.text[4:]
            if content == "Maison" or content == "Appartement":
                return  content
            raise NonValide
    return NO_RESULT

def surface(soup):
    content = soup.find_all('li')

    for li in content:
        if "Surface" in li.text:
            match = re.search(r'\d+', li.text)  # Recherche la première séquence de chiffres
            if match:
                return int(match.group())
    return NO_RESULT


def nbrpieces(soup):
    content = soup.find_all('li')

    for li in content:
        if "Nb. de pièces" in li.text:
            return li.text[-1]

    return NO_RESULT


def nbrchambres(soup):
    content = soup.find_all('li')

    for li in content:
        if "Nb. de chambres" in li.text:
            return li.text[-1]

    return NO_RESULT


def nbrsdb(soup):
    content = soup.find_all('li')

    for li in content:
        if "Nb. de sales de bains" in li.text:
            return li.text[-1]

    return NO_RESULT


def dpe(soup):
    content = soup.find_all('li')

    for li in content:
        if "Consommation d'énergie (DPE)" in li.text:
            match = re.search(r'\b([A-G])\b', li.text)
            if match:
                return match.group(1)

    return NO_RESULT


def informations(soup):
    try:
        ville_data = ville(soup)
        type_data = type(soup)
        surface_data = surface(soup)
        nbr_pieces_data = nbrpieces(soup)
        nbr_chambres_data = nbrchambres(soup)
        nbr_sdb_data = nbrsdb(soup)
        dpe_data = dpe(soup)
        prix_data = prix(soup)

        result = f"{ville_data},{type_data},{surface_data},{nbr_pieces_data},{nbr_chambres_data},{nbr_sdb_data},{dpe_data},{prix_data}"
        return result

    except NonValide as e:
        raise e


def write_to_csv(soup):

    headers = ["Ville", "Type", "Surface", "NbrPieces", "NbrChambres", "NbrSdb", "DPE", "Prix"]
    filename = "annonces.csv"
    mode = "a"

    try:
        # Vérifie si le fichier existe on le creer sinon
        file_empty = True
        try:
            with open(FILE_NAME, 'r') as f:
                file_empty = f.readline() == ''
        except FileNotFoundError:
            file_empty = True

        with open(FILE_NAME, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            if file_empty or mode == 'w':
                writer.writerow(headers)

            try:
                info_str = informations(soup)
                info_list = info_str.split(',')
                writer.writerow(info_list)
                return True

            except NonValide:
                return False

    except Exception as e:
        print(f"Erreur lors de l'écriture dans le CSV: {str(e)}")
        return False


def get_all_annonces_links(base_url):
    annonces_links = []
    page = 1

    while True:
        # On construit l'URL de la page actuel
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}/{page}"

        try:
            # Récupere la page
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Fin des pages à la page {page - 1}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            annonces = soup.find_all('div', class_='row product shadow p-2 rounded-3')

            if not annonces:
                print(f"Plus d'annonces à la page {page}")
                break

            # Pour chaque annonce, trouve le lien correspondant
            for annonce in annonces:
                link = annonce.find('a', href=True)
                if link:
                    full_link = f"https://www.immo-entre-particuliers.com{link['href']}"
                    annonces_links.append(full_link)

            print(f"Page {page} traitée : {len(annonces)} annonces trouvées")
            page += 1

            time.sleep(DELAY_TIME)

        except Exception as e:
            print(f"Erreur lors du traitement de la page {page}: {str(e)}")
            break

    return annonces_links


def scrape_and_save_annonces():

    base_url = "https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente/ta-offer"

    print("Recuperation des liens d'annonces :")
    links = get_all_annonces_links(base_url)
    print(f"Total de {len(links)} liens d'annonces trouvés")

    write_to_csv(None)

    # Puis on traite chaque annonce
    valid_count = 0
    for i, link in enumerate(links, 1):
        try:
            print(f"Traitement de l'annonce {i}/{len(links)}: {link}")
            soup = getSoup(link)
            if write_to_csv(soup):
                valid_count += 1

            time.sleep(DELAY_TIME)

        except Exception as e:
            print(f"Erreur lors du traitement de l'annonce {link}: {str(e)}")
            continue

    print(f"Total des annonces valides sauvegardes : {valid_count}")

soup = getSoup("https://www.immo-entre-particuliers.com/annonce-val-de-marne-lhay-les-roses/407514-belle-maison-familiale-au-calme")
# print("Prix:", prix(soup))
# print("Ville:", ville(soup))
# print("Type:", type(soup))
# print("Surface:", surface(soup))
# print("Nbr Pieces:", nbrpieces(soup))
# print("Nbr Chambre:", nbrchambres(soup))
# print("Nbr Salle de bain:", nbrsdb(soup))
# print("DPE:", dpe(soup))
# print("Information:", informations(soup))


scrape_and_save_annonces()
